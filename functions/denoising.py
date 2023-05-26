import torch
from tqdm import tqdm
import torchvision.utils as tvu

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def sample_gibbsddrm(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, etaD, config, cls_fn=None, classes=None):

    enable_Hupdate = False
    y_0s = []
    kernels = []

    with torch.no_grad():

        bsz = y_0.shape[0]

        U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, x, seq, b,  y_0, sigma_0, y_0s)

        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        max_steps = seq.stop
        cnt = 0
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):

            xt = xs[-1].to('cuda')
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            x0_t, et = est_x0_t(model, xt, t, at)

            if i  < int(max_steps * 0.7):
                U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, x, seq, b,  y_0, sigma_0, y_0s, x0_t)
            else:
                U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, x, seq, b,  y_0, sigma_0, y_0s)

            xt_next = update_x(x, H_funcs, U_t_y, sigma_0, singulars, Sigma, Sig_inv_U_t_y, xt, x0_t, et, t, at, at_next, cnt, etaA, etaB, etaC, etaD)

            # Hupdate
            enable_Hupdate = (i <= max_steps * config.deblur.Hupdate_start)
            if enable_Hupdate:

                for i_Hupdate in range(config.deblur.iter_Hupdate):
                    if i == seq.start:
                        continue
                    
                    x0_t_next, et_next = est_x0_t(model, xt_next, next_t, at_next)

                    if config.deblur.alg_Hupdate == "optim":
                        H_funcs.update_H_optim(y_0, x0_t_next, n_iter=config.deblur.iter_optim, lr=float(config.deblur.lr_Hupdate), \
                            reg_H_gamma=config.deblur.reg_H_gamma, reg_H_type = config.deblur.reg_H_type)
                    elif config.deblur.alg_Hupdate == "langevin": # linear operator's parameter update of GibbsDDRM
                        H_funcs.update_H_langevin(y_0, x0_t_next, n_iter=config.deblur.iter_optim, lr=float(config.deblur.lr_Hupdate), \
                            reg_H_gamma=config.deblur.reg_H_gamma, reg_H_type = config.deblur.reg_H_type)

                    if i_Hupdate == (config.deblur.iter_Hupdate - 1) and config.deblur.resample_after_Hupdate is False:
                        continue

                    U_t_y, singulars, Sigma, Sig_inv_U_t_y, init_y = calc_vars_for_xupdate(H_funcs, x, seq, b,  y_0, sigma_0, y_0s, x0_t)                    
                    xt_next = update_x(x, H_funcs, U_t_y, sigma_0, singulars, Sigma, Sig_inv_U_t_y, xt, x0_t, et, t, at, at_next, cnt, etaA, etaB, etaC, etaD)
                    

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))
            kernels.append(H_funcs.kernel.detach().clone().to('cpu'))

            cnt += 1

    return xs, x0_preds, y_0s, kernels

def calc_vars_for_xupdate(H_funcs, x, seq, b,  y_0, sigma_0, y_0s, x_0 = None):
    """
        calculate variables that are used when samling x_t.
        As these variables depends on linear operator's parameters (phi), 
        they must be updated after the update of phi.

    """
    bsz = y_0.shape[0]
    
    if H_funcs.conv_shape == "same_interp":
        if x_0 is not None:
            y_0_interp = H_funcs.interp_y_0(y_0, x_0, sigma_0)
        else:
            y_0_interp = H_funcs.interp_y_0(y_0, y_0, sigma_0)
        U_t_y = H_funcs.Ut(y_0_interp)
        y_0s.append(y_0_interp)
    else:
        U_t_y = H_funcs.Ut(y_0)
    _dim_all_singulars = U_t_y.view(U_t_y.shape[0], -1).shape[1]

    #setup vectors used in the algorithm
    singulars = H_funcs.singulars()
    Sigma = torch.zeros(bsz, _dim_all_singulars, device=x.device)
    Sigma[:, :singulars.shape[-1]] = singulars
    Sig_inv_U_t_y = U_t_y / singulars[:, :U_t_y.shape[-1]]

    #initialize x_T as given in the paper
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
    large_singulars_index = torch.where((singulars * largest_sigmas[:, 0, 0, 0][:, None]) > sigma_0)
    inv_singulars_and_zero = torch.zeros(singulars.shape).to(singulars.device)
    inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
    inv_singulars_and_zero = inv_singulars_and_zero.view(bsz, -1)     

    # implement p(x_T | x_0, y) as given in the paper
    #   if eigenvalue is too small, we just treat it as zero (only for init) 
    init_y = torch.zeros(x.shape[0], singulars.shape[-1], dtype=U_t_y.dtype).to(x.device)
    init_y[large_singulars_index] = U_t_y[large_singulars_index] / singulars[large_singulars_index]
    remaining_s = (largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2)
    remaining_s = remaining_s.clamp_min(0.0).sqrt()
    V_t_x_init = H_funcs.Vt(x)
    init_y = init_y + remaining_s * V_t_x_init
    init_y = init_y / largest_sigmas.view(largest_sigmas.shape[0], -1)
    return U_t_y, singulars, Sigma, Sig_inv_U_t_y,init_y

def update_x(x, H_funcs, U_t_y, sigma_0, singulars, Sigma, Sig_inv_U_t_y, xt, x0_t, et, t, at, at_next, cnt, etaA, etaB, etaC, etaD):

    """
        perform the modified DDRM steps defined in Eq. (9) in the paper, 
        x_t is sampled from p_theta (x_t | x_{t+1}, phi, y)

        Returns:
            xt_next : the sampled x_t
        
    """

    bsz = x.shape[0]
    #variational inference conditioned on y
    sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
    sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
    xt_mod = xt / at.sqrt()[0, 0, 0, 0]
    V_t_x = H_funcs.Vt(xt_mod)
    SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
    V_t_x0 = H_funcs.Vt(x0_t)
    SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

    falses = torch.zeros(bsz, V_t_x0.shape[1] - singulars.shape[-1], dtype=torch.bool, device=xt.device)

    cond_before_lite = (singulars * sigma_next > sigma_0) * (singulars > 1e-10)
    cond_after_lite =  (singulars * sigma_next < sigma_0) * (singulars > 1e-10)

    cond_before = torch.hstack((cond_before_lite, falses))
    cond_after  = torch.hstack((cond_after_lite, falses))

    std_nextD = sigma_next * etaD
    sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextD ** 2)

    std_nextA = sigma_next * etaA
    sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)

    diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

    #missing pixels        
    Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextD * H_funcs.Vt(torch.randn_like(x))

    #less noisy than y (after)
    coef_A = sigma_next * etaA
    coef_C = sigma_next * etaC

    update_A = H_funcs.Vt(et) * cond_after_lite
    update_C = ((U_t_y - SVt_x0) / sigma_0) * cond_after_lite

    corr_coef = torch.abs((update_A * torch.conj(update_C)).sum(-1)) / (update_A.norm(dim=-1) * update_C.norm(dim=-1) + 1e-10)
    std_coef = sigma_next * torch.sqrt(1 - etaA**2-etaC**2 - 2 * etaA*etaC*corr_coef)
    Vt_xt_mod_next[cond_after] = \
                V_t_x0[cond_after] + coef_A * update_A[cond_after] + coef_C * update_C[cond_after] + (std_coef[:, None] * H_funcs.Vt(torch.randn_like(x)))[cond_after]

    #noisier than y (before)
    Vt_xt_mod_next[cond_before] = \
                (Sig_inv_U_t_y[cond_before_lite] * etaB + (1 - etaB) * V_t_x0[cond_before] + diff_sigma_t_nextB * H_funcs.Vt(torch.randn_like(x))[cond_before_lite])

    #aggregate all 3 cases and give next predictionz
    xt_mod_next = H_funcs.V(Vt_xt_mod_next)
    xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

    return xt_next


def est_x0_t(model, xt, t, at):
    et = model(xt, t)
    if et.size(1) == 6:
        et = et[:, :3]
    x0_t = (xt - et *(1-at).sqrt()) / at.sqrt()
    return x0_t, et
