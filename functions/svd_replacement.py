import torch
import numpy as np
from torch import optim
from motionblur.motionblur import Kernel
from PIL import Image

class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))

class DeblurringArbitral2D(H_functions):

    def __init__(self, kernel, channels, img_dim, device, conv_shape='same'):
        self.img_dim   = img_dim
        self.channels  = channels
        self.conv_shape = conv_shape
        _nextpow2 = lambda x : int(np.power(2, np.ceil(np.log2(x))))
        self.fft2_size = _nextpow2(img_dim + kernel.shape[0] - 1) # next pow 2
        self.kernel_size = (kernel.shape[-2], kernel.shape[-1])
        self.kernel = kernel
        self.init_kernel = kernel.detach().clone()
        self.update_kernel(kernel)
        self.device = device

        if conv_shape == 'same':
            self.out_img_dim = img_dim
        elif conv_shape == 'full':
            # TODO: rectangular kernel size
            self.out_img_dim = img_dim + (self.kernel_size[0] - 1)
        elif conv_shape == "same_interp":
            self.out_img_dim = img_dim
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        ret = self.U(singulars * temp[:, :singulars.shape[1]])

        return ret

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[1]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[1]] = temp[:, :singulars.shape[1]] / singulars
        return self.V(self.add_zeros(temp))

    def V(self, vec):
        
        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = vec / self._singular_phases[:, None, :]
        vec = vec.reshape(vec.shape[0], self.channels, -1)

        vec = self._batch_inv_perm(vec, self._perm)

        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size),\
            norm="ortho").real


        out = vec_ifft[:, :, :self.img_dim, :self.img_dim].reshape(vec.shape[0], -1)

        return out

    def Vt(self, vec):
        
        vec_fft = torch.fft.fft2(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")

        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)
        vec_fft = vec_fft * self._singular_phases[:, None, :]

        return vec_fft.reshape(vec.shape[0], -1)

    def U(self, vec):

        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = self._batch_inv_perm(vec, self._perm)


        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size),\
            norm="ortho").real

        if self.conv_shape == 'same':        
            out = vec_ifft[:, :, self.kernel_size[0]//2:int(self.kernel_size[0]//2+self.img_dim), \
                self.kernel_size[1]//2:int(self.kernel_size[1]//2+self.img_dim)]
        elif self.conv_shape == 'full':
            out = vec_ifft[:, :, :self.out_img_dim, :self.out_img_dim]
        else: # elif self.conv_shape == "same_interp":
            out = vec_ifft[:, :, self.kernel_size[0]//2:int(self.kernel_size[0]//2+self.img_dim), \
                self.kernel_size[1]//2:int(self.kernel_size[1]//2+self.img_dim)]

        return out

    def Ut(self, vec):
        
        _ks0 = self.kernel_size[0]
        _ks1 = self.kernel_size[1]
        _Nf  =  self.fft2_size

        if self.conv_shape == 'same':
            exec_zeropad = torch.nn.ZeroPad2d((_ks0//2, _Nf-_ks0//2-self.img_dim,\
                _ks1//2, _Nf-_ks1//2-self.img_dim))
            
            vec = exec_zeropad(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim))
        elif self.conv_shape == 'full':
            vec = vec.reshape(vec.shape[0], self.channels, self.out_img_dim, self.out_img_dim)
            exec_zeropad = torch.nn.ZeroPad2d((0, _Nf-self.out_img_dim, 0, _Nf-self.out_img_dim))
            vec = exec_zeropad(vec)

        elif self.conv_shape == "same_interp":
            pass

        vec_fft = torch.fft.fft2(vec, (self.fft2_size, self.fft2_size), norm="ortho")

        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)

        return vec_fft.reshape(vec.shape[0], -1)

    def singulars(self):
        
        bsz = self._singulars.shape[0]
        return self._singulars.repeat(1, 3).reshape(bsz, -1)

    def add_zeros(self, vec):
        tmp = torch.zeros(vec.shape[0], self.channels, self.fft2_size**2, device=vec.device, dtype=vec.dtype)
        reshaped = vec.clone().reshape(vec.shape[0], self.channels, -1)
        tmp[:, :, :reshaped.shape[2]] = reshaped

        return tmp.reshape(vec.shape[0], -1)
    
    def update_kernel(self, kernel):
        """
        Update the internal kernel and associated variables using the provided kernel tensor.

        Args:
            kernel (torch.Tensor): The kernel tensor for the update. It should have the same shape of self.kernel

        Returns:
            None
        """

        self.kernel = kernel
        self.k_fft = torch.fft.fft2(kernel, (self.fft2_size, self.fft2_size), norm="ortho")

        bsz = kernel.shape[0]
        _eps_singulars = 1e-10 * torch.randn_like(self.k_fft)
        self._singular_phases = ((self.k_fft + _eps_singulars) / torch.abs(self.k_fft + _eps_singulars)).reshape(bsz, -1)
        self._singulars = torch.abs(self.k_fft * self.fft2_size).reshape(bsz, -1)
        ZERO = 0.05
        self._singulars[self._singulars < ZERO] = 0.0
        self._singulars, self._perm = self._singulars.sort(descending=True)
        self._singular_phases = self._batch_perm(self._singular_phases.reshape(bsz, -1), self._perm)
    
    def _batch_perm(self, tensor, perm):

        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, :] = tensor[i_bsz, perm[i_bsz]]
            elif tensor.dim() == 3:
                tensor[i_bsz, :, :] = tensor[i_bsz, :, perm[i_bsz]]

        return tensor

    def _batch_inv_perm(self, tensor, perm):

        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, perm[i_bsz]] = tensor[i_bsz, :].clone()
            elif tensor.dim() == 3:
                tensor[i_bsz, :, perm[i_bsz]] = tensor[i_bsz, :, :].clone()

        return tensor

    def update_H_optim(self, y, x, n_iter=1000, lr = 1e-3, reg_H_gamma=0.0, reg_H_type="norm"):
        """
            update the kernel parameter so that it satisfies y = H(k)x
                using Adam optimizer
            The optimizer minimizes the following objective:
            if reg_H_type == "norm":
                ||y - H(x)||_2 + reg_H_gamma * ||k||_1
            else reg_H_type == "diff_norm":
                ||y - H(x)||_2 + reg_H_gamma * ||k - k_init||_2
            
            Args:
                y : (#batch, #channel, self.out_img_dim, self.out_img_dim) blurry image
                x : (#batch, #channel, self.img_dim, self.img_dim) (estimated) clean image
                n_iter : (int) the optimizer iterations
                lr : (float) leraning rate for the optimizer
                reg_H_gamma : weight of regularization
                reg_H_type : ["norm", "diff_norm"]  l1 norm or norm of difference between kernel and init_kernel        
        """

        with torch.set_grad_enabled(True):
            for i_batch in range(x.shape[0]):
                self.kernel.requires_grad_()

                params = [{'params': self.kernel}]
                optimizer = optim.Adam(params, lr=lr)

                x = x.to(y.device)

                for i in range(n_iter):
                    
                    optimizer.zero_grad()
                    
                    y_est = self.H_fftconv(x[i_batch][None, :, :, :], self.kernel[i_batch][None, :, :])

                    if reg_H_type == "norm":
                        loss = torch.norm(y[i_batch] - y_est) + reg_H_gamma * torch.sum(torch.abs(self.kernel[i_batch])) # L1 Regularization
                    else: # elif reg_H_type == "diff_norm":
                        loss = torch.norm(y[i_batch] - y_est) + reg_H_gamma * torch.norm(self.kernel[i_batch] - self.init_kernel[i_batch])

                    if i % 100 == 0:
                        print(f"loss : {loss.item()}")
                    
                    loss.backward()
                    optimizer.step()

            self.kernel.requires_grad_(False)
            self.kernel = self.kernel / self.kernel.sum(dim=(-2, -1), keepdim=True)
            self.update_kernel(self.kernel)
            return None

    def update_H_langevin(self, y, x, n_iter=1000, lr = 1e-3, reg_H_gamma=0.0, reg_H_type="norm"):
        """
        Update the kernel parameter using Langevin dynamics.

        Args:
            y (torch.Tensor): Blurry image tensor of shape (#batch, #channel, self.out_img_dim, self.out_img_dim).
            x (torch.Tensor): (Estimated) clean image tensor of shape (#batch, #channel, self.img_dim, self.img_dim).
            n_iter (int): Number of optimizer iterations.
            lr (float): Learning rate for the optimizer.
            reg_H_gamma (float): Weight of regularization.
            reg_H_type (str): Type of regularization. Can be either "norm" or "diff_norm".

        Returns:
            None
        """

        with torch.set_grad_enabled(True):
            self.kernel.requires_grad_()

            x = x.to(y.device)

            for i in range(n_iter):
                                    
                y_est = self.H_fftconv(x, self.kernel)

                if reg_H_type == "norm":
                    loss = torch.norm(y - y_est)**2 / (2*0.04**2) + reg_H_gamma * torch.sum(torch.abs(self.kernel)) # L1 Regularization
                else: # elif reg_H_type == "diff_norm":
                    loss = torch.norm(y - y_est)**2 + reg_H_gamma * torch.norm(self.kernel - self.init_kernel)
                
                loss.backward()
                with torch.no_grad():
                    self.kernel.add_(self.kernel.grad, alpha=-lr/2)
                    self.kernel.add_(torch.randn_like(self.kernel), alpha=np.sqrt(lr))
                self.kernel.grad.zero_()

            self.kernel.requires_grad_(False)
            self.kernel = self.kernel / self.kernel.sum(dim=(-2, -1), keepdim=True)
            self.update_kernel(self.kernel)
            return None

    def reset_Hupdate(self):

        self.update_kernel(self.init_kernel.detach().clone())

    def H_fftconv(self, x, kernel):

        x_fft = torch.fft.fft2(x.reshape(x.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")

        k_fft = torch.fft.fft2(kernel, (self.fft2_size, self.fft2_size), norm="ortho")[:, None, :, :]

        y_fft = k_fft * x_fft

        y_fftconv = torch.fft.ifft2(y_fft, norm="ortho").real * self.fft2_size

        if self.conv_shape == "same" or self.conv_shape == "same_interp":
            shifts = (self.kernel_size[0]//2, self.kernel_size[1]//2)
            y_fftconv_clip = y_fftconv[:, :, shifts[0]:int(shifts[0]+self.img_dim), \
                shifts[1]:int(shifts[1]+self.img_dim)]
        else: # self.conv_type == "full"
            y_fftconv_clip = y_fftconv[:, :, :self.img_dim+(self.kernel_size[0]-1), :self.img_dim+(self.kernel_size[1]-1)]
        
        return y_fftconv_clip

    def interp_y_0(self, y_0, x_0, sigma_0):

        x_fft = torch.fft.fft2(x_0.reshape(x_0.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")

        k_fft = torch.fft.fft2(self.kernel, (self.fft2_size, self.fft2_size), norm="ortho")[:, None, :, :]

        y_fft = k_fft * x_fft
        y_fftconv = torch.fft.ifft2(y_fft, norm="ortho").real * self.fft2_size

        shifts = (self.kernel_size[0]//2, self.kernel_size[1]//2)

        y_fftconv += sigma_0 * torch.randn_like(y_fftconv)
        y_fftconv[:, :, shifts[0]:int(shifts[0]+self.img_dim), \
                shifts[1]:int(shifts[1]+self.img_dim)] = y_0
        
        return y_fftconv
    
    @staticmethod
    def get_blur_kernel_batch(batch_size, kernel_type, device):
        """
        Generates a batch of blur kernels of the specified type.

        Args:
            batch_size (int): The number of blur kernels to generate.
            kernel_type (str): The type of blur kernel to generate. Can be one of "gauss", "motionblur", "from_png", or "uniform".
            device (torch.device): The device on which to generate the blur kernels.

        Returns:
            torch.Tensor: A batch of blur kernels of shape (batch_size, kernel_size, kernel_size).
        """

        if kernel_type == "gauss":
            sigma = 5
            pdf = lambda x : torch.exp(torch.Tensor([-0.5 * (x / sigma)]))
            kernel_size = 9 # must be odd
            kernel = torch.zeros((kernel_size, kernel_size)).to(device)
            for i in range(-(kernel_size//2), kernel_size//2+1):
                for j in range(-(kernel_size//2), kernel_size//2+1):
                    kernel[i+kernel_size//2, j+kernel_size//2] = pdf(torch.sqrt(torch.Tensor([i**2+j**2])))
                # zeropad_fun = torch.nn.ZeroPad2d((10, 10, 10, 10))
                # kernel = zeropad_fun(kernel)
            kernel = kernel / kernel.sum()
            kernel_batch = kernel.repeat(batch_size, 1, 1)

        elif kernel_type == "motionblur":
            kernel_size = 64
            kernel_batch = torch.zeros(batch_size, kernel_size, kernel_size, device=device)
            for i_batch in range(batch_size):
                kernel = (Kernel(size=(kernel_size, kernel_size), intensity=0.50).kernelMatrix)
                kernel = torch.from_numpy(kernel).clone().to(device)
                kernel = kernel / kernel.sum()
                kernel_batch[i_batch] = kernel
        else: # config.deblur.kernel_type == "uniform":
            kernel_size = 31
            kernel = torch.ones((kernel_size, kernel_size)).to(device)
            kernel = kernel / kernel.sum()
            kernel_batch = kernel.repeat(batch_size, 1, 1)
            if kernel_type != "uniform":
                print("please specify the kernel type from [gauss, mnist, uniform, motionblur]. uniform kernel is used.")

        return kernel_batch

    @staticmethod
    def corrupt_kernel_batch(kernel_batch, kernel_corruption, kernel_corruption_coef=None):
        """
        Adds corruption to a batch of blur kernels.

        Args:
            kernel_batch (torch.Tensor): A batch of blur kernels of shape (batch_size, kernel_size, kernel_size).
            kernel_corruption (str): The type of corruption to add. Can be one of "additive", "multiplicative", "random_init", or "gauss_init".
            kernel_corruption_coef (float, optional): The coefficient of corruption. Defaults to None.

        Returns:
            torch.Tensor: A batch of corrupted blur kernels of shape (batch_size, kernel_size, kernel_size).
        """
        if kernel_corruption == "additive":                
            kernel_uncert_batch = kernel_batch + torch.abs(kernel_corruption_coef * torch.randn_like(kernel_batch))

        elif kernel_corruption == "multiplicative":
            kernel_uncert_batch = kernel_batch + kernel_corruption_coef * torch.randn_like(kernel_batch) * kernel_batch

        elif kernel_corruption == "random_init":
            kernel_uncert_batch = torch.rand_like(kernel_batch)

        elif kernel_corruption == "gauss_init":

            _batch_size = kernel_batch.shape[0]
            _kernel_size = kernel_batch.shape[1]

            sigma = 5
            pdf = lambda x : torch.exp(torch.Tensor([-0.5 * (x / sigma)]))
            kernel = torch.zeros((_kernel_size, _kernel_size)).to(kernel_batch.device)
            for i in range(-_kernel_size//2+1, _kernel_size//2+1):
                for j in range(-_kernel_size//2+1, _kernel_size//2+1):
                    kernel[i+_kernel_size//2-1, j+_kernel_size//2-1] = pdf(torch.sqrt(torch.Tensor([i**2+j**2])))
            kernel = kernel / kernel.sum()
            kernel_uncert_batch = kernel.repeat(_batch_size, 1, 1)

        else:
            kernel_uncert_batch = kernel_batch
        
        # Normalization
        kernel_uncert_batch = kernel_uncert_batch / kernel_uncert_batch.sum(dim=(-2, -1), keepdim=True)
        return kernel_uncert_batch
