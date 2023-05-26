import glob
import logging
import os
import random
import shutil
import time
from datetime import datetime

import lpips
import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
from torch.nn.parallel import DataParallel
import tqdm
from datasets import data_transform, get_dataset, inverse_data_transform
from functions.ckpt_util import download, get_ckpt_path
from functions.denoising import sample_gibbsddrm
from guided_diffusion.script_util import (args_to_dict, classifier_defaults,
                                          create_classifier, create_model)
from guided_diffusion.unet import UNetModel
from models.diffusion import Model
from pytz import timezone
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torch.utils import tensorboard
from torch.utils.tensorboard.writer import SummaryWriter


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            elif self.config.data.dataset == "FFHQ":
                name = 'ffhq'
            else:
                raise ValueError
            if name != 'celeba_hq' and name != 'ffhq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            elif name == "ffhq":
                ckpt = os.path.join(self.args.exp, "logs/ffhq/ffhq_10m.pt")
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()

            if self.config.data.dataset == "FFHQ":
                ckpt = os.path.join(self.args.exp, "logs/ffhq/ffhq_10m.pt")
            elif self.config.data.dataset == "AFHQ":
                ckpt = os.path.join(self.args.exp, "logs/afhq/afhqdog_p2.pt")
            else:
                if self.config.model.class_cond:
                    ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                    if not os.path.exists(ckpt):
                        download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
                else:
                    ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                    if not os.path.exists(ckpt):
                        download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn
        else:
            model=None

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        
        device_count = torch.cuda.device_count()
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg == "deblur_arbitral":
            
            # Since H_funcs is 
            from functions.svd_replacement import DeblurringArbitral2D

            conv_type = config.deblur.conv_type

        else:
            print("ERROR: degradation type not supported")
            quit()
        sigma_0 = 2 * config.deblur.sigma_0 # to account for scaling to [-1, 1]
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)

        # Make directory which stores result images
        dt_now = datetime.now(timezone('Asia/Tokyo'))
        dt_str = dt_now.strftime('%Y_%m%d_%H%M%S')
        image_folder = self.args.image_folder + dt_str
        os.makedirs(image_folder, exist_ok=False)
        # save config file
        config_path = os.path.join("configs", self.args.config)
        shutil.copyfile(config_path, os.path.join(image_folder, self.args.config))

        if config.logger.enable_log:
            # Tensorboard
            config.logger.writer = SummaryWriter(image_folder)
            config.logger.image_folder = image_folder
            config.logger.inverse_data_transform = inverse_data_transform

        for x_orig, classes in pbar:
            
            batch_size = x_orig.shape[0]
            kernel_batch = DeblurringArbitral2D.get_blur_kernel_batch(batch_size, config.deblur.kernel_type, self.device)
            kernel_uncert_batch = \
                DeblurringArbitral2D.corrupt_kernel_batch(kernel_batch, \
                                                                 config.deblur.kernel_corruption, \
                                                                 config.deblur.kernel_corruption_coef)

            H_funcs = DeblurringArbitral2D(kernel_batch, config.data.channels, self.config.data.image_size, self.device, conv_type)
            H_funcs_uncert = DeblurringArbitral2D(kernel_uncert_batch, config.data.channels, self.config.data.image_size, self.device, conv_type)

            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y_0 = H_funcs.H(x_orig)            
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
            
            # Save images to the directory
            for i in range(len(y_0)):
                tvu.save_image(
                    inverse_data_transform(config, y_0[i].view(config.data.channels, H_funcs.out_img_dim, H_funcs.out_img_dim)), os.path.join(image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]), os.path.join(image_folder, f"orig_{idx_so_far + i}.png")
                )

            ##Begin GibbsDDRM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():

                x = self.sample_image(x, model, H_funcs_uncert, y_0, sigma_0, last=True, cls_fn=cls_fn, classes=classes)                    
                
                # Save images of estimated kernel
                estimated_kernel = H_funcs_uncert.kernel
                for i in range(estimated_kernel.shape[0]):
                    tvu.save_image(torch.abs(estimated_kernel[i]) / torch.max(torch.abs(estimated_kernel[i])), os.path.join(image_folder, f"estimated_kernel_{idx_so_far + i}.png"))

            x = [inverse_data_transform(config, y) for y in x]

            fun_ssim = StructuralSimilarityIndexMeasure().to(self.device)
            fun_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)
            with torch.no_grad():
                        
                for i in [-1]: #range(len(x)):
                    for j in range(x[i].size(0)):
                        tvu.save_image(
                            x[i][j], os.path.join(image_folder, f"{idx_so_far + j}_{i}.png")
                        )
                        if i == len(x)-1 or i == -1:
                            orig = inverse_data_transform(config, x_orig[j])
                            mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                            psnr = 10 * torch.log10(1 / mse)
                            avg_psnr += psnr

            idx_so_far += y_0.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

        if config.logger.enable_log:
            config.logger.writer.close()

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.config.deblur.timesteps
        seq = range(0, self.num_timesteps, skip)

        x_init = x

        for i_ddrm in range(self.config.deblur.iter_DDRM):

            x = sample_gibbsddrm(x_init, seq, model, self.betas, H_funcs, y_0, sigma_0, \
                    etaB=self.config.deblur.etaB, etaA=self.config.deblur.etaA, etaC=self.config.deblur.etaC, etaD = self.config.deblur.etaD, cls_fn=cls_fn, classes=classes, 
                    config=self.config)

            if self.config.logger.enable_log:
                # Log DDRM output
                x_on_cpu = self.config.logger.inverse_data_transform(self.config, x[0][-1])
                x_on_cpu = x_on_cpu.to("cpu").detach()
                self.config.logger.writer.add_images("DDRM output", x_on_cpu, i_ddrm)
                # Log kernel
                kernel_on_cpu = H_funcs.kernel[:, None, :, :].repeat(1, 3, 1, 1).to("cpu").detach()
                self.config.logger.writer.add_images("Refined kernel", torch.abs(kernel_on_cpu)/torch.max(torch.abs(kernel_on_cpu)), i_ddrm)
                
        if last:
            x = [x[0][-1]]
        return x
