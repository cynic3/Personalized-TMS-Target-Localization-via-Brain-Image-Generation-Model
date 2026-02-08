# pytorch3dunet/unet3d/diffusion.py

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, betas):
        super().__init__()
        betas = np.array(betas, dtype=np.float64)
        assert betas.shape == (len(betas),)

        self.n_timesteps = len(betas)

        # Register buffers for diffusion parameters
        self.register_buffer('betas', torch.from_numpy(betas).float())
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.register_buffer('alphas_cumprod', torch.from_numpy(alphas_cumprod).float())
        self.register_buffer('alphas_cumprod_prev', torch.from_numpy(alphas_cumprod_prev).float())

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recipm1_alphas', torch.sqrt(1.0 / alphas - 1))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_loss(self, model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)

        loss = nn.functional.mse_loss(noise, predicted_noise)
        return loss

    def p_sample(self, model, x, t_index):
        betas_t = extract(self.betas, t_index, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_index, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_index, x.shape)

        # Equation (14). mean_theta = 1/sqrt(alpha_t) * (x_t - betas_t * predicted_noise / sqrt(1-alphas_cumprod_t))
        predicted_noise = model(x, t_index)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index[0].item() == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            sqrt_beta_t = extract(torch.sqrt(self.betas), t_index, x.shape)
            return model_mean + sqrt_beta_t * noise

    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.n_timesteps)), desc='Sampling', total=self.n_timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return img


def extract(a, t, x_shape):
    """
    Gather values from a at indices t and reshape to match x_shape.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
