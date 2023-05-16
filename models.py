import math

import numpy as np
import torch
from model_utils import PositionalEmbedding
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def init_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def spectral_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def ntk_init(m):
    if isinstance(m, torch.nn.Linear):
        fan_out = m.weight.shape[0]
        scale = math.sqrt(2.0 / fan_out)
        m.weight.data.normal_(0, scale)


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)  ## Just a sinusoidal embedding without learnable parameters
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)  ## Just a sinusoidal embedding without learnable parameters
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)  ## Just a sinusoidal embedding without learnable parameters

        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0]).float()
        x2_emb = self.input_mlp2(x[:, 1]).float()
        t_emb = self.time_mlp(t)

        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x



class NoiseScheduler():
    """Helper Class that implements the noise schedule and saves alpha, alphas_cumprod, and beta."""
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 cosine_epsilon= 0.008,
                 device="cpu"):

        self.device = device
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32).to(device) ** 2
        elif beta_schedule == "cosine":
            # Cosine schedule from https://arxiv.org/abs/2102.09672
            # beta_t = clip(1 - alpha_cumprod_t / alpha_cumprod_{t-1}, 0.999)
            # alpha_cumprod_t = f(t) / f(0)
            # f(t) = cos((t/T + cosine_epsilon) / (1 + cosine_epsilon) * 0.5 pi)^2
            steps = torch.linspace(0, num_timesteps, steps=num_timesteps + 1, dtype=torch.float32)
            f_t = torch.cos(((steps / num_timesteps + cosine_epsilon) / (1.0 + cosine_epsilon)) * math.pi * 0.5) ** 2
            self.betas = torch.clip(1.0 - f_t[1:] / f_t[:num_timesteps], 0.0, 0.999).to(device)
        self.alphas = 1.0 - self.betas  # alpha_t = 1 - beta_t

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)   # alphas_cumprod_t = prod_{i=0}^{t-1} alpha_i
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        ############## required for self.add_noise ################################################################
        # We have x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * eps
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5   # sqrt(alphas_cumprod_t)
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5   # sqrt(1 - alphas_cumprod_t)

        ############## required to reconstruct_x0 ################################################################
        # We have x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * eps
        # x_0 = (1 / sqrt(alphas_cumprod_t)) * x_t - (sqrt(1 / alphas_cumprod_t - 1) * eps))
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        ############## required for q_posterior q(x_{t-1} | x_t, x_0) = N(x_{t-1} | mu_t, beta_t) ################

        # mu_t = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        # posterior_mean_coef1 = beta_t * sqrt(alphas_cumprod_{t-1}) / (1 - alphas_cumprod_t)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # posterior_mean_coef2 = sqrt(alpha_t) * (1 - alphas_cumprod_{t-1}) / (1 - alphas_cumprod_t)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        """Calculate x_0 = (1 / sqrt(alphas_cumprod_t)) * x_t - (sqrt(1 / alphas_cumprod_t - 1) * eps))."""
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]

        # Reshape s1 and s2 to have shape [bs, 1, ..., 1] to match the shape of x_start and x_noise.
        s1 = s1.reshape(-1, *((len(x_t.shape) - 1) * (1,)))
        s2 = s2.reshape(-1, *((len(noise.shape) - 1) * (1,)))
        return s1 * x_t - s2 * noise

    def get_posterior_mean(self, x_0, x_t, t):
        """We calculate mu_t = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t for q(x_{t-1} | x_t, x_0)"""
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]

        # Reshape s1 and s2 to have shape [bs, 1, ..., 1] to match the shape of x_start and x_noise.
        s1 = s1.reshape(-1, *((len(x_0.shape) - 1) * (1,)))
        s2 = s2.reshape(-1, *((len(x_t.shape) - 1) * (1,)))
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_posterior_variance(self, t):
        """Get tilde_beta_t = beta_t * (1 - alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t) for q(x_{t-1} | x_t, x_0)"""
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, clip=False):
        t = timestep
        # Calculate x_0 given x_t and eps.
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)

        if clip:
            pred_original_sample = pred_original_sample.clamp_(-1., 1.)

        # Calculate x_{t-1} given x_t and x_0.
        pred_prev_sample = self.get_posterior_mean(pred_original_sample, sample, t)

        # Obtain the variance tilde_beta_t for q(x_{t-1} | x_t, x_0)
        std_dev = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            std_dev = (self.get_posterior_variance(t) ** 0.5) * noise

        # x_{t-1} += sqrt(tilde_beta_t) * z
        pred_prev_sample = pred_prev_sample + std_dev

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        """Uses reparametrisation trick to sample x_t given x_0 and alpha_cumprod_t.

        From the DDPM, we have:
            x_t = N(sqrt(bar(alpha_cumprod_t)) x_0, (1 - alpha_cumprod_t) I)

        Using reparametrisation trick, we can write this using a sample from unit Normal as:
            x_t = sqrt(bar(alpha_cumprod_t)) x_0 + sqrt(1 - alpha_cumprod_t) * noise

        x_start: x_0
        x_noise: noise N(0, 1)
        timesteps: timesteps to sample from
        """
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape s1 and s2 to have shape [bs, 1, ..., 1] to match the shape of x_start and x_noise.
        s1 = s1.reshape(-1, *((len(x_start.shape) - 1) * (1,)))
        s2 = s2.reshape(-1, *((len(x_start.shape) - 1) * (1,)))

        return s1 * x_start + s2 * x_noise

    def sample(self, x_T, timesteps, model, stepsize=10, clip=False):
        eval_batch_size = x_T.shape[0]

        samples = [x_T]
        sample = x_T
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(self.device)

            # Get eps_{theta}(x_t, t)
            with torch.no_grad():
                residual = model(sample, t)

            # Get x_{t-1} = (1/sqrt(alpha_cumprod_t)) * x_t - (sqrt(1/alpha_cumprod_t - 1) * eps) + sqrt(tilde_beta_t) * z
            sample = self.step(residual, t[0], sample, clip=clip)

            if i % stepsize == 0 or i == len(timesteps) - 1:
                samples.append(sample)

        return samples

    def __len__(self):
        return self.num_timesteps
