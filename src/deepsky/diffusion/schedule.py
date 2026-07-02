"""Beta schedules and every derived quantity the diffusion process needs.

All buffers are float64 during derivation for numerical accuracy, returned
as float32 tensors of shape [T].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def linear_betas(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    # The DDPM paper's endpoints are tuned for T=1000; scale for other T.
    scale = 1000 / timesteps
    return torch.linspace(beta_start * scale, beta_end * scale, timesteps, dtype=torch.float64)


def cosine_betas(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule: betas derived from a cosine ᾱ curve."""

    def alpha_bar(t: float) -> float:
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


@dataclass
class Schedule:
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef_x0: torch.Tensor
    posterior_mean_coef_xt: torch.Tensor

    @property
    def timesteps(self) -> int:
        return len(self.betas)

    def to(self, device: torch.device) -> "Schedule":
        moved = {k: v.to(device) for k, v in self.__dict__.items()}
        return Schedule(**moved)


def make_schedule(name: str, timesteps: int) -> Schedule:
    if name == "linear":
        betas = linear_betas(timesteps)
    elif name == "cosine":
        betas = cosine_betas(timesteps)
    else:
        raise ValueError(f"Unknown schedule '{name}'")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

    # q(x_{t-1} | x_t, x0) posterior: mean = c_x0 * x0 + c_xt * x_t
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef_x0 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef_xt = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    return Schedule(
        betas=betas.float(),
        alphas_cumprod=alphas_cumprod.float(),
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod).float(),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod).float(),
        posterior_variance=posterior_variance.float(),
        posterior_mean_coef_x0=posterior_mean_coef_x0.float(),
        posterior_mean_coef_xt=posterior_mean_coef_xt.float(),
    )


def gather(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Index a [T] buffer by per-sample timesteps and reshape to broadcast over [B, C, H, W]."""
    out = values.gather(0, t)
    return out.reshape(-1, *([1] * (ndim - 1)))
