"""Forward process, training loss, and the eps <-> x0 conversions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from deepsky.diffusion.schedule import Schedule, gather


class GaussianDiffusion:
    """Stateless diffusion math bound to a Schedule. Model is passed in per call."""

    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    @property
    def timesteps(self) -> int:
        return self.schedule.timesteps

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x0): sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε."""
        s = self.schedule
        return (
            gather(s.sqrt_alphas_cumprod, t, x0.ndim) * x0
            + gather(s.sqrt_one_minus_alphas_cumprod, t, x0.ndim) * noise
        )

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        s = self.schedule
        return (
            xt - gather(s.sqrt_one_minus_alphas_cumprod, t, xt.ndim) * eps
        ) / gather(s.sqrt_alphas_cumprod, t, xt.ndim)

    def posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of q(x_{t-1} | x_t, x0)."""
        s = self.schedule
        mean = (
            gather(s.posterior_mean_coef_x0, t, xt.ndim) * x0
            + gather(s.posterior_mean_coef_xt, t, xt.ndim) * xt
        )
        var = gather(s.posterior_variance, t, xt.ndim)
        return mean, var

    def loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        """Epsilon-prediction MSE with uniform t. x0 in [-1, 1]."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)
