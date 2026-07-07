"""DDPM ancestral sampling and DDIM (eta=0)."""

from __future__ import annotations

import torch
from tqdm import tqdm

from deepsky.diffusion.gaussian import GaussianDiffusion
from deepsky.diffusion.schedule import gather


@torch.no_grad()
def ddpm_sample(
    model,
    diffusion: GaussianDiffusion,
    shape: tuple[int, ...],
    device: torch.device,
    progress: bool = True,
) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    steps = range(diffusion.timesteps - 1, -1, -1)
    if progress:
        steps = tqdm(steps, desc="ddpm", leave=False)
    for i in steps:
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        x0 = diffusion.predict_x0_from_eps(x, t, eps).clamp(-1, 1)
        mean, var = diffusion.posterior(x0, x, t)
        if i > 0:
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x


@torch.no_grad()
def ddim_sample(
    model,
    diffusion: GaussianDiffusion,
    shape: tuple[int, ...],
    device: torch.device,
    steps: int = 100,
    progress: bool = True,
    x_start: torch.Tensor | None = None,
    t_start: int | None = None,
) -> torch.Tensor:
    """Deterministic DDIM (eta=0) sampling.

    By default starts from pure noise at t=T-1. Pass ``x_start`` (a batch already
    noised to timestep ``t_start``) to resume the reverse process partway — e.g.
    SDEdit-style editing, or reconstructing a known latent.
    """
    s = diffusion.schedule
    T = diffusion.timesteps
    t_hi = T - 1 if t_start is None else t_start
    # Evenly spaced subsequence of timesteps, descending, always ending at 0.
    ts = torch.linspace(t_hi, 0, steps).round().long().tolist()
    ts = sorted(set(ts), reverse=True)

    x = torch.randn(shape, device=device) if x_start is None else x_start
    pairs = list(zip(ts, ts[1:] + [-1]))
    if progress:
        pairs = tqdm(pairs, desc="ddim", leave=False)
    for t_cur, t_prev in pairs:
        t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
        eps = model(x, t)
        x0 = diffusion.predict_x0_from_eps(x, t, eps).clamp(-1, 1)
        if t_prev < 0:
            x = x0
            break
        tp = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
        acp_prev = gather(s.alphas_cumprod, tp, x.ndim)
        # eta=0: deterministic update straight along the predicted direction.
        x = acp_prev.sqrt() * x0 + (1 - acp_prev).sqrt() * eps
    return x
