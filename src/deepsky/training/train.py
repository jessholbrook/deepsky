"""The training loop. Single device (cuda / mps / cpu), infinite dataloader,
EMA, warmup + optional cosine LR decay, bf16 autocast on CUDA only,
periodic checkpoints and DDIM sample grids.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepsky.config import Config, resolve_device
from deepsky.data.dataset import CropDataset
from deepsky.diffusion.gaussian import GaussianDiffusion
from deepsky.diffusion.samplers import ddim_sample
from deepsky.diffusion.schedule import make_schedule
from deepsky.models.unet import UNet
from deepsky.training import checkpoint
from deepsky.training.ema import EMA
from deepsky.utils.grid import save_grid
from deepsky.utils.logging import MetricsLogger


def infinite(loader: DataLoader):
    while True:
        yield from loader


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.lr_decay == "cosine":
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * progress))
    return cfg.lr


def train(cfg: Config, resume: Path | None = None) -> Path:
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    run_dir = Path(cfg.train.run_dir)
    print(f"device: {device}, run_dir: {run_dir}")

    dataset = CropDataset(cfg.data.crops_dir, cfg.data.image_size, seed=cfg.train.seed)
    print(f"dataset: {len(dataset)} crops at {cfg.data.image_size}px")
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    model = UNet(cfg.model, cfg.data.image_size).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    diffusion = GaussianDiffusion(make_schedule(cfg.diffusion.schedule, cfg.diffusion.timesteps).to(device))
    ema = EMA(model, cfg.train.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    start_step = 0
    if resume is not None:
        start_step = checkpoint.load(resume, model, ema, optimizer, device)
        print(f"resumed from {resume} at step {start_step}")

    use_amp = cfg.train.amp and device.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )

    logger = MetricsLogger(run_dir)
    batches = infinite(loader)
    model.train()

    for step in range(start_step, cfg.train.total_steps):
        x0 = next(batches).to(device, non_blocking=True)
        lr = lr_at(step, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with autocast:
            loss = diffusion.loss(model, x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        ema.update(model)

        done = step + 1
        if done % cfg.train.log_every == 0:
            logger.log(done, loss.item(), lr)

        if done % cfg.train.checkpoint_every == 0 or done == cfg.train.total_steps:
            checkpoint.save(run_dir / f"ckpt_{done:07d}.pt", model, ema, optimizer, done)
            checkpoint.prune_old(run_dir)

        if done % cfg.train.sample_every == 0 or done == cfg.train.total_steps:
            sample_with_ema(model, ema, diffusion, cfg, device, run_dir / f"samples_{done:07d}.png")
            model.train()

    return run_dir


@torch.no_grad()
def sample_with_ema(model, ema, diffusion, cfg: Config, device, out_path: Path) -> None:
    """Swap in EMA weights, render a DDIM grid, swap back."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    model.eval()
    shape = (cfg.train.sample_count, 3, cfg.data.image_size, cfg.data.image_size)
    samples = ddim_sample(model, diffusion, shape, device, steps=cfg.train.sample_ddim_steps)
    save_grid(samples, out_path)
    model.load_state_dict(backup)
    print(f"wrote {out_path}")
