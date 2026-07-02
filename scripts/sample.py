"""Generate a sample grid from a checkpoint (uses EMA weights).

    uv run python scripts/sample.py --config configs/mac-64px-validation.yaml \\
        --ckpt runs/mac64/ckpt_0075000.pt --n 64 --steps 250 --out samples.png
"""

import argparse
from pathlib import Path

import torch

from deepsky.config import load_config, resolve_device
from deepsky.diffusion.gaussian import GaussianDiffusion
from deepsky.diffusion.samplers import ddim_sample, ddpm_sample
from deepsky.diffusion.schedule import make_schedule
from deepsky.models.unet import UNet
from deepsky.utils.grid import save_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=100, help="DDIM steps (ddpm always uses all)")
    parser.add_argument("--out", type=Path, default=Path("samples.png"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = UNet(cfg.model, cfg.data.image_size).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["ema"]["shadow"])
    model.eval()

    diffusion = GaussianDiffusion(
        make_schedule(cfg.diffusion.schedule, cfg.diffusion.timesteps).to(device)
    )
    shape = (args.n, 3, cfg.data.image_size, cfg.data.image_size)
    if args.sampler == "ddim":
        samples = ddim_sample(model, diffusion, shape, device, steps=args.steps)
    else:
        samples = ddpm_sample(model, diffusion, shape, device)
    save_grid(samples, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
