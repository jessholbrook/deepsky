"""Resume-safe checkpointing: model + EMA + optimizer + step."""

from __future__ import annotations

from pathlib import Path

import torch


def save(path: Path, model, ema, optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        tmp,
    )
    tmp.rename(path)  # atomic: a crash mid-save never corrupts the checkpoint


def load(path: Path, model, ema, optimizer, device) -> int:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]


def prune_old(run_dir: Path, keep: int = 3) -> None:
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()
