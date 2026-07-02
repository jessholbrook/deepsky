"""Config dataclasses with YAML overrides.

YAML files only specify what differs from the dataclass defaults, e.g.:

    model:
      base_channels: 128
    train:
      batch_size: 64
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class ModelCfg:
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16, 8)
    time_emb_dim: int = 256
    dropout: float = 0.1


@dataclass
class DiffusionCfg:
    timesteps: int = 1000
    schedule: str = "cosine"  # cosine | linear
    prediction: str = "eps"  # eps (v reserved for later)


@dataclass
class DataCfg:
    crops_dir: str = "data/crops256"
    image_size: int = 64
    num_workers: int = 4


@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 2e-4
    warmup_steps: int = 1000
    lr_decay: str = "constant"  # constant | cosine
    lr_min: float = 1e-5
    total_steps: int = 75_000
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    amp: bool = False  # bf16 autocast; CUDA only
    device: str = "auto"
    run_dir: str = "runs/default"
    checkpoint_every: int = 5000
    sample_every: int = 5000
    sample_count: int = 64
    sample_ddim_steps: int = 100
    log_every: int = 50
    seed: int = 0


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _apply(dc, overrides: dict):
    fields = {f.name: f for f in dataclasses.fields(dc)}
    for key, value in overrides.items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {type(dc).__name__}")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current):
            _apply(current, value)
        elif isinstance(current, tuple):
            setattr(dc, key, tuple(value))
        else:
            setattr(dc, key, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        overrides = yaml.safe_load(Path(path).read_text()) or {}
        _apply(cfg, overrides)
    return cfg


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
