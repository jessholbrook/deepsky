"""Write a batch of [-1, 1] samples as one PNG grid."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_grid(samples: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = ((samples.clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
    b, c, h, w = x.shape
    cols = math.ceil(math.sqrt(b))
    rows = math.ceil(b / cols)
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(b):
        r, col = divmod(i, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = x[i].permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(path)
