"""Training dataset over preprocessed 256px crops.

Augmentation: random sub-crop to the training resolution, full dihedral
group (all flips/rotations are physically valid for deep-sky images —
a free 8x augmentation), normalize to [-1, 1].
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CropDataset(Dataset):
    def __init__(self, crops_dir: str | Path, image_size: int, seed: int = 0):
        self.files = sorted(Path(crops_dir).glob("*.webp"))
        if not self.files:
            raise FileNotFoundError(f"No .webp crops in {crops_dir} — run scripts/build_dataset.py")
        self.image_size = image_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.files[idx]) as img:
            arr = np.asarray(img.convert("RGB"))

        size = self.image_size
        h, w = arr.shape[:2]
        if h > size and w > size:
            y = self.rng.randint(0, h - size)
            x = self.rng.randint(0, w - size)
            arr = arr[y : y + size, x : x + size]
        elif (h, w) != (size, size):
            arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.LANCZOS))

        # Dihedral group: 4 rotations x optional flip.
        k = self.rng.randint(0, 3)
        if k:
            arr = np.rot90(arr, k)
        if self.rng.random() < 0.5:
            arr = arr[:, ::-1]

        arr = np.ascontiguousarray(arr, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)
