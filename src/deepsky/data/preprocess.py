"""Derive 256px training crops from raw images.

Per image: build a 2-level zoom pyramid (short side 2048 and 1024 — two
effective zoom levels = cheap scale augmentation), sample random 256x256
crops, keep only informative ones (deep-sky frames are mostly black), cap
accepted crops per source image so one gigapixel mosaic can't dominate.

Output: data/crops256/{imageid}_{n}.webp + manifest.csv (crop -> source
image -> credit) for CC BY attribution.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

CROP = 256
PYRAMID_SHORT_SIDES = (2048, 1024)
MAX_CROPS_PER_IMAGE = 48
ATTEMPTS_PER_ACCEPTED = 8  # sampling budget: attempts = cap * this

# Informativeness thresholds on [0, 255] grayscale.
MIN_STD = 12.0
MIN_MEAN = 4.0


def crop_is_informative(arr: np.ndarray) -> bool:
    gray = arr.mean(axis=2)
    return gray.std() >= MIN_STD and gray.mean() >= MIN_MEAN


def extract_crops(path: Path, rng: random.Random) -> list[np.ndarray]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            crops: list[np.ndarray] = []
            for short_side in PYRAMID_SHORT_SIDES:
                if min(img.size) < short_side:
                    level = img if min(img.size) >= CROP else None
                else:
                    scale = short_side / min(img.size)
                    level = img.resize(
                        (round(img.width * scale), round(img.height * scale)),
                        Image.LANCZOS,
                    )
                if level is None:
                    continue
                w, h = level.size
                if w < CROP or h < CROP:
                    continue
                arr = np.asarray(level)
                budget = MAX_CROPS_PER_IMAGE * ATTEMPTS_PER_ACCEPTED // len(PYRAMID_SHORT_SIDES)
                per_level_cap = MAX_CROPS_PER_IMAGE // len(PYRAMID_SHORT_SIDES)
                accepted = 0
                for _ in range(budget):
                    if accepted >= per_level_cap:
                        break
                    x = rng.randint(0, w - CROP)
                    y = rng.randint(0, h - CROP)
                    crop = arr[y : y + CROP, x : x + CROP]
                    if crop_is_informative(crop):
                        crops.append(crop)
                        accepted += 1
            return crops
    except Exception:
        return []


def load_credits(raw_root: Path) -> dict[str, dict]:
    credits: dict[str, dict] = {}
    for meta_file in raw_root.glob("*/metadata.jsonl"):
        for line in meta_file.read_text().splitlines():
            rec = json.loads(line)
            credits[f"{rec['source']}/{rec['id']}"] = rec
    return credits


def build_crops(keep_files: list[Path], raw_root: Path, out_dir: Path, seed: int = 0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    credits = load_credits(raw_root)
    rng = random.Random(seed)
    total = 0

    manifest_path = out_dir / "manifest.csv"
    done_ids = set()
    if manifest_path.exists():  # resume: skip already-processed source images
        with open(manifest_path) as f:
            done_ids = {row["source_key"] for row in csv.DictReader(f)}

    mode = "a" if done_ids else "w"
    with open(manifest_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop", "source_key", "title", "credit", "rights"])
        if mode == "w":
            writer.writeheader()
        for path in tqdm(keep_files, desc="crops"):
            key = f"{path.parent.name}/{path.stem}"
            if key in done_ids:
                continue
            meta = credits.get(key, {})
            for n, crop in enumerate(extract_crops(path, rng)):
                name = f"{path.parent.name}_{path.stem}_{n:03d}.webp"
                Image.fromarray(crop).save(out_dir / name, "WEBP", quality=95)
                writer.writerow(
                    {
                        "crop": name,
                        "source_key": key,
                        "title": meta.get("title", ""),
                        "credit": meta.get("credit", ""),
                        "rights": meta.get("rights", ""),
                    }
                )
                total += 1
    return total
