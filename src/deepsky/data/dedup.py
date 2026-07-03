"""Perceptual-hash dedup across sources.

Hubble press images appear in both the ESA feed and the NASA library; ESA
copies win (better metadata, guaranteed full-res Large asset). Priority is
the order of PREFERENCE below.
"""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # mosaics exceed PIL's decompression-bomb default

PREFERENCE = ["esahubble", "esawebb", "eso", "noirlab", "nasa"]
HAMMING_THRESHOLD = 6


def _phash(path: Path) -> imagehash.ImageHash | None:
    try:
        with Image.open(path) as img:
            img.draft("RGB", (1024, 1024))  # decode at reduced scale — fast on huge JPGs
            return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


def find_duplicates(raw_root: Path) -> tuple[list[Path], list[Path]]:
    """Return (keep, drop) lists over all downloaded images."""
    files = sorted(
        raw_root.glob("*/*.jpg"),
        key=lambda p: (
            PREFERENCE.index(p.parent.name) if p.parent.name in PREFERENCE else 99,
            p.name,
        ),
    )
    keep: list[Path] = []
    drop: list[Path] = []
    seen: list[tuple[imagehash.ImageHash, Path]] = []

    for path in tqdm(files, desc="dedup phash"):
        h = _phash(path)
        if h is None:
            drop.append(path)  # unreadable file
            continue
        dup_of = next((p for hs, p in seen if h - hs <= HAMMING_THRESHOLD), None)
        if dup_of is None:
            seen.append((h, path))
            keep.append(path)
        else:
            drop.append(path)
    return keep, drop
