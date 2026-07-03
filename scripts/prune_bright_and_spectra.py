"""Retroactive prune matching the updated filters: crops from spectrum/plot
titled sources, and crops with paper-white backgrounds (median gray >= 250).
Multiprocess — scans every crop once.

    uv run python scripts/prune_bright_and_spectra.py --apply
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from deepsky.data.filtering import TITLE_BLACKLIST
from deepsky.data.preprocess import MAX_MEDIAN

CROPS = Path("data/crops256")


def is_bright(path: Path) -> bool:
    with Image.open(path) as img:
        return float(np.median(np.asarray(img.convert("L")))) >= MAX_MEDIAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    manifest_path = CROPS / "manifest.csv"
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    title_drop = {r["crop"] for r in rows if TITLE_BLACKLIST.search(r["title"] or "")}
    remaining = [CROPS / r["crop"] for r in rows if r["crop"] not in title_drop]

    with Pool() as pool:
        bright = pool.map(is_bright, remaining, chunksize=256)
    bright_drop = {p.name for p, b in zip(remaining, bright) if b}

    drop = title_drop | bright_drop
    print(f"title-blacklist: {len(title_drop)}, bright-background: {len(bright_drop)}, "
          f"total drop: {len(drop)} of {len(rows)}")

    if not args.apply:
        print("dry run — pass --apply to delete")
        return

    for name in drop:
        (CROPS / name).unlink(missing_ok=True)
    keep_rows = [r for r in rows if r["crop"] not in drop]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(keep_rows)
    print(f"pruned. {len(keep_rows)} crops remain.")


if __name__ == "__main__":
    main()
