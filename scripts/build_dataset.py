"""Dedup raw images, then derive 256px training crops.

    uv run python scripts/build_dataset.py
"""

import argparse
from pathlib import Path

from deepsky.data.dedup import find_duplicates
from deepsky.data.preprocess import build_crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/crops256"))
    args = parser.parse_args()

    keep, drop = find_duplicates(args.raw_root)
    print(f"dedup: keeping {len(keep)}, dropping {len(drop)} duplicates/unreadable")

    total = build_crops(keep, args.raw_root, args.out)
    print(f"wrote {total} new crops to {args.out}")


if __name__ == "__main__":
    main()
