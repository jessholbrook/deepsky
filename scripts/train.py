"""Train a model.

    uv run python scripts/train.py --config configs/mac-64px-validation.yaml
    uv run python scripts/train.py --config configs/smoke.yaml --resume runs/smoke/ckpt_0000100.pt
"""

import argparse
from pathlib import Path

from deepsky.config import load_config
from deepsky.training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
