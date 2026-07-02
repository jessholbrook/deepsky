"""Download raw images from all sources. Idempotent — safe to re-run.

    uv run python scripts/download_all.py                 # everything
    uv run python scripts/download_all.py --source eso    # one source
"""

import argparse
from pathlib import Path

from deepsky.data import d2d_client
from deepsky.data.download import download_d2d_source, download_nasa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=[*d2d_client.SOURCES, "nasa"], default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    sources = [args.source] if args.source else [*d2d_client.SOURCES, "nasa"]
    for source in sources:
        if source == "nasa":
            download_nasa(args.raw_root)
        else:
            download_d2d_source(source, args.raw_root)


if __name__ == "__main__":
    main()
