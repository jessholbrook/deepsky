#!/usr/bin/env bash
# Pod bootstrap: run from the repo root on a fresh RunPod/Vast.ai instance.
set -euo pipefail

DATA_URL="${DATA_URL:?set DATA_URL to the crops256.tar download URL}"

apt-get update -qq && apt-get install -y -qq curl tmux rsync

# uv + deps (resolves modern torch+CUDA on linux automatically)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync

# dataset
mkdir -p data
curl -L "$DATA_URL" -o /tmp/crops256.tar
tar xf /tmp/crops256.tar -C data
rm /tmp/crops256.tar
# macOS tar embeds AppleDouble (._*) companion files; strip them so the dataset
# loader's *.webp glob doesn't try to open metadata blobs.
find data/crops256 -name '._*' -delete
echo "crops: $(find data/crops256 -name '*.webp' | wc -l)"

uv run python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0))"
echo "ready. next: uv run pytest && uv run python scripts/train.py --config configs/cloud-128px-full.yaml"
