# Cloud training runbook (RunPod / Vast.ai, single RTX 4090)

Budget expectation: 200k steps at 128px ≈ 16–24 h ≈ **$8–20** at $0.40–0.80/hr.

## Before renting anything

- [ ] `data/crops256/` is built and uploaded as `crops256.tar` to object storage
      (Backblaze B2 / S3 / Hugging Face Datasets — anywhere `curl`-able).
      `tar cf crops256.tar -C data crops256`
- [ ] Full test suite green locally: `uv run pytest`
- [ ] Overfit gate passed on the Mac mini: `uv run pytest tests/test_overfit.py -m slow -s`
- [ ] 64px overnight validation run produced recognizable space imagery
      (see main README milestones)
- [ ] Checkpoint resume verified: kill a smoke run mid-way, `--resume`, watch it
      continue from the saved step
- [ ] Decide kill criteria NOW (see below) — the budget dies from hesitation

## On the pod

```bash
git clone <your repo> && cd diffustion
bash cloud/setup.sh                       # installs uv, syncs deps, pulls data
uv run pytest                             # ~1 min; abort if anything fails
```

**Throughput gate** — 500-step timing run before committing to 20 hours:

```bash
uv run python scripts/train.py --config configs/cloud-128px-full.yaml 2>&1 | head -40
# watch the it/s column. Expect 2.5-3.5 it/s.
# < 2 it/s: something is wrong (thermal throttle, wrong image, PCIe-starved pod).
# Debug briefly or KILL THE POD and rent another. Do not accept a 2x-slower run.
```

Then the real run under tmux (survives SSH disconnects):

```bash
tmux new -s train
uv run python scripts/train.py --config configs/cloud-128px-full.yaml
# Ctrl-b d to detach; tmux attach -t train to return
```

In a second tmux window, keep checkpoints flowing off the pod:

```bash
bash cloud/sync_checkpoints.sh runs/cloud128 <rclone-remote>:deepsky-ckpts
```

## Monitoring from the Mac

Pull the latest sample grid periodically; visual progress is the real signal:

```bash
rclone copy <remote>:deepsky-ckpts/samples_latest.png . && open samples_latest.png
```

Expected milestones (mirror the 64px run, shifted later for 128px):
~5k steps: colored blobs, space palette. ~20k: nebula textures. ~50k: coherent
structures. 100k+: keepers appearing in every grid. Stars sharpen LAST.

## Kill criteria (write the time you'll check back in, then hold to it)

- Loss not below 0.10 by step 10k → config/data bug. Kill, debug locally.
- NaN loss at any point → kill, inspect last checkpoint locally.
- Sample grids all-gray/all-black at 10k+ → normalization or sampler bug. Kill.
- it/s degrades >30% mid-run → pod is being throttled; checkpoint, move pods.

## Resume after any interruption

```bash
uv run python scripts/train.py --config configs/cloud-128px-full.yaml \
    --resume runs/cloud128/ckpt_<latest>.pt
```

## After the run

```bash
uv run python scripts/sample.py --config configs/cloud-128px-full.yaml \
    --ckpt runs/cloud128/ckpt_0200000.pt --n 64 --sampler ddpm --out final.png
rclone copy runs/cloud128/ckpt_0200000.pt <remote>:deepsky-ckpts/   # then KILL THE POD
```
