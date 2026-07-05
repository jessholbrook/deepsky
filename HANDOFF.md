# Handoff — current state & next steps

Snapshot for picking up work on a new machine (esp. the Mac mini). If you're a
fresh Claude Code session: read this, then [README.md](README.md) (run milestones)
and [cloud/README.md](cloud/README.md) (the paid run).

## Where the project stands (as of the last session)

- **Code: complete and verified.** From-scratch DDPM — cosine-schedule diffusion
  math, configurable U-Net (19M params @ 64px, 87M @ 128px), training loop with
  EMA / atomic checkpoints / DDIM sample grids, DDPM+DDIM samplers. 23 fast tests
  pass; smoke run + checkpoint-resume proven on CPU.
- **Dataset: built and curated — 159,462 clean 256px crops (3.8 GB).** Sources:
  ESA/Hubble, ESA/Webb, ESO (all CC BY 4.0) + NASA (public domain). Per-crop
  credits in `data/crops256/manifest.csv`. Multi-stage curation (category scrape →
  title/desc blacklist → pHash dedup → AVM image-type prune → brightness/spectra
  prune → CLIP zero-shot prune). Verified visually to near-zero contamination.
- **Not in git (by design):** `data/` and `runs/`. The 3.8 GB `crops256/` moves
  machine-to-machine separately (AirDrop / rsync / external drive), not via GitHub.

## The machine split

| Machine | Role |
|---|---|
| Intel MacBook Pro | code + data pipeline (no usable GPU) — where the dataset was built |
| **Mac mini (Apple Silicon)** | **overnight 64px validation runs on MPS ← next step** |
| Rented RTX 4090 | the real 128px run (~$10–20) |

## Next steps (all on the Mac mini)

```bash
# 0. one-time setup
git clone https://github.com/jessholbrook/deepsky.git   # or: git pull
cd deepsky && uv sync
uv run pytest                                            # expect 23 passed

# 1. dataset in place (moved separately — see above)
find data/crops256 -name "*.webp" | wc -l               # expect 159462
ls data/crops256/manifest.csv

# 2. the overfit GATE — do not skip (~15 min). If it fails, stop and debug.
uv run pytest tests/test_overfit.py -m slow -s

# 3. overnight validation run. Time the first 200 steps, then size total_steps
#    in configs/mac-64px-validation.yaml to fit the night.
uv run python scripts/train.py --config configs/mac-64px-validation.yaml
```

Healthy-run loss values, visual milestones per step count, and red flags are in
the [README](README.md#2-validate-on-the-mac-mini) table. Remember: **point stars
sharpen last** — smeared stars at 30k steps is normal, not failure.

## Known caveat — MPS is the one untested path

The Intel dev machine has no Apple GPU, so the first real MPS execution happens on
the mini. Device handling auto-detects (cuda→mps→cpu) and is CPU-verified, and the
config keeps `amp: false` because MPS autocast is historically flaky. If anything
MPS-specific throws (a missing op, a dtype quirk), that's the likely culprit —
capture the error for debugging.

## Gotchas already hit (don't rediscover these)

- `.gitignore` `data/` once silently excluded `src/deepsky/data/` — patterns are
  now anchored (`/data/`). If you add source dirs, check `git status` includes them.
- ESO paginates archive category pages with `list/N/`, not `page/N/`.
- torch is pinned `<2.3` only on Intel macOS (dropped wheels); the mini/cloud get
  modern torch automatically via the platform markers in `pyproject.toml`.

## After the 64px run looks like space

Follow [cloud/README.md](cloud/README.md): tar `crops256/`, upload to object
storage, rent a 4090, run `configs/cloud-128px-full.yaml` under tmux with the
checkpoint-sync loop. Kill criteria are written down there — hold to them.
