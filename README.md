# deepsky

A from-scratch DDPM (denoising diffusion) model that generates deep-sky
astronomical images — nebulae, galaxies, star clusters — trained on public
imagery from ESA/Hubble, ESA/Webb, ESO, and NASA.

Everything (U-Net, noise schedules, samplers, training loop) is hand-written
PyTorch — no diffusers dependency.

## Workflow

The project is built around a three-machine split:

1. **Any machine** — data pipeline + tests (CPU only)
2. **Mac mini (Apple Silicon)** — overnight 64px validation runs on MPS
3. **Rented RTX 4090** — the real 128px run (~$10–20, see [cloud/README.md](cloud/README.md))

## Setup

```bash
uv sync
uv run pytest          # fast suite, ~30s
```

## 1. Build the dataset (once, ~hours of downloading)

```bash
uv run python scripts/download_all.py            # → data/raw/{source}/, resumable
uv run python scripts/build_dataset.py           # dedup + derive 256px crops → data/crops256/
uv run python scripts/scrape_image_types.py      # AVM Type per ESA/ESO image (~45 min)
uv run python scripts/prune_non_observations.py --apply   # drop chart/simulation/collage crops
```

## 2. Validate on the Mac mini

```bash
# The gate: a tiny model must memorize 8 images (~15 min). If this fails, stop.
uv run pytest tests/test_overfit.py -m slow -s

# Time 200 steps, then size total_steps to fit the night (see config comment):
uv run python scripts/train.py --config configs/mac-64px-validation.yaml
```

What a healthy 64px run looks like (loss + `runs/mac64/samples_*.png` grids):

| Step | Expectation |
|---|---|
| init | loss ≈ 1.0 (zero-init output head predicts nothing) |
| 2k | loss ≈ 0.15; grids = colored blobs with a space palette (blacks + nebula hues) |
| 10k | nebula-like textures, obviously "space" statistics |
| 30k | coherent structures: gas filaments, galaxy smudges, cluster densities |
| 75k | loss plateau ~0.03–0.06; some samples pass as real thumbnails |

**Point stars sharpen last** — smeared stars at 30k steps is normal, not failure.
Red flags: loss stuck > 0.1 at 10k, NaN, all-gray grids, EMA grids worse than raw.

## 3. The real run

Follow [cloud/README.md](cloud/README.md). Resume from any checkpoint:

```bash
uv run python scripts/train.py --config configs/cloud-128px-full.yaml \
    --resume runs/cloud128/ckpt_0100000.pt
```

## 4. Generate

```bash
uv run python scripts/sample.py --config configs/cloud-128px-full.yaml \
    --ckpt runs/cloud128/ckpt_0200000.pt --n 64 --sampler ddpm --out space.png
```

## Dataset attribution

Training data is derived from publicly released imagery:

- **ESA/Hubble** (esahubble.org) — CC BY 4.0
- **ESA/Webb** (esawebb.org) — CC BY 4.0
- **ESO** (eso.org) — CC BY 4.0
- **NASA Image and Video Library** (images.nasa.gov) — public domain

Per-image credits are preserved in `data/crops256/manifest.csv`. If you
publish samples or weights, retain this attribution list.
