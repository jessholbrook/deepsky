"""Content-based curation pass: score each SOURCE image with CLIP and flag
figure-like sources (annotated overlays, heatmaps, diagrams) that metadata
filtering can't catch. Scores up to 3 crops per source and averages.

Requires the curation dependency group:  uv sync --group curation

    uv run python scripts/clip_filter.py                 # score + write suspects.csv
    uv run python scripts/clip_filter.py --apply         # also delete flagged crops
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

CROPS = Path("data/crops256")

GOOD_PROMPTS = [
    "a telescope photograph of deep space with stars, galaxies, or nebulae",
    "an astrophotograph of the night sky",
]
BAD_PROMPTS = [
    "a scientific chart, diagram, or graph",
    "an annotated figure with markers, circles, and labels",
    "a pixelated false-color data heatmap",
    "a map with text labels",
]
# A source is flagged when mean P(bad) exceeds this.
THRESHOLD = 0.72


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = model.encode_text(tokenizer(GOOD_PROMPTS + BAD_PROMPTS))
    text = text / text.norm(dim=-1, keepdim=True)

    with open(CROPS / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source_key"]].append(r["crop"])

    scores: dict[str, float] = {}
    for key, crops in tqdm(by_source.items(), desc="clip"):
        imgs = []
        for name in crops[:3]:
            try:
                imgs.append(preprocess(Image.open(CROPS / name).convert("RGB")))
            except Exception:
                pass
        if not imgs:
            continue
        feats = model.encode_image(torch.stack(imgs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probs = (100 * feats @ text.T).softmax(dim=-1)
        p_bad = probs[:, len(GOOD_PROMPTS):].sum(dim=-1).mean().item()
        scores[key] = p_bad

    flagged = {k: v for k, v in scores.items() if v >= args.threshold}
    out = CROPS / "clip_suspects.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_key", "p_bad"])
        for k, v in sorted(flagged.items(), key=lambda kv: -kv[1]):
            w.writerow([k, f"{v:.3f}"])
    n_crops = sum(len(by_source[k]) for k in flagged)
    print(f"flagged {len(flagged)} of {len(by_source)} sources ({n_crops} crops) -> {out}")

    if not args.apply:
        print("dry run — inspect suspects, then pass --apply")
        return

    drop = {name for k in flagged for name in by_source[k]}
    for name in drop:
        (CROPS / name).unlink(missing_ok=True)
    keep_rows = [r for r in rows if r["crop"] not in drop]
    with open(CROPS / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(keep_rows)
    print(f"pruned. {len(keep_rows)} crops remain.")


if __name__ == "__main__":
    main()
