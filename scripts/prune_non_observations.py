"""Delete crops whose source image isn't a real observation (charts,
simulations, collages, artwork), using data/raw/image_types.json from
scripts/scrape_image_types.py. Rewrites the manifest to match.

NASA sources have no type page and are left untouched.

    uv run python scripts/prune_non_observations.py            # dry run
    uv run python scripts/prune_non_observations.py --apply
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

KEEP_TYPES = {"Observation", "Photographic"}
CROPS = Path("data/crops256")
TYPES = Path("data/raw/image_types.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    types = json.loads(TYPES.read_text())
    manifest_path = CROPS / "manifest.csv"
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    keep_rows, drop_rows = [], []
    for row in rows:
        image_type = types.get(row["source_key"])
        if image_type is None or image_type in KEEP_TYPES:  # None = NASA
            keep_rows.append(row)
        else:
            drop_rows.append(row)

    dropped_types = Counter(types[r["source_key"]] for r in drop_rows)
    dropped_sources = len({r["source_key"] for r in drop_rows})
    print(f"crops: keep {len(keep_rows)}, drop {len(drop_rows)} "
          f"(from {dropped_sources} source images)")
    print(f"dropped by type: {dict(dropped_types)}")

    if not args.apply:
        print("dry run — pass --apply to delete")
        return

    for row in drop_rows:
        (CROPS / row["crop"]).unlink(missing_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(keep_rows)
    print("pruned.")


if __name__ == "__main__":
    main()
