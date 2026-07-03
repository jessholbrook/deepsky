"""Scrape the AVM image Type (Observation / Collage / Simulation / Chart /
Artwork...) from each d2d source image's public page. The d2d feed doesn't
carry this field, and it's the only reliable way to separate real photographs
of the sky from figures and composites.

Writes data/raw/image_types.json  {"eso/eso9631a": "Simulation", ...}
Resumable: already-scraped ids are skipped.

    uv run python scripts/scrape_image_types.py
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

from deepsky.data.d2d_client import SOURCES, _get

# "About the Image" table row. esahubble/eso render `Type:</th><td>X`,
# esawebb spreads it over lines: `Type: </th> <td colspan="2"> X `.
# Only the FIRST Type row is the image type (later ones describe the object).
TYPE_RE = re.compile(r"Type:\s*</t[hd]>\s*<td[^>]*>\s*([^<]+?)\s*<")
OUT = Path("data/raw/image_types.json")


def image_page_url(source: str, image_id: str) -> str:
    return f"{SOURCES[source]['site']}/images/{image_id}/"


def main():
    types: dict[str, str] = json.loads(OUT.read_text()) if OUT.exists() else {}

    todo = []
    for source in SOURCES:
        meta = Path(f"data/raw/{source}/metadata.jsonl")
        if not meta.exists():
            continue
        for line in meta.read_text().splitlines():
            rec = json.loads(line)
            key = f"{source}/{rec['id']}"
            if key not in types:
                todo.append((source, rec["id"], key))

    print(f"{len(types)} cached, {len(todo)} to scrape")
    for n, (source, image_id, key) in enumerate(tqdm(todo, desc="types")):
        try:
            html = _get(image_page_url(source, image_id)).text
            match = TYPE_RE.search(html)
            types[key] = match.group(1).strip() if match else "Unknown"
        except Exception as e:
            tqdm.write(f"  FAILED {key}: {e}")
            types[key] = "Error"
        if n % 200 == 0:
            OUT.write_text(json.dumps(types, indent=0))

    OUT.write_text(json.dumps(types, indent=0))
    from collections import Counter

    print(Counter(types.values()).most_common())


if __name__ == "__main__":
    main()
