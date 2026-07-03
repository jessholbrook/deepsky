"""Download orchestrator: d2d sources (category-whitelisted) + NASA search.

Layout:
    data/raw/{source}/{image_id}.jpg
    data/raw/{source}/metadata.jsonl   (one record per KEPT image)

Idempotent: existing files are skipped, metadata is rewritten from scratch
each run (it's cheap — the feed walk happens anyway).
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from deepsky.data import d2d_client, nasa_client
from deepsky.data.d2d_client import _get
from deepsky.data.filtering import dimensions_ok, nasa_hit_ok, title_ok


def _download_file(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = _get(url)
    except Exception as e:
        tqdm.write(f"  FAILED {url}: {e}")
        return False
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.rename(dest)
    return True


def download_d2d_source(name: str, raw_root: Path) -> None:
    spec = d2d_client.SOURCES[name]
    out_dir = raw_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] scraping category pages: {spec['categories']}")
    keep_ids: set[str] = set()
    for cat in spec["categories"]:
        ids = d2d_client.scrape_category_ids(spec["site"], cat, spec.get("page_fmt", "page/{n}/"))
        print(f"[{name}]   {cat}: {len(ids)} ids")
        keep_ids |= ids

    print(f"[{name}] walking feed {spec['feed']}")
    kept = []
    for item in tqdm(d2d_client.iter_feed(spec["feed"]), desc=f"{name} feed"):
        if item.get("ID") not in keep_ids:
            continue
        if not title_ok(item.get("Title", ""), item.get("Description", "")):
            continue
        resource = d2d_client.best_resource(item)
        if resource is None or not dimensions_ok(resource.get("Dimensions")):
            continue
        kept.append((item, resource))

    print(f"[{name}] downloading {len(kept)} images")
    with open(out_dir / "metadata.jsonl", "w") as meta:
        for item, resource in tqdm(kept, desc=f"{name} download"):
            dest = out_dir / f"{item['ID']}.jpg"
            if not _download_file(resource["URL"], dest):
                continue
            meta.write(
                json.dumps(
                    {
                        "id": item["ID"],
                        "source": name,
                        "title": item.get("Title"),
                        "credit": item.get("Credit"),
                        "rights": item.get("Rights"),
                        "url": resource["URL"],
                        "dimensions": resource.get("Dimensions"),
                        "reference": item.get("ReferenceURL"),
                    }
                )
                + "\n"
            )


def download_nasa(raw_root: Path) -> None:
    out_dir = raw_root / "nasa"
    out_dir.mkdir(parents=True, exist_ok=True)

    hits: dict[str, dict] = {}
    for term in nasa_client.SEARCH_TERMS:
        for hit in tqdm(nasa_client.iter_search(term), desc=f"nasa search '{term}'"):
            if hit["nasa_id"] and hit["nasa_id"] not in hits and nasa_hit_ok(hit):
                hits[hit["nasa_id"]] = hit

    print(f"[nasa] downloading {len(hits)} images")
    with open(out_dir / "metadata.jsonl", "w") as meta:
        for nasa_id, hit in tqdm(hits.items(), desc="nasa download"):
            dest = out_dir / f"{nasa_id}.jpg"
            if not dest.exists():
                try:
                    url = nasa_client.original_url(nasa_id)
                except Exception as e:
                    tqdm.write(f"  FAILED manifest {nasa_id}: {e}")
                    continue
                if url is None or not _download_file(url, dest):
                    continue
            meta.write(
                json.dumps(
                    {
                        "id": nasa_id,
                        "source": "nasa",
                        "title": hit.get("title"),
                        "credit": "NASA",
                        "rights": "Public domain (NASA media guidelines)",
                        "keywords": hit.get("keywords"),
                    }
                )
                + "\n"
            )
