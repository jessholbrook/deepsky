"""Client for the NASA Image and Video Library API (images-api.nasa.gov).

Search returns Collection+JSON; per-hit originals come from the asset
manifest endpoint. All NASA media is public domain per NASA media guidelines.
"""

from __future__ import annotations

from typing import Iterator

from deepsky.data.d2d_client import _get

SEARCH_URL = "https://images-api.nasa.gov/search"
ASSET_URL = "https://images-api.nasa.gov/asset/{nasa_id}"

SEARCH_TERMS = [
    "nebula",
    "galaxy",
    "star cluster",
    "globular cluster",
    "supernova remnant",
    "deep field",
]

# The API rejects requests beyond page_size * page = 10_000.
PAGE_SIZE = 100
MAX_PAGES = 100


def iter_search(term: str) -> Iterator[dict]:
    """Yield search hits: {nasa_id, title, description, keywords, center}."""
    for page in range(1, MAX_PAGES + 1):
        resp = _get(
            SEARCH_URL,
            params={"q": term, "media_type": "image", "page": page, "page_size": PAGE_SIZE},
        )
        items = resp.json().get("collection", {}).get("items", [])
        if not items:
            return
        for item in items:
            data = (item.get("data") or [{}])[0]
            if data.get("media_type") != "image":
                continue
            yield {
                "nasa_id": data.get("nasa_id"),
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "keywords": data.get("keywords", []),
                "center": data.get("center", ""),
            }


def original_url(nasa_id: str) -> str | None:
    """Resolve the largest JPG from the asset manifest (skip TIFFs — huge)."""
    manifest = _get(ASSET_URL.format(nasa_id=nasa_id)).json()
    urls = [i.get("href", "") for i in manifest.get("collection", {}).get("items", [])]
    for suffix in ("~orig.jpg", "~large.jpg", "~medium.jpg"):
        for url in urls:
            if url.endswith(suffix):
                return url
    return None
