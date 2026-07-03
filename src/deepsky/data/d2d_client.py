"""Client for the Djangoplicity "Data2Dome" JSON feed shared by
esahubble.org, esawebb.org, and eso.org, plus the category-page scraper.

Feed shape (verified 2026-07-02):
    {"Count": int, "Next": url | null, "Collections": [item, ...]}
Item: ID, Title, Description, Credit, Rights, PublicationDate,
      Assets[].Resources[] with ResourceType in
      {Original (tiff), Large (full-res jpg), Small (1280px), Thumbnail, Icon}.

The feed carries no category taxonomy, so deep-sky selection comes from
scraping the HTML archive category pages, whose thumbnails embed image IDs
(https://<site>/images/archive/category/<slug>/[page/N/]).
"""

from __future__ import annotations

import re
import time
from typing import Iterator

import requests

USER_AGENT = "deepsky-diffusion/0.1 (personal research; jess.holbrook@gmail.com)"
REQUEST_INTERVAL_S = 0.5

SOURCES = {
    "esahubble": {
        "feed": "https://esahubble.org/images/d2d/",
        "site": "https://esahubble.org",
        "categories": ["nebulae", "galaxies", "starclusters", "cosmology"],
        "page_fmt": "page/{n}/",
    },
    "esawebb": {
        "feed": "https://esawebb.org/images/d2d/",
        "site": "https://esawebb.org",
        "categories": ["nebulae", "galaxies"],
        "page_fmt": "page/{n}/",
    },
    "eso": {
        "feed": "https://www.eso.org/public/images/d2d/",
        "site": "https://www.eso.org/public",
        "categories": ["nebulae", "galaxies", "starclusters", "cosmology"],
        "page_fmt": "list/{n}/",
    },
    # NOIRLab runs the same CMS; endpoint unverified from this network.
    # Enable after `curl -s https://noirlab.edu/public/images/d2d/ | head` works.
}

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0


def _get(url: str, retries: int = 4, **kwargs) -> requests.Response:
    global _last_request
    for attempt in range(retries):
        wait = REQUEST_INTERVAL_S - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_request = time.time()
            resp = _session.get(url, timeout=60, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt * 2)
    raise RuntimeError("unreachable")


def iter_feed(feed_url: str) -> Iterator[dict]:
    """Yield every image record in a d2d feed, following pagination."""
    url = feed_url
    while url:
        data = _get(url).json()
        yield from data.get("Collections", [])
        url = data.get("Next")


def best_resource(item: dict) -> dict | None:
    """Pick the full-res JPG ('Large'); fall back to 'Small'. Never the TIFF."""
    resources = []
    for asset in item.get("Assets", []):
        if asset.get("MediaType") != "Image":
            continue
        resources.extend(asset.get("Resources", []))
    by_type = {r.get("ResourceType"): r for r in resources}
    return by_type.get("Large") or by_type.get("Small")


def scrape_category_ids(site_base: str, category: str, page_fmt: str = "page/{n}/") -> set[str]:
    """Collect image IDs from an archive category's paginated HTML."""
    ids: set[str] = set()
    page = 1
    while True:
        suffix = page_fmt.format(n=page) if page > 1 else ""
        url = f"{site_base}/images/archive/category/{category}/{suffix}"
        try:
            html = _get(url).text
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                break  # ran past the last page
            raise
        # esahubble/esawebb: .../archives/images/thumb300y/{id}.jpg
        # eso:               https://cdn.eso.org/images/thumb300y/{id}.jpg
        found = set(re.findall(r"(?:archives/)?images/thumb\w+/([A-Za-z0-9_\-.]+)\.\w+", html))
        if not found or found <= ids:
            break
        ids |= found
        page += 1
    return ids
