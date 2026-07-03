"""Metadata-level filtering: what gets downloaded at all."""

from __future__ import annotations

import re

MIN_SHORT_SIDE = 512

# Anything matching these in the title is not a photograph of the sky.
TITLE_BLACKLIST = re.compile(
    r"artist|impression|illustration|annotat|comparison|chart|diagram|logo|"
    r"screenshot|infographic|graphic|composite sketch|drawing|animation still|"
    r"spacecraft|telescope|mirror|instrument|launch|cleanroom|clean room|"
    r"ground-based view|wide-field view of the sky around|"  # ESO finder charts
    r"\bspectr|light curve",  # spectra/spectrograms are plots, not photos
    re.IGNORECASE,
)

# Descriptions legitimately mention hardware ("imaged by the James Webb Space
# Telescope"), so only unambiguous not-a-photo markers apply there.
DESCRIPTION_BLACKLIST = re.compile(
    r"artist'?s? (?:impression|concept|rendering)|illustration of|annotated|infographic",
    re.IGNORECASE,
)

# NASA search results need positive evidence too — the library is full of
# events, people, and hardware that happen to mention "galaxy" etc.
NASA_KEYWORD_WHITELIST = re.compile(
    r"nebula|galax|cluster|supernova|deep field|hubble|webb|jwst|chandra|"
    r"spitzer|interstellar|star[- ]forming",
    re.IGNORECASE,
)
NASA_CENTER_BLACKLIST = {"HQ", "KSC", "JSC", "SSC", "AFRC"}  # mostly people/hardware photos


def title_ok(title: str, description: str = "") -> bool:
    return not TITLE_BLACKLIST.search(title or "") and not DESCRIPTION_BLACKLIST.search(
        (description or "")[:300]
    )


def dimensions_ok(dimensions) -> bool:
    if not dimensions or len(dimensions) < 2:
        return True  # unknown — keep; pixel check happens at preprocess time
    try:
        return min(float(dimensions[0]), float(dimensions[1])) >= MIN_SHORT_SIDE
    except (TypeError, ValueError):
        return True


def nasa_hit_ok(hit: dict) -> bool:
    if hit.get("center") in NASA_CENTER_BLACKLIST:
        return False
    haystack = " ".join([hit.get("title", ""), " ".join(hit.get("keywords") or [])])
    if not NASA_KEYWORD_WHITELIST.search(haystack):
        return False
    return title_ok(hit.get("title", ""), hit.get("description", ""))
