
"""Fetch lat/lon for every Europe / outside-Europe destination (Nominatim),
cached in data/world_coords_cache.json, so build_predictions.py can penalise
candidates that sit very close to a destination shown in a recent episode.

Uses each place's country code from data/resmal_regions.csv to disambiguate
Nominatim results (e.g. "Alexandria" in Egypt vs. elsewhere).

Usage: python scripts/fetch_world_coords.py
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
CACHE_PATH = ROOT / "data" / "world_coords_cache.json"

USER_AGENT = "ontrack-resmal-geoclassify/1.0 (one-off research script)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Places whose names Nominatim can't reliably resolve to the intended spot
# (translated/landmark names). Pinned so the proximity penalty is anchored
# correctly and re-runs stay deterministic.
COORD_OVERRIDES = {
    "Alexandria": {"lat": 31.199181, "lon": 29.895172},
    "Heraklion": {"lat": 35.339080, "lon": 25.133284},
    "Omaha Beach": {"lat": 49.370552, "lon": -0.878939},
    "Parthenon": {"lat": 37.971503, "lon": 23.726618},
    "Pyramiderna": {"lat": 29.979165, "lon": 31.134213},
    "Santiago": {"lat": 42.906458, "lon": -8.519298},
    "Titanic": {"lat": 54.608143, "lon": -5.909836},
    "Waterloo": {"lat": 50.703286, "lon": 4.408428},
    "Panama City": {"lat": 8.971449, "lon": -79.534180},
}


def geocode(place: str, country_code: str):
    params = {"q": place, "format": "json", "limit": 1}
    if country_code:
        params["countrycodes"] = country_code
    req = urllib.request.Request(
        f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data:
        return None
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}


def main():
    with REGIONS_CSV.open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["region"] in ("europe", "outside_europe")]
    country_code = {row["resmal"]: row["country_code"] for row in rows}
    places = sorted(country_code)
    print(f"{len(places)} Europe/outside-Europe places to ensure coordinates for")

    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    # Pin coordinates for names Nominatim can't reliably resolve by name.
    cache.update(COORD_OVERRIDES)

    for i, place in enumerate(places, start=1):
        if place in cache:
            continue
        cc = country_code.get(place)
        try:
            coords = geocode(place, cc)
        except Exception as exc:
            print(f"  ! error geocoding {place!r}: {exc}")
            coords = None
        cache[place] = coords
        print(f"[{i}/{len(places)}] {place!r} ({cc}) -> {coords}")
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.1)  # respect Nominatim's 1 req/sec usage policy

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    unresolved = sorted(p for p, c in cache.items() if c is None)
    if unresolved:
        print(f"\n{len(unresolved)} unresolved (manual review needed):")
        for p in unresolved:
            print(" ", p)
    print(f"\nWrote coordinates for {len(places)} places to {CACHE_PATH}")


if __name__ == "__main__":
    main()













