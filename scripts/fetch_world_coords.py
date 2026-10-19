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

    unresolved = sorted(p for p, c in cache.items() if c is None)
    if unresolved:
        print(f"\n{len(unresolved)} unresolved (manual review needed):")
        for p in unresolved:
            print(" ", p)
    print(f"\nWrote coordinates for {len(places)} places to {CACHE_PATH}")


if __name__ == "__main__":
    main()
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.1)  # respect Nominatim's 1 req/sec usage policy

    unresolved = sorted(p for p, c in cache.items() if c is None)
    if unresolved:
        print(f"\n{len(unresolved)} unresolved (manual review needed):")
        for p in unresolved:
            print(" ", p)
    print(f"\nWrote coordinates for {len(places)} places to {CACHE_PATH}")


if __name__ == "__main__":
    main()
