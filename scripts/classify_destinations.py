"""Classify each unique 'resmal' destination as Sweden / Europe / Outside Europe.

Uses OpenStreetMap Nominatim (rate-limited to 1 req/sec per its usage policy)
to geocode each place name to a country code, then geonames countryInfo.txt
to map country -> continent. Results are cached to data/geo_cache.json so
re-runs are fast and don't re-hit the API.

Usage: python scripts/classify_destinations.py
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "resmal.csv"
CACHE_PATH = ROOT / "data" / "geo_cache.json"
OUT_PATH = ROOT / "data" / "resmal_regions.csv"

USER_AGENT = "ontrack-resmal-geoclassify/1.0 (one-off research script)"
COUNTRY_INFO_URL = "http://download.geonames.org/export/dump/countryInfo.txt"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Manual fixes for places Nominatim resolves ambiguously (picks a same-named
# place in the wrong country) or can't resolve at all (landmark-style names).
# Maps place -> final region, applied after geocoding/classification.
REGION_OVERRIDES = {
    "Boden": "sweden",            # resolved to a German hamlet
    "Malung": "sweden",           # resolved to somewhere in the US
    "Falkenberg": "sweden",       # resolved to France
    "Mora": "sweden",             # resolved to Spain
    "Sala": "sweden",              # resolved to Indonesia
    "Sandhamn": "sweden",          # resolved to Finland
    "Solna": "sweden",             # resolved to Slovakia
    "Santiago": "europe",          # resolved to Costa Rica; likely Santiago de Compostela
    "Nimis": "sweden",            # resolved to a town in Italy; means the Kullaberg sculpture
    "Vitön": "sweden",            # resolved to Norway; it's a Höga Kusten island
    "Alexandria": "outside_europe",  # resolved to Italy; show means Egypt
    "Heraklion": "europe",        # resolved to Egypt; it's in Crete, Greece
    "Niagarafallen": "outside_europe",
    "Omaha Beach": "europe",          # resolved to the US; it's in Normandy, France
    "Parthenon": "europe",            # resolved to the Nashville replica; means Athens
    "Pyramiderna": "outside_europe",  # resolved to Sweden; means the pyramids of Giza
    "Pyonyang": "outside_europe",     # typo for Pyongyang, unresolved
    "Titanic": "europe",              # resolved to Czech Republic; likely Titanic Belfast
    "Waterloo": "europe",             # resolved to the US; likely the Belgian battlefield
    "Istanbul": "europe",            # Turkey is geonames-Asia, but its major sights sit on the European side of the Bosphorus
    "Blå moskén, Istanbul": "europe",
    "Hollywoodskylten": "outside_europe",
    "Konungarnas dal": "outside_europe",
    "Louvren i Paris": "europe",
    "Lusail Iconic Stadium": "outside_europe",
    "Marianergraven": "outside_europe",
    "Slottet Bran": "europe",         # Bran Castle, Romania
    "Terrakottarmén i Xi'an": "outside_europe",  # Terracotta Army, China
    "Trevifontänen i Rom": "europe",  # Trevi Fountain, Rome
    "Woodstockfestivalen": "outside_europe",  # Woodstock, New York, USA
    "Ön Iwo Jima i Japan": "outside_europe",  # Iwo Jima, Japan
}


def load_country_continents() -> dict:
    req = urllib.request.Request(COUNTRY_INFO_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    mapping = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        iso2 = fields[0].strip().lower()
        continent = fields[8].strip()
        mapping[iso2] = continent
    return mapping


def geocode(place: str):
    params = urllib.parse.urlencode({
        "q": place, "format": "json", "limit": 1, "addressdetails": 1,
    })
    req = urllib.request.Request(f"{NOMINATIM_URL}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data:
        return None
    return data[0].get("address", {}).get("country_code")


def classify(country_code, continents: dict) -> str:
    if country_code is None:
        return "unknown"
    if country_code == "se":
        return "sweden"
    continent = continents.get(country_code)
    return "europe" if continent == "EU" else "outside_europe"


def main():
    continents = load_country_continents()

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unique_places = sorted({r["resmal"] for r in rows})
    print(f"{len(unique_places)} unique destinations to classify")

    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    for i, place in enumerate(unique_places, start=1):
        if place in cache:
            continue
        try:
            country_code = geocode(place)
        except Exception as exc:
            print(f"  ! error geocoding {place!r}: {exc}")
            country_code = None
        cache[place] = country_code
        print(f"[{i}/{len(unique_places)}] {place!r} -> {country_code}")
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.1)  # respect Nominatim's 1 req/sec usage policy

    region_by_place = {place: classify(cache.get(place), continents) for place in unique_places}
    region_by_place.update({p: r for p, r in REGION_OVERRIDES.items() if p in region_by_place})

    unknown = sorted(p for p, r in region_by_place.items() if r == "unknown")
    print(f"\n{len(unknown)} unresolved destinations (manual review needed):")
    for p in unknown:
        print(" ", p)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resmal", "country_code", "region"])
        for place in unique_places:
            writer.writerow([place, cache.get(place) or "", region_by_place[place]])
    print(f"\nWrote region classification for {len(unique_places)} places to {OUT_PATH}")


if __name__ == "__main__":
    main()
