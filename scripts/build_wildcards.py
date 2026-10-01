"""Build a candidate pool of world capitals + very large cities (population
>= 1,000,000) that have NEVER appeared on the show, ranked by Swedish
Wikipedia fame (see fetch_fame.py). These are the "could be a first-timer"
wildcards build_predictions.py surfaces alongside the history-based
rankings, to cover destinations a pure frequency model can never predict.

Source: GeoNames cities15000 dump (free, no key) for capitals/population,
geonames countryInfo.txt for country -> continent (same approach as
classify_destinations.py). Sweden is excluded entirely - the Sweden slot
can be literally any town, so a "famous/capital" filter doesn't apply there.

Usage: python scripts/build_wildcards.py
"""
import csv
import io
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

from fetch_fame import ensure_fame, load_cache


def normalize(name: str) -> str:
    """Accent/case-insensitive key so e.g. 'Reykjavík' matches known 'Reykjavik'."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()

ROOT = Path(__file__).resolve().parent.parent
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
OUT_PATH = ROOT / "data" / "wildcards.csv"

USER_AGENT = "ontrack-resmal-wildcards/1.0 (one-off research script)"
COUNTRY_INFO_URL = "http://download.geonames.org/export/dump/countryInfo.txt"
CITIES_ZIP_URL = "http://download.geonames.org/export/dump/cities15000.zip"
POPULATION_THRESHOLD = 1_000_000

# GeoNames' "name" column is usually the local/English form, but the show
# (and Swedish Wikipedia) uses the Swedish exonym for many capitals - without
# this, those candidates both get a bad fame lookup (wrong article title) and
# wrongly look like "never seen before" when the Swedish name already has
# show history (e.g. "Vienna" vs. the already-known "Wien").
EXONYM_MAP = {
    "Vienna": "Wien",
    "Brussels": "Bryssel",
    "Moscow": "Moskva",
    "Rome": "Rom",
    "Prague": "Prag",
    "Copenhagen": "Köpenhamn",
    "Athens": "Aten",
    "Lisbon": "Lissabon",
    "Bucharest": "Bukarest",
    "Belgrade": "Belgrad",
    "Warsaw": "Warszawa",
    "Cairo": "Kairo",
    "Beijing": "Peking",
    "Tehran": "Teheran",
    "Damascus": "Damaskus",
    "Baghdad": "Bagdad",
    "Helsinki": "Helsingfors",
    "Ulaanbaatar": "Ulan Bator",
    "Addis Ababa": "Addis Abeba",
    "Ho Chi Minh City": "Ho Chi Minh-staden",
    "Munich": "München",
    "Milan": "Milano",
    "Algiers": "Alger",
    "Havana": "Havanna",
    "Kyiv": "Kiev",
    "Odesa": "Odessa",
    # Faroese spelling vs. the Swedish exonym the show/known-places list uses.
    "Tórshavn": "Torshamn",
}

# GeoNames lists these as separate large-population "cities", but they're
# boroughs/districts/sub-areas of a place already known to the show under a
# different name (New York, Hong Kong, Budapest) - not genuine first-timers.
# Longyearbyen is also here: the show's entry is "Svalbard" (the territory),
# which doesn't textually match its capital city at all.
EXCLUDE_NAMES = {
    "Pest", "Manhattan", "Brooklyn", "Queens", "The Bronx", "New York City",
    "Kowloon", "New Territories", "Hong Kong Island",
    "Longyearbyen",
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
        mapping[fields[0].strip().lower()] = fields[8].strip()
    return mapping


def load_country_capitals() -> dict:
    """country name (lowercase) -> capital city name, for destinations the show
    recorded as a country/territory name rather than a specific city (e.g.
    "Fiji", "Qatar", "Puerto Rico") so their capital isn't mistaken for a
    genuine first-timer."""
    req = urllib.request.Request(COUNTRY_INFO_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    mapping = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        mapping[fields[4].strip().lower()] = fields[5].strip()
    return mapping


def classify(country_code, continents) -> str:
    if country_code == "se":
        return "sweden"
    return "europe" if continents.get(country_code) == "EU" else "outside_europe"


def main():
    continents = load_country_continents()
    country_capitals = load_country_capitals()

    with REGIONS_CSV.open(encoding="utf-8") as f:
        known_places = {row["resmal"] for row in csv.DictReader(f)}
    known_normalized = {normalize(p) for p in known_places}

    # Places the show recorded as a country/territory name (e.g. "Fiji",
    # "Qatar") rather than a city - their capital would otherwise look like
    # an unseen wildcard.
    exclude_names = set(EXCLUDE_NAMES)
    for place in known_places:
        capital = country_capitals.get(place.lower())
        if capital:
            exclude_names.add(capital)

    print("Downloading GeoNames cities15000 dump...")
    req = urllib.request.Request(CITIES_ZIP_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        text = zf.read("cities15000.txt").decode("utf-8")

    candidates = {}  # name -> (country_code, population)
    for line in text.splitlines():
        fields = line.split("\t")
        name = EXONYM_MAP.get(fields[1], fields[1])
        country_code = fields[8].strip().lower()
        feature_code = fields[7]
        population = int(fields[14] or 0)
        if country_code == "se" or name in known_places or name in exclude_names:
            continue
        if normalize(name) in known_normalized:
            continue
        if not (feature_code == "PPLC" or population >= POPULATION_THRESHOLD):
            continue
        # keep the highest-population entry if the same name appears twice
        if name not in candidates or population > candidates[name][1]:
            candidates[name] = (country_code, population)

    rows = []
    for name, (country_code, population) in candidates.items():
        region = classify(country_code, continents)
        if region == "sweden":
            continue
        rows.append({"resmal": name, "country_code": country_code, "region": region, "population": population})

    print(f"{len(rows)} wildcard candidates (capitals / population >= {POPULATION_THRESHOLD:,}), fetching fame...")
    fame = ensure_fame([r["resmal"] for r in rows], load_cache())

    for r in rows:
        r["fame"] = fame.get(r["resmal"], 0)
    rows.sort(key=lambda r: r["fame"], reverse=True)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resmal", "country_code", "region", "population", "fame"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} wildcard candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()
