"""Fetch Swedish Wikipedia pageview counts as a proxy for "how famous is
this place to a Swedish audience". Uses the public Wikimedia pageviews
REST API (no key required). Results are cached in data/fame_cache.json so
re-runs are fast and only fetch places not already cached.

A missing/404 sv.wikipedia article is treated as 0 views - itself a weak
signal that the place isn't well known to Swedes, not just a fetch failure.

Usage:
  python scripts/fetch_fame.py                  # fame for every place in resmal_regions.csv
  python scripts/fetch_fame.py extra.csv ...     # also fetch fame for the 'resmal' column of extra CSVs
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
CACHE_PATH = ROOT / "data" / "fame_cache.json"

USER_AGENT = "ontrack-resmal-fame/1.0 (one-off research script)"
PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "sv.wikipedia/all-access/user/{article}/monthly/{start}/{end}"
)
START, END = "20240101", "20241231"


def fetch_pageviews(place: str) -> int:
    title = urllib.parse.quote(place.replace(" ", "_"), safe="")
    url = PAGEVIEWS_URL.format(article=title, start=START, end=END)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.load(resp)
            return sum(item["views"] for item in data["items"])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0
            if e.code == 429:
                wait = float(e.headers.get("Retry-After", 5)) if e.headers else 5
                wait = max(wait, 5) * (attempt + 1)
                print(f"  ! rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"gave up on {place!r} after repeated 429s")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def ensure_fame(places, cache: dict = None) -> dict:
    """Fill (and return) the cache with fame scores for any of `places` not already cached.

    Only a confirmed 404 (no sv.wikipedia article) is cached as 0 views. Any
    other failure (e.g. repeated 429s even after retries) is left uncached
    so it gets retried on the next run, instead of being silently recorded
    as a false "0 views" - a real 0 should mean "no article", not "we
    couldn't reach the API".
    """
    if cache is None:
        cache = load_cache()
    todo = [p for p in places if p not in cache]
    for i, place in enumerate(todo, start=1):
        try:
            views = fetch_pageviews(place)
        except Exception as exc:
            print(f"  ! error fetching fame for {place!r}: {exc} - will retry next run")
            continue
        cache[place] = views
        print(f"[{i}/{len(todo)}] {place!r} -> {views} views/year")
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(2.0)
    return cache


def main():
    with REGIONS_CSV.open(encoding="utf-8") as f:
        places = [row["resmal"] for row in csv.DictReader(f)]
    for extra_path in sys.argv[1:]:
        with open(extra_path, encoding="utf-8") as f:
            places += [row["resmal"] for row in csv.DictReader(f)]
    places = sorted(set(places))
    print(f"{len(places)} places to ensure fame for")
    ensure_fame(places)
    print("Done.")


if __name__ == "__main__":
    main()
