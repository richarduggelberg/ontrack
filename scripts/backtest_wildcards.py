"""Walk-forward backtest of the fame index specifically for first-time
destinations: for every genuinely new place (Europe/outside-Europe) the show
has ever used, was it a capital/population>=1,000,000 city (i.e. eligible
for the wildcards pool at all), and if so, how highly would Swedish
Wikipedia fame have ranked it among every other not-yet-seen eligible place
at that point in the show's history? Compares the fame-ranked hit rate
against the pool-size/2 rank you'd expect from picking at random.

Usage: python scripts/backtest_wildcards.py [--seasons N] [--top N]
"""
import argparse
import csv
import unicodedata
from collections import defaultdict
from pathlib import Path

from build_wildcards import (
    EXCLUDE_NAMES,
    classify,
    load_country_capitals,
    load_country_continents,
    load_raw_candidates,
)
from fetch_fame import ensure_fame, load_cache

ROOT = Path(__file__).resolve().parent.parent
RESMAL_CSV = ROOT / "data" / "resmal.csv"
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"

FAME_REGIONS = ("europe", "outside_europe")


def normalize(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=15, help="how many of the most recent seasons to backtest")
    parser.add_argument("--top", type=int, default=15, help="top-N cutoff to count as a 'hit'")
    args = parser.parse_args()

    with REGIONS_CSV.open(encoding="utf-8") as f:
        known_places = {row["resmal"] for row in csv.DictReader(f)}
    known_normalized = {normalize(p) for p in known_places}

    continents = load_country_continents()
    country_capitals = load_country_capitals()
    exclude_names = set(EXCLUDE_NAMES)
    for place in known_places:
        capital = country_capitals.get(place.lower())
        if capital:
            exclude_names.add(capital)

    raw = load_raw_candidates(continents, exclude_names)
    # Resolve each GeoNames candidate to the show's own spelling when it's an
    # accent/case variant of an already-known place (e.g. Reykjavík -> Reykjavik),
    # so training counts line up with the show's actual resmal spelling.
    known_by_normalized = {normalize(p): p for p in known_places}
    for r in raw:
        r["resolved"] = known_by_normalized.get(normalize(r["resmal"]), r["resmal"])

    print(f"{len(raw)} capital/population>=1,000,000 candidates in the full GeoNames universe, fetching fame...")
    fame = ensure_fame([r["resolved"] for r in raw], load_cache())
    for r in raw:
        r["fame"] = fame.get(r["resolved"], 0)
    by_region = defaultdict(list)
    for r in raw:
        by_region[r["region"]].append(r)

    with RESMAL_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    resa_rows = [r for r in all_rows if r["typ"] == "resa"]
    with REGIONS_CSV.open(encoding="utf-8") as f:
        region_by_place = {row["resmal"]: row["region"] for row in csv.DictReader(f)}

    episode_index = {}
    episode_order = []
    for r in resa_rows:
        key = (r["sasong"], r["avsnitt"])
        if key not in episode_index:
            episode_index[key] = len(episode_index)
            episode_order.append(key)

    seasons_sorted = sorted({int(s) for s, _ in episode_order})
    backtest_seasons = set(seasons_sorted[-args.seasons:])
    print(f"Backtesting first-timers in seasons {sorted(backtest_seasons)} ({args.seasons} most recent)\n")

    seen_so_far = defaultdict(int)  # resolved place -> count, updated as we walk forward
    results = {region: [] for region in FAME_REGIONS}  # list of (rank, pool_size)
    ineligible = {region: 0 for region in FAME_REGIONS}

    for key in episode_order:
        actual = [r["resmal"] for r in resa_rows if (r["sasong"], r["avsnitt"]) == key]
        in_window = int(key[0]) in backtest_seasons

        if in_window:
            for place in actual:
                region = region_by_place.get(place)
                is_first_timer = seen_so_far[place] == 0
                if is_first_timer and region in FAME_REGIONS:
                    candidate = next((c for c in by_region[region] if c["resolved"] == place), None)
                    if candidate is None:
                        ineligible[region] += 1
                    else:
                        pool = [c for c in by_region[region] if seen_so_far[c["resolved"]] == 0]
                        pool.sort(key=lambda c: c["fame"], reverse=True)
                        rank = next(i for i, c in enumerate(pool, start=1) if c["resolved"] == place)
                        results[region].append((rank, len(pool)))

        # advance time: mark everything aired in this episode as seen, whether
        # or not it's in the backtest window (we still need pre-window history
        # to know what's already "known" once the window starts).
        for place in actual:
            seen_so_far[place] += 1

    print(f"{'Region':<16}{'N':>5}{'Eligible':>10}{'Top1':>7}{'Top5':>7}{f'Top{args.top}':>7}{'Median rank':>13}{'Median pool':>13}{'Rand. exp.':>11}")
    for region in FAME_REGIONS:
        ranked = results[region]
        n_first_timers = len(ranked) + ineligible[region]
        n_eligible = len(ranked)
        if n_eligible == 0:
            print(f"{region:<16}{n_first_timers:>5}{n_eligible:>10}{'--':>7}{'--':>7}{'--':>7}{'--':>13}{'--':>13}{'--':>11}")
            continue
        ranks = sorted(r for r, _ in ranked)
        pools = sorted(p for _, p in ranked)
        top1 = sum(1 for r in ranks if r <= 1)
        top5 = sum(1 for r in ranks if r <= 5)
        topN = sum(1 for r in ranks if r <= args.top)
        median_rank = ranks[len(ranks) // 2]
        median_pool = pools[len(pools) // 2]
        random_expected = median_pool / 2
        print(f"{region:<16}{n_first_timers:>5}{n_eligible:>10}{top1/n_eligible*100:>6.0f}%{top5/n_eligible*100:>6.0f}%{topN/n_eligible*100:>6.0f}%{median_rank:>13}{median_pool:>13}{random_expected:>11.0f}")

    print("\n'Eligible' = was a capital or population>=1,000,000 city (the only kind of")
    print("place the wildcards pool can ever surface); the rest are small towns no")
    print("fame-based wildcard mechanism could have predicted.")
    print("'Rand. exp.' = the median rank a dice roll would get from the same pool size -")
    print("compare to 'Median rank' to see if fame actually beats random chance.")


if __name__ == "__main__":
    main()
