"""Walk-forward backtest of the prediction model in build_predictions.py.

For each real episode in the last N seasons, train the same
count * recency-cooldown scoring model using only episodes that aired
strictly before it, then check where that episode's actual Sweden /
Europe / outside-Europe destinations ranked in the resulting list.
This answers "if we had generated predictions.json right before this
episode aired, how good would they actually have been?" - a much
stronger check than eyeballing a single holdout episode.

Usage: python scripts/backtest_predictions.py [--seasons N] [--top N]
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from fetch_fame import load_cache as load_fame_cache

ROOT = Path(__file__).resolve().parent.parent
RESMAL_CSV = ROOT / "data" / "resmal.csv"
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"

COOLDOWN_EPISODES = 20
REGIONS = ("sweden", "europe", "outside_europe")
# Sweden can be any town at all (per manual review), so fame isn't a useful
# discriminator there - only apply the fame boost to Europe/outside-Europe.
FAME_REGIONS = ("europe", "outside_europe")

FAME = load_fame_cache()
MAX_FAME = max(FAME.values()) if FAME else 1


def fame_multiplier(place, region):
    """1x (no change) for obscure places, up to 2x for the most sv.wikipedia-famous ones."""
    if region not in FAME_REGIONS:
        return 1.0
    return 1.0 + math.log1p(FAME.get(place, 0)) / math.log1p(MAX_FAME)


# Candidate scoring strategies, each: (place, region, count, episodes_since) -> weight.
# "current" is the formula live in build_predictions.py today.
STRATEGIES = {
    "current": lambda p, r, c, since: c * min(1.0, since / COOLDOWN_EPISODES),
    "recency_only": lambda p, r, c, since: since,
    "count_only": lambda p, r, c, since: c,
    "recency_sqrt_count": lambda p, r, c, since: since / (c ** 0.5),
    "recency_div_count": lambda p, r, c, since: since / c,
    "sqrt_count_cooldown": lambda p, r, c, since: (c ** 0.5) * min(1.0, since / COOLDOWN_EPISODES),
    "log_count_cooldown": lambda p, r, c, since: math.log1p(c) * min(1.0, since / COOLDOWN_EPISODES),
    "cooldown_10": lambda p, r, c, since: c * min(1.0, since / 10),
    "cooldown_30": lambda p, r, c, since: c * min(1.0, since / 30),
    "cooldown_50": lambda p, r, c, since: c * min(1.0, since / 50),
    "cooldown_70": lambda p, r, c, since: c * min(1.0, since / 70),
    "cooldown_100": lambda p, r, c, since: c * min(1.0, since / 100),
    "cooldown_150": lambda p, r, c, since: c * min(1.0, since / 150),
    "cooldown_200": lambda p, r, c, since: c * min(1.0, since / 200),
    "cooldown_300": lambda p, r, c, since: c * min(1.0, since / 300),
    "sqrt_count_cooldown_50": lambda p, r, c, since: (c ** 0.5) * min(1.0, since / 50),
    "log_count_cooldown_50": lambda p, r, c, since: math.log1p(c) * min(1.0, since / 50),
    "cooldown_100_fame": lambda p, r, c, since: c * min(1.0, since / 100) * fame_multiplier(p, r),
    "cooldown_150_fame": lambda p, r, c, since: c * min(1.0, since / 150) * fame_multiplier(p, r),
}


def rank_places(train_resa, train_all, train_episode_index, region_by_place, strategy):
    """Score every candidate place for each region, trained on a prefix of episodes."""
    score_fn = STRATEGIES[strategy]
    train_total = len(train_episode_index)
    count = defaultdict(int)
    last_seen = {}
    for r in train_resa:
        place = r["resmal"]
        idx = train_episode_index[(r["sasong"], r["avsnitt"])]
        count[place] += 1
        last_seen[place] = max(last_seen.get(place, -1), idx)
    for r in train_all:
        if r["typ"] == "resa":
            continue
        idx = train_episode_index.get((r["sasong"], r["avsnitt"]))
        if idx is None:
            continue
        last_seen[r["resmal"]] = max(last_seen.get(r["resmal"], -1), idx)

    by_region = defaultdict(list)
    for place, c in count.items():
        region = region_by_place.get(place)
        if region not in REGIONS:
            continue
        episodes_since = train_total - 1 - last_seen[place]
        by_region[region].append((place, score_fn(place, region, c, episodes_since)))

    ranked = {}
    for region in REGIONS:
        places = by_region.get(region, [])
        places.sort(key=lambda p: p[1], reverse=True)
        ranked[region] = [p[0] for p in places]
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=5, help="how many of the most recent seasons to backtest")
    parser.add_argument("--top", type=int, default=15, help="top-N cutoff to count as a 'hit'")
    parser.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], default="all", help="scoring formula to test, or 'all' to compare every candidate")
    args = parser.parse_args()
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    with REGIONS_CSV.open(encoding="utf-8") as f:
        region_by_place = {row["resmal"]: row["region"] for row in csv.DictReader(f)}
    with RESMAL_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    resa_rows = [r for r in all_rows if r["typ"] == "resa"]

    episode_index = {}
    episode_order = []
    for r in resa_rows:
        key = (r["sasong"], r["avsnitt"])
        if key not in episode_index:
            episode_index[key] = len(episode_index)
            episode_order.append(key)

    seasons_sorted = sorted({int(s) for s, _ in episode_order})
    backtest_seasons = set(seasons_sorted[-args.seasons:])
    print(f"Backtesting seasons {sorted(backtest_seasons)} ({args.seasons} most recent)")

    episodes_to_test = [key for key in episode_order if int(key[0]) in backtest_seasons and episode_index[key] > 0]

    for strategy in strategies:
        # rank, in [1, len(region)], or None if the place had zero prior history
        hits = {region: [] for region in REGIONS}

        for key in episodes_to_test:
            idx = episode_index[key]
            train_keys = set(episode_order[:idx])
            train_episode_index = {k: episode_index[k] for k in train_keys}
            train_resa = [r for r in resa_rows if (r["sasong"], r["avsnitt"]) in train_keys]
            train_all = [r for r in all_rows if (r["sasong"], r["avsnitt"]) in train_keys]

            ranked = rank_places(train_resa, train_all, train_episode_index, region_by_place, strategy)
            actual = [r["resmal"] for r in resa_rows if (r["sasong"], r["avsnitt"]) == key]

            for place in actual:
                region = region_by_place.get(place)
                if region not in REGIONS:
                    continue
                place_list = ranked[region]
                rank = place_list.index(place) + 1 if place in place_list else None
                hits[region].append(rank)

        print(f"\n=== strategy: {strategy} ===")
        print(f"{'Region':<16}{'N':>5}{'Top1':>7}{'Top5':>7}{f'Top{args.top}':>7}{'Unseen':>8}{'Median rank':>13}")
        overall_ranks = []
        for region in REGIONS:
            ranks = hits[region]
            n = len(ranks)
            seen = [r for r in ranks if r is not None]
            unseen = n - len(seen)
            top1 = sum(1 for r in seen if r <= 1)
            top5 = sum(1 for r in seen if r <= 5)
            topN = sum(1 for r in seen if r <= args.top)
            median = sorted(seen)[len(seen) // 2] if seen else float("nan")
            overall_ranks.extend(seen)
            print(f"{region:<16}{n:>5}{top1/n*100:>6.0f}%{top5/n*100:>6.0f}%{topN/n*100:>6.0f}%{unseen:>8}{median:>13}")
        print(f"overall median rank: {sorted(overall_ranks)[len(overall_ranks)//2]}")


if __name__ == "__main__":
    main()
