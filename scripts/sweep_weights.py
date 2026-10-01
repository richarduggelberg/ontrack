"""Grid-search sweep over build_predictions.py's tunable weights
(COUNT_WEIGHT, RECENCY_WEIGHT, FAME_WEIGHT, COOLDOWN_EPISODES) using the
same walk-forward backtest as backtest_predictions.py.

For each historical test episode, precomputes the (count, episodes_since,
fame, per-region max_fame) context once - this is identical for every
weight combo - then cheaply re-scores and re-ranks candidates for every
combo in the grid. Reports the combos with the best combined top-N hit
rate across all three regions.

Usage: python scripts/sweep_weights.py [--seasons N] [--top N]
"""
import argparse
import csv
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

from fetch_fame import load_cache as load_fame_cache

ROOT = Path(__file__).resolve().parent.parent
RESMAL_CSV = ROOT / "data" / "resmal.csv"
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
REGIONS = ("sweden", "europe", "outside_europe")
FAME_REGIONS = ("europe", "outside_europe")

COUNT_WEIGHTS = (0.5, 0.75, 1.0, 1.25, 1.5)
RECENCY_WEIGHTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
FAME_WEIGHTS = (0.0, 0.5, 1.0, 1.5, 2.0)
COOLDOWNS = (50, 70, 100, 150, 200, 300)
CURRENT = (1.0, 1.0, 1.0, 100)  # today's production defaults, for comparison


def build_episode_snapshots(args):
    """One entry per test episode: precomputed per-region candidate context
    (count, episodes_since, fame, per-region max_fame) that every weight
    combo re-scores, plus which place was actually picked."""
    with REGIONS_CSV.open(encoding="utf-8") as f:
        region_by_place = {row["resmal"]: row["region"] for row in csv.DictReader(f)}
    with RESMAL_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    resa_rows = [r for r in all_rows if r["typ"] == "resa"]
    fame = load_fame_cache()

    episode_index = {}
    episode_order = []
    for r in resa_rows:
        key = (r["sasong"], r["avsnitt"])
        if key not in episode_index:
            episode_index[key] = len(episode_index)
            episode_order.append(key)

    seasons_sorted = sorted({int(s) for s, _ in episode_order})
    end = len(seasons_sorted) - args.offset
    backtest_seasons = set(seasons_sorted[max(0, end - args.seasons):end])
    episodes_to_test = [key for key in episode_order if int(key[0]) in backtest_seasons and episode_index[key] > 0]
    print(f"Backtesting seasons {sorted(backtest_seasons)} (offset={args.offset}, {len(episodes_to_test)} episodes)")

    snapshots = []
    for key in episodes_to_test:
        idx = episode_index[key]
        train_keys = set(episode_order[:idx])
        train_total = len(train_keys)
        train_episode_index = {k: episode_index[k] for k in train_keys}

        count = defaultdict(int)
        last_seen = {}
        for r in resa_rows:
            rkey = (r["sasong"], r["avsnitt"])
            if rkey not in train_keys:
                continue
            place = r["resmal"]
            ridx = train_episode_index[rkey]
            count[place] += 1
            last_seen[place] = max(last_seen.get(place, -1), ridx)
        for r in all_rows:
            if r["typ"] == "resa":
                continue
            rkey = (r["sasong"], r["avsnitt"])
            if rkey not in train_keys:
                continue
            place = r["resmal"]
            last_seen[place] = max(last_seen.get(place, -1), train_episode_index[rkey])

        max_fame_by_region = defaultdict(int)
        candidates_by_region = defaultdict(list)
        for place, c in count.items():
            region = region_by_place.get(place)
            if region not in REGIONS:
                continue
            since = train_total - 1 - last_seen[place]
            fval = fame.get(place, 0)
            candidates_by_region[region].append((place, c, since, fval))
            if region in FAME_REGIONS:
                max_fame_by_region[region] = max(max_fame_by_region[region], fval)

        actual = [r["resmal"] for r in resa_rows if (r["sasong"], r["avsnitt"]) == key]
        actual_by_region = defaultdict(list)
        for place in actual:
            region = region_by_place.get(place)
            if region in REGIONS:
                actual_by_region[region].append(place)

        snapshots.append((candidates_by_region, dict(max_fame_by_region), actual_by_region))
    return snapshots


def evaluate(snapshots, count_weight, recency_weight, fame_weight, cooldown, top):
    hits = {region: [] for region in REGIONS}
    for candidates_by_region, max_fame_by_region, actual_by_region in snapshots:
        for region in REGIONS:
            candidates = candidates_by_region.get(region, [])
            max_fame = max_fame_by_region.get(region, 0)
            scored = []
            for place, c, since, fval in candidates:
                recency_factor = min(1.0, since / cooldown) ** recency_weight
                fame_mult = (1.0 + math.log1p(fval) / math.log1p(max_fame)) if (region in FAME_REGIONS and max_fame > 0) else 1.0
                weight = (c ** count_weight) * recency_factor * (fame_mult ** fame_weight)
                scored.append((place, weight))
            scored.sort(key=lambda p: p[1], reverse=True)
            place_list = [p[0] for p in scored]
            for place in actual_by_region.get(region, []):
                rank = place_list.index(place) + 1 if place in place_list else None
                hits[region].append(rank)

    overall_ranks = []
    region_stats = {}
    for region in REGIONS:
        ranks = hits[region]
        n = len(ranks)
        seen = [r for r in ranks if r is not None]
        topN = sum(1 for r in seen if r <= top)
        overall_ranks.extend(seen)
        region_stats[region] = (n, topN, sorted(seen)[len(seen) // 2] if seen else None)
    total_n = len(overall_ranks)
    total_topN = sum(1 for r in overall_ranks if r <= top)
    overall_median = sorted(overall_ranks)[len(overall_ranks) // 2] if overall_ranks else None
    return total_topN / total_n, overall_median, region_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=8, help="how many seasons to backtest")
    parser.add_argument("--offset", type=int, default=0, help="skip this many of the most recent seasons first (for a disjoint validation window)")
    parser.add_argument("--top", type=int, default=15, help="top-N cutoff to count as a 'hit'")
    parser.add_argument("--show", type=int, default=15, help="how many top combos to print")
    parser.add_argument("--check", type=str, default=None, help="evaluate one specific 'count_w,recency_w,fame_w,cooldown' combo instead of the full grid")
    args = parser.parse_args()

    snapshots = build_episode_snapshots(args)

    if args.check:
        cw, rw, fw, cd = (float(x) for x in args.check.split(","))
        hit_rate, median, region_stats = evaluate(snapshots, cw, rw, fw, cd, args.top)
        print(f"\ncombo count={cw} recency={rw} fame={fw} cooldown={cd}: "
              f"top{args.top} hit rate {hit_rate*100:.1f}%, overall median rank {median}")
        for region, (n, topN, med) in region_stats.items():
            print(f"  {region:<16} N={n:<5} top{args.top}={topN/n*100:.0f}%  median={med}")
        return

    combos = list(product(COUNT_WEIGHTS, RECENCY_WEIGHTS, FAME_WEIGHTS, COOLDOWNS))
    print(f"Evaluating {len(combos)} weight combos...")

    results = []
    for cw, rw, fw, cd in combos:
        hit_rate, median, region_stats = evaluate(snapshots, cw, rw, fw, cd, args.top)
        results.append((hit_rate, median, cw, rw, fw, cd, region_stats))

    results.sort(key=lambda r: r[0], reverse=True)

    current_hit_rate, current_median, current_region_stats = evaluate(snapshots, *CURRENT, args.top)
    print(f"\ncurrent production (count=1.0 recency=1.0 fame=1.0 cooldown=100): "
          f"top{args.top} hit rate {current_hit_rate*100:.1f}%, overall median rank {current_median}")
    for region, (n, topN, median) in current_region_stats.items():
        print(f"  {region:<16} N={n:<5} top{args.top}={topN/n*100:.0f}%  median={median}")

    print(f"\nTop {args.show} combos by overall top{args.top} hit rate:")
    print(f"{'count_w':>8}{'recency_w':>10}{'fame_w':>8}{'cooldown':>10}{f'top{args.top}':>8}{'median':>8}")
    for hit_rate, median, cw, rw, fw, cd, region_stats in results[:args.show]:
        print(f"{cw:>8}{rw:>10}{fw:>8}{cd:>10}{hit_rate*100:>7.1f}%{median:>8}")


if __name__ == "__main__":
    main()
