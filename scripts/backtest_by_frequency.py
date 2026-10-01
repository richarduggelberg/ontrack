"""Same walk-forward backtest as backtest_predictions.py, but bucketed by
how many times the actual destination had already appeared in training
data (its "prior count") instead of by region.

Answers: "are rare/low-count destinations predicted worse than their
actual recurrence rate justifies?" If count==1 places reappear about as
often as count>=5 places relative to pool size, but rank far worse under
the live model, that's evidence the count*recency weighting (linear in
count) overvalues frequent repeats and undervalues rare ones.

Usage: python scripts/backtest_by_frequency.py [--seasons N] [--strategy NAME]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from backtest_predictions import RESMAL_CSV, REGIONS_CSV, REGIONS, STRATEGIES, rank_places

BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, None)]


def bucket_label(c):
    for lo, hi in BUCKETS:
        if c >= lo and (hi is None or c <= hi):
            return f"{lo}-{hi}" if hi and hi != lo else (f"{lo}+" if hi is None else str(lo))
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=8, help="how many of the most recent seasons to backtest")
    parser.add_argument("--top", type=int, default=15, help="top-N cutoff to count as a 'hit'")
    parser.add_argument("--strategy", default="current", help="scoring formula from backtest_predictions.STRATEGIES")
    args = parser.parse_args()

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
    episodes_to_test = [key for key in episode_order if int(key[0]) in backtest_seasons and episode_index[key] > 0]
    print(f"Backtesting seasons {sorted(backtest_seasons)}, strategy={args.strategy}")

    # bucket -> list of (rank_or_None, pool_size_for_that_region)
    by_bucket = defaultdict(list)
    pool_counts_seen = defaultdict(int)  # bucket -> total candidates ever in a pool (for share-of-pool context)

    for key in episodes_to_test:
        idx = episode_index[key]
        train_keys = set(episode_order[:idx])
        train_resa = [r for r in resa_rows if (r["sasong"], r["avsnitt"]) in train_keys]
        train_all = [r for r in all_rows if (r["sasong"], r["avsnitt"]) in train_keys]
        train_episode_index = {k: episode_index[k] for k in train_keys}

        prior_count = defaultdict(int)
        for r in train_resa:
            prior_count[r["resmal"]] += 1

        ranked = rank_places(train_resa, train_all, train_episode_index, region_by_place, args.strategy)
        for region in REGIONS:
            for place in ranked[region]:
                pool_counts_seen[bucket_label(prior_count[place])] += 1

        actual = [r["resmal"] for r in resa_rows if (r["sasong"], r["avsnitt"]) == key]
        for place in actual:
            region = region_by_place.get(place)
            if region not in REGIONS:
                continue
            c = prior_count.get(place, 0)
            if c == 0:
                by_bucket["0 (unseen)"].append(None)
                continue
            place_list = ranked[region]
            rank = place_list.index(place) + 1 if place in place_list else None
            by_bucket[bucket_label(c)].append(rank)

    order = ["0 (unseen)"] + [bucket_label(lo) for lo, hi in BUCKETS]
    seen_labels = [b for b in order if b in by_bucket]
    print(f"\n{'Prior count':<14}{'N picks':>9}{'Pool share':>12}{'Top1':>7}{'Top5':>7}{f'Top{args.top}':>7}{'Median rank':>13}")
    total_pool = sum(pool_counts_seen.values()) or 1
    for label in seen_labels:
        ranks = by_bucket[label]
        n = len(ranks)
        seen = [r for r in ranks if r is not None]
        pool_share = pool_counts_seen.get(label, 0) / total_pool
        top1 = sum(1 for r in seen if r <= 1)
        top5 = sum(1 for r in seen if r <= 5)
        topN = sum(1 for r in seen if r <= args.top)
        median = sorted(seen)[len(seen) // 2] if seen else float("nan")
        top1_s = f"{top1/n*100:>6.0f}%" if n else "    n/a"
        top5_s = f"{top5/n*100:>6.0f}%" if n else "    n/a"
        topN_s = f"{topN/n*100:>6.0f}%" if n else "    n/a"
        median_s = f"{median:>13}" if seen else f"{'n/a':>13}"
        print(f"{label:<14}{n:>9}{pool_share*100:>11.1f}%{top1_s}{top5_s}{topN_s}{median_s}")


if __name__ == "__main__":
    main()
