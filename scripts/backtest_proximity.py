"""Walk-forward backtest of the proximity penalty (build_predictions.py).

Compares the cooldown+count baseline (no proximity) against adding the
proximity multiplier with various (lookback, decay, weight) combos, for
one region at a time, across one or more disjoint windows of seasons -
to check whether it actually improves top-N hit rate / median rank
rather than just looking good on the single most recent window.

Usage: python scripts/backtest_proximity.py [--region sweden|europe|outside_europe] [--seasons N] [--offset N] [--top N]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESMAL_CSV = ROOT / "data" / "resmal.csv"
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
SWEDEN_COORDS_CACHE = ROOT / "data" / "sweden_coords_cache.json"
WORLD_COORDS_CACHE = ROOT / "data" / "world_coords_cache.json"

COOLDOWN_EPISODES = 100
LOCAL_SPACING_K = 3
PROXIMITY_RADIUS_SCALE = 4.5
PROXIMITY_RADIUS_MIN_KM = {"sweden": 50, "europe": 150, "outside_europe": 300}
PROXIMITY_RADIUS_MAX_KM = {"sweden": 400, "europe": 1200, "outside_europe": 3000}
PROXIMITY_MIN_MULT = 0.15

# (label, lookback, decay, weight) - lookback=None means no proximity penalty
# at all; weight is the exponent applied to the multiplier (PROXIMITY_WEIGHT).
CONFIGS = [
    ("baseline (no proximity)", None, None, None),
    ("lookback=1 weight=0.5", 1, 1.0, 0.5),
    ("lookback=1 weight=1.0", 1, 1.0, 1.0),
    ("lookback=1 weight=1.5", 1, 1.0, 1.5),
    ("lookback=1 weight=2.0", 1, 1.0, 2.0),
    ("lookback=1 weight=3.0", 1, 1.0, 3.0),
    ("lookback=2 decay=0.5 weight=1.0", 2, 0.5, 1.0),
    ("lookback=2 decay=0.5 weight=1.5", 2, 0.5, 1.5),
    ("lookback=2 decay=0.5 weight=2.0", 2, 0.5, 2.0),
]


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def build_radius_by_place(coords, region):
    radius = {}
    for place, c in coords.items():
        dists = sorted(haversine_km(c, c2) for p2, c2 in coords.items() if p2 != place)
        k = dists[:LOCAL_SPACING_K]
        spacing = sum(k) / len(k) if k else 0.0
        radius[place] = min(
            PROXIMITY_RADIUS_MAX_KM[region],
            max(PROXIMITY_RADIUS_MIN_KM[region], PROXIMITY_RADIUS_SCALE * spacing),
        )
    return radius


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["sweden", "europe", "outside_europe"], default="sweden")
    parser.add_argument("--seasons", type=int, default=8, help="how many seasons to backtest")
    parser.add_argument("--offset", type=int, default=0, help="skip this many of the most recent seasons first (disjoint validation window)")
    parser.add_argument("--top", type=int, default=15, help="top-N cutoff to count as a 'hit'")
    args = parser.parse_args()
    region = args.region

    with REGIONS_CSV.open(encoding="utf-8") as f:
        region_by_place = {row["resmal"]: row["region"] for row in csv.DictReader(f)}
    with RESMAL_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    resa_rows = [r for r in all_rows if r["typ"] == "resa"]

    coords_cache = SWEDEN_COORDS_CACHE if region == "sweden" else WORLD_COORDS_CACHE
    all_coords = json.loads(coords_cache.read_text(encoding="utf-8")) if coords_cache.exists() else {}
    coords = {p: c for p, c in all_coords.items() if region_by_place.get(p) == region and c}
    radius_by_place = build_radius_by_place(coords, region)

    episode_index = {}
    episode_order = []
    for r in resa_rows:
        key = (r["sasong"], r["avsnitt"])
        if key not in episode_index:
            episode_index[key] = len(episode_index)
            episode_order.append(key)

    seasons_sorted = sorted({int(s) for s, _ in episode_order})
    usable_seasons = seasons_sorted[: len(seasons_sorted) - args.offset] if args.offset else seasons_sorted
    backtest_seasons = set(usable_seasons[-args.seasons:])
    print(f"Backtesting region={region} seasons {sorted(backtest_seasons)} (offset={args.offset})")

    episodes_to_test = [key for key in episode_order if int(key[0]) in backtest_seasons and episode_index[key] > 0]

    for label, lookback, decay, weight_exp in CONFIGS:
        ranks = []
        for key in episodes_to_test:
            idx = episode_index[key]
            train_keys = set(episode_order[:idx])
            train_resa = [r for r in resa_rows if (r["sasong"], r["avsnitt"]) in train_keys]
            train_all = [r for r in all_rows if (r["sasong"], r["avsnitt"]) in train_keys]
            train_episode_index = {k: episode_index[k] for k in train_keys}
            train_total = len(train_keys)

            count = defaultdict(int)
            last_seen = {}
            for r in train_resa:
                place = r["resmal"]
                tidx = train_episode_index[(r["sasong"], r["avsnitt"])]
                count[place] += 1
                last_seen[place] = max(last_seen.get(place, -1), tidx)
            for r in train_all:
                if r["typ"] == "resa":
                    continue
                tidx = train_episode_index.get((r["sasong"], r["avsnitt"]))
                if tidx is None:
                    continue
                last_seen[r["resmal"]] = max(last_seen.get(r["resmal"], -1), tidx)

            # Chronological sequence of actual picks within the training window.
            # An episode can contribute more than one pick to the same region,
            # so every matching row is kept, not just the first.
            region_sequence = []
            for r in train_resa:
                place = r["resmal"]
                if region_by_place.get(place) == region:
                    tidx = train_episode_index[(r["sasong"], r["avsnitt"])]
                    region_sequence.append((tidx, place))
            recent = [p for _, p in region_sequence[-lookback:]] if lookback else []

            def proximity_mult(place):
                if lookback is None:
                    return 1.0
                place_coord = coords.get(place)
                if not place_coord:
                    return 1.0
                radius = radius_by_place.get(place)
                if not radius:
                    return 1.0
                best = 1.0
                for lag, rp in enumerate(reversed(recent), start=1):
                    if rp == place:
                        continue
                    rc = coords.get(rp)
                    if not rc:
                        continue
                    dist = haversine_km(place_coord, rc)
                    raw = 1.0 if dist >= radius else PROXIMITY_MIN_MULT + (1 - PROXIMITY_MIN_MULT) * (dist / radius)
                    eff = 1.0 - (1.0 - raw) * (decay ** (lag - 1))
                    best = min(best, eff)
                return best

            scored = []
            for place, c in count.items():
                if region_by_place.get(place) != region:
                    continue
                episodes_since = train_total - 1 - last_seen[place]
                recency_factor = min(1.0, episodes_since / COOLDOWN_EPISODES)
                mult = proximity_mult(place) ** weight_exp if weight_exp is not None else 1.0
                weight = c * recency_factor * mult
                scored.append((place, weight))
            scored.sort(key=lambda p: p[1], reverse=True)
            ranked_places = [p for p, _ in scored]

            actual = [
                r["resmal"] for r in resa_rows
                if (r["sasong"], r["avsnitt"]) == key and region_by_place.get(r["resmal"]) == region
            ]
            for place in actual:
                rank = ranked_places.index(place) + 1 if place in ranked_places else None
                ranks.append(rank)

        n = len(ranks)
        seen = [r for r in ranks if r is not None]
        unseen = n - len(seen)
        top1 = sum(1 for r in seen if r <= 1)
        top5 = sum(1 for r in seen if r <= 5)
        topN = sum(1 for r in seen if r <= args.top)
        median = sorted(seen)[len(seen) // 2] if seen else float("nan")
        print(
            f"{label:<28} n={n:>4} top1={top1/n*100:>5.1f}% top5={top5/n*100:>5.1f}% "
            f"top{args.top}={topN/n*100:>5.1f}% unseen={unseen:>3} median={median}"
        )


if __name__ == "__main__":
    main()
