"""Build ranked next-destination predictions, split into Sweden / Europe /
Outside Europe, from data/resmal.csv + data/resmal_regions.csv.

Working assumption (verified against seasons 33-37, 98.3% match rate):
each episode's 3 main destinations are one Sweden + one Europe + one
outside-Europe place, which is why predictions are split into exactly
these 3 regional slots rather than one combined ranked list.

Scoring (simple, explainable):
  weight = historical_count**COUNT_WEIGHT * recency_factor**RECENCY_WEIGHT
            * fame_multiplier**FAME_WEIGHT
  recency_factor = min(1.0, episodes_since_last_seen / COOLDOWN_EPISODES)
So a place that appeared recently is strongly suppressed; it recovers back
to full weight once ~COOLDOWN_EPISODES have passed without it reappearing.
Within each region, weights are normalised to percentages.

COUNT_WEIGHT / RECENCY_WEIGHT / FAME_WEIGHT are exponents (default 1.0,
i.e. today's formula unchanged) that control how much each factor is
allowed to swing the ranking - e.g. COUNT_WEIGHT < 1 compresses count's
naturally unbounded influence, closer to the sqrt/log variants tested in
backtest_predictions.py.

COOLDOWN_EPISODES=100 was chosen by walk-forward backtesting (see
scripts/backtest_predictions.py) over seasons 33-37: it roughly halves
the median rank of the actual destination vs. the original guess of 20
(e.g. outside-Europe top-15 hit rate 21% -> 33%). Even so, the model's
accuracy is modest - run the backtest script for current numbers - since
a meaningful share of real picks are long-tail or first-ever appearances
no history-based model can anticipate.

Fame boost: for Europe/outside-Europe (not Sweden - per manual review
that slot can be any town at all), weight is additionally multiplied by
1 + log1p(sv.wikipedia pageviews) / log1p(max pageviews among that
region's own candidates) - normalised per region, since Europe and
outside-Europe don't need to sit on a comparable fame scale - so a place
known to Swedes (see fetch_fame.py) ranks a bit higher than an
equally-frequent but obscure one in the same region. This only re-ranks
places with existing show history though; it still can't predict true
first-time destinations.

Wildcards: data/wildcards.csv (built by build_wildcards.py) lists world
capitals / population >= 1M cities that have NEVER appeared on the show,
ranked purely by fame. These are surfaced separately per region (not
mixed into the history-based ranking) as the model's best guess at what
a first-time destination might look like.

Proximity penalty: checked against real history (see
scripts/fetch_sweden_coords.py, scripts/fetch_world_coords.py +
ad-hoc distance analysis) - consecutive episodes' picks within the same
region are spaced further apart than random chance, confirming the show
avoids picking two nearby destinations back-to-back. So a candidate
close to one of the last PROXIMITY_LOOKBACK same-region destinations
gets its weight multiplied down towards PROXIMITY_MIN_MULT (at distance
0) scaling linearly back up to 1.0 at that candidate's "too close"
radius or further. Older picks within that lookback window count for
less: each one further back is faded towards no-penalty by
PROXIMITY_DECAY per step (see scripts/backtest_proximity.py for the
walk-forward validation behind these constants).

That radius isn't a flat km value - candidates within a region can be
packed far more densely in some areas than others (e.g. southern Sweden
has far more towns close together than the sparse north: median
distance to a town's 3 nearest neighbours is ~20-24km south of
Mälardalen vs. ~35-45km in Norrland), so a flat threshold would
over-penalise the dense areas and barely matter in the sparse ones.
Instead each place's own radius is PROXIMITY_RADIUS_SCALE times its
local nearest-neighbour spacing (LOCAL_SPACING_K nearest other
candidates in the *same region*), clamped to
[PROXIMITY_RADIUS_MIN_KM, PROXIMITY_RADIUS_MAX_KM] - those clamps are
per-region since Europe/outside-Europe candidates are spread across a
much larger geographic scale than Sweden's towns. Candidates/recent
places missing coordinates are treated as no penalty.

The candidate pool and historical_count are based only on the main "resa"
destinations. But "episodes_since_last_seen" also counts appearances in
the tintin_haddock / narmast_vinner bonus segments: a place flagged there
is unlikely to be picked as the main resmal again right away, so those
mentions reset its cooldown too.

Usage: python scripts/build_predictions.py
"""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESMAL_CSV = ROOT / "data" / "resmal.csv"
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
FAME_CACHE = ROOT / "data" / "fame_cache.json"
WILDCARDS_CSV = ROOT / "data" / "wildcards.csv"
SWEDEN_COORDS_CACHE = ROOT / "data" / "sweden_coords_cache.json"
WORLD_COORDS_CACHE = ROOT / "data" / "world_coords_cache.json"
OUT_PATH = ROOT / "data" / "predictions.json"

COOLDOWN_EPISODES = 100
COUNT_WEIGHT = 1.0
RECENCY_WEIGHT = 1.0
FAME_WEIGHT = 1.0
TOP_N = 25
WILDCARD_N = 10
# Sweden can be literally any town (per manual review), so fame isn't a
# useful discriminator there - only boost Europe/outside-Europe by it.
FAME_REGIONS = ("europe", "outside_europe")

# Suppress candidates that are geographically close to the most recent
# same-region pick(s), scaling linearly from PROXIMITY_MIN_MULT (at
# distance 0) up to full weight (1.0) at that place's "too close" radius
# or beyond. The radius itself scales with local candidate density (see
# local_spacing below) rather than being a flat km value - and the
# min/max clamp is per-region since Sweden's towns and Europe/outside-
# Europe's candidates sit on very different geographic scales.
PROXIMITY_REGIONS = ("sweden", "europe", "outside_europe")
LOCAL_SPACING_K = 3
PROXIMITY_RADIUS_SCALE = 4.5
PROXIMITY_RADIUS_MIN_KM = {"sweden": 50, "europe": 150, "outside_europe": 300}
PROXIMITY_RADIUS_MAX_KM = {"sweden": 400, "europe": 1200, "outside_europe": 3000}
PROXIMITY_MIN_MULT = 0.15
PROXIMITY_WEIGHT = 1.0
PROXIMITY_LOOKBACK = 1
PROXIMITY_DECAY = 0.5

REGION_LABELS = {
    "sweden": "Sverige",
    "europe": "Europa",
    "outside_europe": "Resten av världen",
}


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def main():
    with REGIONS_CSV.open(encoding="utf-8") as f:
        region_by_place = {row["resmal"]: row["region"] for row in csv.DictReader(f)}

    fame = json.loads(FAME_CACHE.read_text(encoding="utf-8")) if FAME_CACHE.exists() else {}

    with RESMAL_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    resa_rows = [r for r in all_rows if r["typ"] == "resa"]

    # Assign a chronological episode index based on first appearance order.
    episode_index = {}
    for r in resa_rows:
        key = (r["sasong"], r["avsnitt"])
        if key not in episode_index:
            episode_index[key] = len(episode_index)
    total_episodes = len(episode_index)

    count = defaultdict(int)
    last_seen = {}
    for r in resa_rows:
        place = r["resmal"]
        idx = episode_index[(r["sasong"], r["avsnitt"])]
        count[place] += 1
        last_seen[place] = max(last_seen.get(place, -1), idx)

    # Bonus-segment mentions (tintin_haddock, narmast_vinner) also reset the
    # cooldown, even for places that were never themselves a main resmal.
    for r in all_rows:
        if r["typ"] == "resa":
            continue
        idx = episode_index.get((r["sasong"], r["avsnitt"]))
        if idx is None:
            continue
        place = r["resmal"]
        last_seen[place] = max(last_seen.get(place, -1), idx)

    # Fame is normalised within each region's own candidate set, not globally -
    # Sweden/Europe/outside-Europe don't need to be on a comparable fame scale.
    max_fame_by_region = defaultdict(int)
    for place, c in count.items():
        region = region_by_place.get(place, "unknown")
        if region in FAME_REGIONS:
            max_fame_by_region[region] = max(max_fame_by_region[region], fame.get(place, 0))

    def fame_multiplier(place, region):
        max_fame = max_fame_by_region.get(region, 0)
        if region not in FAME_REGIONS or max_fame <= 0:
            return 1.0
        return 1.0 + math.log1p(fame.get(place, 0)) / math.log1p(max_fame)

    sweden_coords = (
        json.loads(SWEDEN_COORDS_CACHE.read_text(encoding="utf-8"))
        if SWEDEN_COORDS_CACHE.exists()
        else {}
    )
    world_coords = (
        json.loads(WORLD_COORDS_CACHE.read_text(encoding="utf-8"))
        if WORLD_COORDS_CACHE.exists()
        else {}
    )
    coords_by_region = {
        "sweden": sweden_coords,
        "europe": world_coords,
        "outside_europe": world_coords,
    }

    # Chronological sequence of actual picks per region, to penalise
    # candidates close to the most recent one(s).
    sequence_by_region = defaultdict(list)
    for r in resa_rows:
        place = r["resmal"]
        region = region_by_place.get(place)
        if region in PROXIMITY_REGIONS:
            idx = episode_index[(r["sasong"], r["avsnitt"])]
            seq = sequence_by_region[region]
            if not seq or seq[-1][0] != idx:
                seq.append((idx, place))
    recent_places_by_region = {
        region: [p for _, p in seq[-PROXIMITY_LOOKBACK:]]
        for region, seq in sequence_by_region.items()
    }

    # A place's own "too close" radius scales with how densely packed its
    # neighbours are (within the same region's own candidate pool), so the
    # penalty bites at a similar relative distance whether a place sits
    # among many close neighbours or few, far-apart ones.
    def local_spacing(place, coords):
        other_dists = sorted(
            haversine_km(coords[place], c) for p, c in coords.items() if p != place
        )
        k = other_dists[:LOCAL_SPACING_K]
        return sum(k) / len(k) if k else 0.0

    radius_by_region_place = {}
    for region in PROXIMITY_REGIONS:
        region_coords = {
            p: c for p, c in coords_by_region[region].items() if region_by_place.get(p) == region and c
        }
        radius_by_region_place[region] = {
            place: min(
                PROXIMITY_RADIUS_MAX_KM[region],
                max(PROXIMITY_RADIUS_MIN_KM[region], PROXIMITY_RADIUS_SCALE * local_spacing(place, region_coords)),
            )
            for place in region_coords
        }

    def proximity_multiplier(place, region):
        if region not in PROXIMITY_REGIONS:
            return 1.0
        coords = coords_by_region[region]
        place_coord = coords.get(place)
        if not place_coord:
            return 1.0
        radius = radius_by_region_place[region].get(place)
        if not radius:
            return 1.0
        best = 1.0
        recent_places = recent_places_by_region.get(region, [])
        # recent_places is chronological (oldest first); lag 1 = most recent.
        for lag, recent in enumerate(reversed(recent_places), start=1):
            if recent == place:
                continue
            recent_coord = coords.get(recent)
            if not recent_coord:
                continue
            dist = haversine_km(place_coord, recent_coord)
            if dist >= radius:
                raw_mult = 1.0
            else:
                raw_mult = PROXIMITY_MIN_MULT + (1 - PROXIMITY_MIN_MULT) * (dist / radius)
            decay = PROXIMITY_DECAY ** (lag - 1)
            effective_mult = 1.0 - (1.0 - raw_mult) * decay
            best = min(best, effective_mult)
        return best

    by_region = defaultdict(list)
    for place, c in count.items():
        region = region_by_place.get(place, "unknown")
        if region not in REGION_LABELS:
            continue
        episodes_since = total_episodes - 1 - last_seen[place]
        recency_factor = min(1.0, episodes_since / COOLDOWN_EPISODES)
        weight = (
            c ** COUNT_WEIGHT
            * recency_factor ** RECENCY_WEIGHT
            * fame_multiplier(place, region) ** FAME_WEIGHT
            * proximity_multiplier(place, region) ** PROXIMITY_WEIGHT
        )
        by_region[region].append({
            "place": place,
            "count": c,
            "episodes_since_last_seen": episodes_since,
            "weight": weight,
        })

    output_regions = {}
    for region, label in REGION_LABELS.items():
        places = by_region.get(region, [])
        total_weight = sum(p["weight"] for p in places) or 1.0
        places.sort(key=lambda p: p["weight"], reverse=True)
        ranked = []
        for p in places[:TOP_N]:
            ranked.append({
                "place": p["place"],
                "count": p["count"],
                "episodes_since_last_seen": p["episodes_since_last_seen"],
                "percentage": round(100 * p["weight"] / total_weight, 1),
            })
        output_regions[region] = {"label": label, "predictions": ranked}

    wildcards_by_region = defaultdict(list)
    if WILDCARDS_CSV.exists():
        with WILDCARDS_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wildcards_by_region[row["region"]].append(row)
    for region in FAME_REGIONS:
        candidates = sorted(wildcards_by_region.get(region, []), key=lambda r: int(r["fame"]), reverse=True)
        output_regions[region]["wildcards"] = [
            {
                "place": r["resmal"],
                "country_code": r["country_code"],
                "population": int(r["population"]),
                "fame": int(r["fame"]),
            }
            for r in candidates[:WILDCARD_N]
        ]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episode_count": total_episodes,
        "cooldown_episodes": COOLDOWN_EPISODES,
        "regions": output_regions,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions for {total_episodes} episodes to {OUT_PATH}")


if __name__ == "__main__":
    main()
