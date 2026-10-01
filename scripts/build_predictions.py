"""Build ranked next-destination predictions, split into Sweden / Europe /
Outside Europe, from data/resmal.csv + data/resmal_regions.csv.

Working assumption (verified against seasons 33-37, 98.3% match rate):
each episode's 3 main destinations are one Sweden + one Europe + one
outside-Europe place, which is why predictions are split into exactly
these 3 regional slots rather than one combined ranked list.

Scoring (simple, explainable):
  weight = historical_count * recency_factor
  recency_factor = min(1.0, episodes_since_last_seen / COOLDOWN_EPISODES)
So a place that appeared recently is strongly suppressed; it recovers back
to full weight once ~COOLDOWN_EPISODES have passed without it reappearing.
Within each region, weights are normalised to percentages.

COOLDOWN_EPISODES=100 was chosen by walk-forward backtesting (see
scripts/backtest_predictions.py) over seasons 33-37: it roughly halves
the median rank of the actual destination vs. the original guess of 20
(e.g. outside-Europe top-15 hit rate 21% -> 33%). Even so, the model's
accuracy is modest - run the backtest script for current numbers - since
a meaningful share of real picks are long-tail or first-ever appearances
no history-based model can anticipate.

The candidate pool and historical_count are based only on the main "resa"
destinations. But "episodes_since_last_seen" also counts appearances in
the tintin_haddock / narmast_vinner bonus segments: a place flagged there
is unlikely to be picked as the main resmal again right away, so those
mentions reset its cooldown too.

Usage: python scripts/build_predictions.py
"""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESMAL_CSV = ROOT / "data" / "resmal.csv"
REGIONS_CSV = ROOT / "data" / "resmal_regions.csv"
OUT_PATH = ROOT / "data" / "predictions.json"

COOLDOWN_EPISODES = 100
TOP_N = 15

REGION_LABELS = {
    "sweden": "Sverige",
    "europe": "Europa",
    "outside_europe": "Resten av världen",
}


def main():
    with REGIONS_CSV.open(encoding="utf-8") as f:
        region_by_place = {row["resmal"]: row["region"] for row in csv.DictReader(f)}

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

    by_region = defaultdict(list)
    for place, c in count.items():
        region = region_by_place.get(place, "unknown")
        if region not in REGION_LABELS:
            continue
        episodes_since = total_episodes - 1 - last_seen[place]
        recency_factor = min(1.0, episodes_since / COOLDOWN_EPISODES)
        weight = c * recency_factor
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
