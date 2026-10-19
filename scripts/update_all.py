"""Run the full data-refresh pipeline in order: fetch newly aired episodes,
classify any new destinations, fetch fame for them, then rebuild
predictions.json. Stops immediately if any step fails.

Usage: python scripts/update_all.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS = [
    "build_resmal_csv.py",
    "classify_destinations.py",
    "fetch_fame.py",
    "fetch_sweden_coords.py",
    "fetch_world_coords.py",
    "build_predictions.py",
]


def main():
    for step in STEPS:
        print(f"\n=== {step} ===")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / step)])
        if result.returncode != 0:
            print(f"\n{step} failed (exit {result.returncode}), stopping.")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
