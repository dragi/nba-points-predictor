"""Run the full pipeline: fetch raw game logs, engineer features, save the training table."""

import os

import pandas as pd

from fetch_data import fetch_and_cache_seasons, get_recent_seasons
from features import build_feature_table

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")


def build_dataset(seasons=None):
    """Fetch the given seasons and return the engineered feature table."""
    seasons = seasons or get_recent_seasons(2)
    raw = fetch_and_cache_seasons(seasons)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    return build_feature_table(raw)


if __name__ == "__main__":
    table = build_dataset()
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(table)} rows to {OUTPUT_PATH}")
