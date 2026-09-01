"""Fetch NBA player game logs for recent seasons and cache them locally.

Uses the LeagueGameLog endpoint, which returns every player's game log for
an entire season in a single API call (much friendlier than pulling one
player at a time).
"""

import os
import time
from datetime import date

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_recent_seasons(n=2, today=None):
    """Return the last n NBA season strings, e.g. ['2024-25', '2025-26'].

    The NBA season runs roughly October through June, so before October we
    treat the season that started last October as the most recent one.
    """
    today = today or date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year - n + 1, start_year + 1)]


def fetch_season_game_log(season, season_type="Regular Season"):
    """Pull every player's game log for a season in a single API call."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
    )
    return log.get_data_frames()[0]


def fetch_and_cache_seasons(seasons):
    """Fetch each season's game log, caching to CSV so re-runs don't hit the API."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    frames = []
    for season in seasons:
        cache_path = os.path.join(RAW_DATA_DIR, f"game_log_{season}.csv")
        if os.path.exists(cache_path):
            print(f"Using cached game log for {season}")
            df = pd.read_csv(cache_path)
        else:
            print(f"Fetching game log for {season}...")
            df = fetch_season_game_log(season)
            df.to_csv(cache_path, index=False)
            time.sleep(1)  # be polite to the API
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    seasons = get_recent_seasons(2)
    print(f"Fetching seasons: {seasons}")
    data = fetch_and_cache_seasons(seasons)
    print(f"Fetched {len(data)} player-game rows across {len(seasons)} seasons")
