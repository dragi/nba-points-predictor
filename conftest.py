"""Shared test fixtures and path setup for the src package."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def _make_raw_log(n_games=30):
    """Build a deterministic two-team synthetic player-game log."""
    rng = np.random.default_rng(0)
    roster = {
        "AAA": [(1, "Alice A"), (2, "Bob A"), (3, "Cy A")],
        "BBB": [(4, "Dan B"), (5, "Eve B"), (6, "Fay B")],
    }
    team_ids = {"AAA": 10, "BBB": 20}
    rows = []
    start = pd.Timestamp("2025-01-01")
    for g in range(n_games):
        date = start + pd.Timedelta(days=2 * g)
        game_id = f"002{g:04d}"
        home, away = ("AAA", "BBB") if g % 2 == 0 else ("BBB", "AAA")
        for team in (home, away):
            opp = away if team == home else home
            matchup = f"{team} vs. {opp}" if team == home else f"{team} @ {opp}"
            for pid, pname in roster[team]:
                rows.append({
                    "GAME_ID": game_id,
                    "GAME_DATE": date,
                    "PLAYER_ID": pid,
                    "PLAYER_NAME": pname,
                    "TEAM_ID": team_ids[team],
                    "TEAM_ABBREVIATION": team,
                    "MATCHUP": matchup,
                    "MIN": int(rng.integers(20, 38)),
                    "PTS": int(rng.integers(5, 30)),
                    "FG_PCT": round(float(rng.uniform(0.35, 0.6)), 3),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_log():
    """A synthetic raw league game log with enough games for the full pipeline."""
    return _make_raw_log()


@pytest.fixture
def feature_table(raw_log):
    """The engineered feature table for the synthetic log, date-sorted."""
    from features import build_feature_table

    return build_feature_table(raw_log).sort_values("GAME_DATE").reset_index(drop=True)
