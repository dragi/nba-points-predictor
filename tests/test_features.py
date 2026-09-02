"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd

from features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_matchup_info,
    add_opponent_defense,
    add_rest_days,
    add_rolling_form,
    build_feature_table,
    filter_rotation_players,
)


def test_add_matchup_info_parses_home_and_away():
    df = pd.DataFrame({"MATCHUP": ["LAL vs. GSW", "LAL @ GSW"]})
    out = add_matchup_info(df)
    assert out["IS_HOME"].tolist() == [True, False]
    assert out["OPPONENT"].tolist() == ["GSW", "GSW"]


def test_add_rest_days_counts_gap_and_flags_back_to_back():
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1, 1],
        "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-05"]),
    })
    out = add_rest_days(df, default_rest=7)
    assert out["REST_DAYS"].tolist() == [7, 1, 3]
    assert out["IS_BACK_TO_BACK"].tolist() == [False, True, False]


def test_add_rest_days_clips_long_layoffs():
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1],
        "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-02-01"]),
    })
    out = add_rest_days(df, max_rest=10)
    assert out["REST_DAYS"].tolist() == [7, 10]


def test_rolling_form_uses_only_prior_games():
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1, 1],
        "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]),
        "PTS": [10, 20, 30],
        "MIN": [30, 30, 30],
        "FG_PCT": [0.5, 0.5, 0.5],
    })
    out = add_rolling_form(df, windows=(2,))
    assert np.isnan(out["PTS_AVG_LAST_2"].iloc[0])
    assert out["PTS_AVG_LAST_2"].tolist()[1:] == [10.0, 15.0]


def test_opponent_defense_averages_points_allowed_from_prior_games(raw_log):
    df = add_matchup_info(raw_log)
    out = add_opponent_defense(df, window=5)
    col = "OPP_PTS_ALLOWED_AVG_LAST_5"
    assert col in out.columns
    assert out[col].notna().any()


def test_filter_rotation_players_drops_small_samples():
    df = pd.DataFrame({
        "PLAYER_ID": [1, 1, 1, 2],
        "GAME_ID": ["a", "b", "c", "a"],
    })
    out = filter_rotation_players(df, min_games=3)
    assert set(out["PLAYER_ID"]) == {1}


def test_build_feature_table_has_no_missing_features(feature_table):
    assert len(feature_table) > 0
    assert not feature_table[FEATURE_COLUMNS].isna().any().any()
    assert TARGET_COLUMN in feature_table.columns


def test_build_feature_table_form_matches_manual_mean(raw_log):
    table = build_feature_table(raw_log)
    player = raw_log[raw_log["PLAYER_ID"] == 1].sort_values("GAME_DATE")
    row = table[table["PLAYER_ID"] == 1].sort_values("GAME_DATE").iloc[5]
    prior = player[player["GAME_DATE"] < row["GAME_DATE"]].tail(5)
    assert row["PTS_AVG_LAST_5"] == prior["PTS"].mean()
