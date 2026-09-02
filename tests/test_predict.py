"""Tests for the prediction module."""

import numpy as np
import pytest

from features import FEATURE_COLUMNS
from predict import (
    available_opponents,
    available_players,
    build_features,
    predict_points,
)


class _ConstantModel:
    """Stand-in model that always predicts a fixed value."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _bundle(value):
    return {"model": _ConstantModel(value), "feature_columns": FEATURE_COLUMNS}


def test_available_players_are_sorted_and_unique(feature_table):
    players = available_players(feature_table)
    assert players == sorted(set(players))


def test_available_opponents_lists_both_teams(feature_table):
    assert set(available_opponents(feature_table)) == {"AAA", "BBB"}


def test_build_features_has_all_model_columns_and_flags_back_to_back(feature_table):
    feats = build_features(feature_table, "Alice A", "BBB", is_home=True, rest_days=1)
    assert set(feats) == set(FEATURE_COLUMNS)
    assert feats["IS_BACK_TO_BACK"] is True
    assert feats["IS_HOME"] is True


def test_recent_form_matches_last_games_mean(feature_table):
    feats = build_features(feature_table, "Alice A", "BBB", is_home=False, rest_days=3)
    games = feature_table[feature_table["PLAYER_NAME"] == "Alice A"]
    assert feats["PTS_AVG_LAST_5"] == games.tail(5)["PTS"].mean()


def test_predict_points_returns_model_output(feature_table):
    pts = predict_points("Alice A", "BBB", history=feature_table, bundle=_bundle(21.0))
    assert pts == pytest.approx(21.0)


def test_predict_points_floors_negative_predictions_at_zero(feature_table):
    pts = predict_points("Alice A", "BBB", history=feature_table, bundle=_bundle(-4.0))
    assert pts == 0.0


def test_predict_points_rejects_unknown_player(feature_table):
    with pytest.raises(ValueError):
        predict_points("Nobody", "BBB", history=feature_table, bundle=_bundle(10.0))
