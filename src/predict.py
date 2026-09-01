"""Load the trained model and predict a player's points for an upcoming game."""

import os

import joblib
import pandas as pd

from features import FORM_WINDOWS
from train import MODEL_PATH, TRAINING_DATA_PATH

_STATS = ("PTS", "MIN", "FG_PCT")


def load_bundle(path=MODEL_PATH):
    """Load the persisted model and its expected feature column order."""
    return joblib.load(path)


def load_history(path=TRAINING_DATA_PATH):
    """Load the engineered game table used to derive recent form."""
    return pd.read_csv(path, parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")


def available_players(history):
    """Return the sorted list of player names present in the history."""
    return sorted(history["PLAYER_NAME"].unique())


def available_opponents(history):
    """Return the sorted list of team abbreviations that appear as opponents."""
    return sorted(history["OPPONENT"].dropna().unique())


def _recent_form(history, player_name):
    """Rolling means of each player's most recent games for every form window."""
    games = history[history["PLAYER_NAME"] == player_name]
    if games.empty:
        raise ValueError(f"No game history for player {player_name!r}")
    feats = {}
    for window in FORM_WINDOWS:
        recent = games.tail(window)
        for stat in _STATS:
            feats[f"{stat}_AVG_LAST_{window}"] = recent[stat].mean()
    return feats


def _opponent_defense(history, opponent):
    """Most recent rolling points-allowed figure recorded for an opponent."""
    rows = history[history["OPPONENT"] == opponent]
    if rows.empty:
        raise ValueError(f"No games found against opponent {opponent!r}")
    return rows["OPP_PTS_ALLOWED_AVG_LAST_10"].iloc[-1]


def build_features(history, player_name, opponent, is_home, rest_days):
    """Assemble the model input row for one upcoming game."""
    feats = _recent_form(history, player_name)
    feats["OPP_PTS_ALLOWED_AVG_LAST_10"] = _opponent_defense(history, opponent)
    feats["REST_DAYS"] = rest_days
    feats["IS_BACK_TO_BACK"] = rest_days <= 1
    feats["IS_HOME"] = bool(is_home)
    return feats


def predict_points(player_name, opponent, is_home=True, rest_days=2,
                   history=None, bundle=None):
    """Predict points for a player against an opponent given rest and venue."""
    history = load_history() if history is None else history
    bundle = load_bundle() if bundle is None else bundle
    feats = build_features(history, player_name, opponent, is_home, rest_days)
    row = pd.DataFrame([feats])[bundle["feature_columns"]]
    return max(0.0, float(bundle["model"].predict(row)[0]))


def main():
    """Print a sample prediction from the command line."""
    points = predict_points("LeBron James", "GSW", is_home=True, rest_days=2)
    print(f"Predicted points: {points:.1f}")


if __name__ == "__main__":
    main()
