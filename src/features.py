"""Turn raw player-game logs into a per-game feature table for points prediction."""

FORM_WINDOWS = (5, 10)
OPPONENT_WINDOW = 10
MIN_GAMES_PLAYED = 20

FEATURE_COLUMNS = [
    "PTS_AVG_LAST_5", "MIN_AVG_LAST_5", "FG_PCT_AVG_LAST_5",
    "PTS_AVG_LAST_10", "MIN_AVG_LAST_10", "FG_PCT_AVG_LAST_10",
    "OPP_PTS_ALLOWED_AVG_LAST_10",
    "REST_DAYS", "IS_BACK_TO_BACK", "IS_HOME",
]
TARGET_COLUMN = "PTS"


def add_matchup_info(df):
    """Parse MATCHUP into opponent + home flag."""
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ")
    df["OPPONENT"] = df["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[1]
    return df


def add_rest_days(df, default_rest=7, max_rest=10):
    """Days since each player's previous game, plus a back-to-back flag."""
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    prev_date = df.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    rest = (df["GAME_DATE"] - prev_date).dt.days
    df["REST_DAYS"] = rest.fillna(default_rest).clip(upper=max_rest)
    df["IS_BACK_TO_BACK"] = df["REST_DAYS"] <= 1
    return df


def add_rolling_form(df, windows=FORM_WINDOWS):
    """Rolling mean of PTS/MIN/FG_PCT over each player's prior N games."""
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    grouped = df.groupby("PLAYER_ID")
    for window in windows:
        for stat in ("PTS", "MIN", "FG_PCT"):
            col = f"{stat}_AVG_LAST_{window}"
            df[col] = grouped[stat].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return df


def add_opponent_defense(df, window=OPPONENT_WINDOW):
    """Attach the opponent's rolling average points allowed coming into each game."""
    team_game = (
        df.groupby(["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE"])["PTS"]
        .sum()
        .reset_index()
        .rename(columns={"PTS": "TEAM_PTS"})
    )
    matchups = team_game.merge(team_game, on="GAME_ID", suffixes=("", "_OPP"))
    matchups = matchups[matchups["TEAM_ID"] != matchups["TEAM_ID_OPP"]]
    matchups = matchups.rename(columns={"TEAM_PTS_OPP": "PTS_ALLOWED"})
    matchups = matchups.sort_values(["TEAM_ID", "GAME_DATE"])

    col = f"OPP_PTS_ALLOWED_AVG_LAST_{window}"
    matchups[col] = matchups.groupby("TEAM_ID")["PTS_ALLOWED"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    opp_defense = matchups[["GAME_ID", "TEAM_ABBREVIATION", col]].rename(
        columns={"TEAM_ABBREVIATION": "OPPONENT"}
    )
    return df.merge(opp_defense, on=["GAME_ID", "OPPONENT"], how="left")


def filter_rotation_players(df, min_games=MIN_GAMES_PLAYED):
    """Keep only players with a meaningful sample of games."""
    games_played = df.groupby("PLAYER_ID")["GAME_ID"].transform("count")
    return df[games_played >= min_games]


def build_feature_table(df):
    """Run the full feature pipeline on a raw player-game log DataFrame."""
    df = add_matchup_info(df)
    df = add_rest_days(df)
    df = add_rolling_form(df)
    df = add_opponent_defense(df)
    df = filter_rotation_players(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df
