"""Train and evaluate points-prediction models, then persist the best one."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURE_COLUMNS, TARGET_COLUMN

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRAINING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
TEST_FRACTION = 0.2
RANDOM_STATE = 42


def load_training_data(path=TRAINING_DATA_PATH):
    """Load the engineered feature table, sorted chronologically."""
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    return df.sort_values("GAME_DATE").reset_index(drop=True)


def time_split(df, test_fraction=TEST_FRACTION):
    """Split into earlier games for training and later games for testing."""
    cutoff = int(len(df) * (1 - test_fraction))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def build_models():
    """Return the candidate models, keyed by display name."""
    return {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def evaluate(model, X_test, y_test):
    """Return MAE and RMSE of the model on the test set."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return mae, rmse


def save_model(model, path=MODEL_PATH):
    """Persist a fitted model to disk with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, path)


def main():
    """Compare models on a time split, then refit the best on all games and save it."""
    df = load_training_data()
    train_df, test_df = time_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print(f"Train games: {len(train_df)}  Test games: {len(test_df)}")
    scores = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        mae, rmse = evaluate(model, X_test, y_test)
        scores[name] = mae
        print(f"{name:<20} MAE: {mae:.2f}  RMSE: {rmse:.2f}")

    best_name = min(scores, key=scores.get)
    best_model = build_models()[best_name]
    best_model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    save_model(best_model)
    print(f"Saved {best_name} (retrained on all {len(df)} games) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
