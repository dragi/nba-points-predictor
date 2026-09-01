"""Train and evaluate a points-prediction model on the engineered feature table."""

import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURE_COLUMNS, TARGET_COLUMN

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRAINING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
TEST_FRACTION = 0.2


def load_training_data(path=TRAINING_DATA_PATH):
    """Load the engineered feature table, sorted chronologically."""
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    return df.sort_values("GAME_DATE").reset_index(drop=True)


def time_split(df, test_fraction=TEST_FRACTION):
    """Split into earlier games for training and later games for testing."""
    cutoff = int(len(df) * (1 - test_fraction))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def evaluate(model, X_test, y_test):
    """Return MAE and RMSE of the model on the test set."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return mae, rmse


def main():
    """Train the baseline linear regression and print its test metrics."""
    df = load_training_data()
    train_df, test_df = time_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(X_train, y_train)
    mae, rmse = evaluate(model, X_test, y_test)

    print(f"Train games: {len(train_df)}  Test games: {len(test_df)}")
    print(f"Linear regression  MAE: {mae:.2f}  RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
