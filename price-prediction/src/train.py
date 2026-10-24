from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "house_prices.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
TARGET = "price"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Add data/house_prices.csv first."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A target value is required for supervised training.
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()

    if len(df) < 10:
        raise ValueError("Please provide at least 10 usable rows for this demo.")

    if (df[TARGET] <= 0).any():
        print("Warning: non-positive target prices found; review the dataset.")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    results = []
    best_model = None
    best_mae = float("inf")
    best_name = None

    print(f"Rows used: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    for name, estimator in models.items():
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            estimator,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

        print(f"\n{name}")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R²:   {r2:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_name = name

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print("\nComparison:")
    print(pd.DataFrame(results).sort_values("MAE").to_string(index=False))
    print(f"\nBest by test MAE: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print("Note: for a robust estimate, use cross-validation and a larger dataset.")


if __name__ == "__main__":
    main()
