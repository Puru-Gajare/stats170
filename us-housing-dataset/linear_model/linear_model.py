import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import json

FEATURES = [
    'bed', 'bath', 'house_size', 'acre_lot', 'city_encoded',
    'median_income', 'population', 'poverty_rate', 'bachelors_rate',
    # School & library proximity features (from schools_with_zipcode / library_with_zipcode)
    'school_count', 'avg_school_dist_km', 'min_school_dist_km',
    'library_count', 'avg_library_dist_km', 'min_library_dist_km',
]

# Paths are resolved relative to this file so the script works from any CWD.
_HERE = Path(__file__).parent          # us-housing-dataset/linear_model/
PKL_PATH     = _HERE.parent / 'ca_2022_sold.pkl'     # us-housing-dataset/ca_2022_sold.pkl
RESULTS_PATH = _HERE / 'model_results.json'          # us-housing-dataset/linear_model/model_results.json


def load_and_clean_data(pkl_path=PKL_PATH) -> pd.DataFrame:
    """Load the CA 2022 housing dataset, apply filters, and encode categorical columns."""
    print(f"Loading {pkl_path}...")
    df = pd.read_pickle(pkl_path)

    # Drop rows with missing values in core features
    core_cols = ['price', 'bed', 'bath', 'house_size', 'acre_lot', 'city',
                 'median_income', 'population', 'poverty_rate', 'bachelors_rate']
    df = df.dropna(subset=core_cols)

    # Fill missing school/library features with 0 (ZIPs with no match in the
    # infrastructure data — roughly 2k rows for schools, 38k for libraries).
    infra_cols = [
        'school_count', 'avg_school_dist_km', 'min_school_dist_km',
        'library_count', 'avg_library_dist_km', 'min_library_dist_km',
    ]
    for col in infra_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0  # column absent from pkl (pre-enrichment pkl)

    # Filter out extreme outliers for a more stable linear model
    df = df[(df['price'] < 5_000_000) & (df['price'] > 100_000)]
    df = df[df['house_size'] < 10_000]

    # Encode city as category codes
    df['city_encoded'] = df['city'].astype('category').cat.codes

    print(f"Dataset size after cleaning/filtering: {len(df)}")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Train the OLS (Linear Regression) model on the cleaned dataframe.

    Returns a dict with keys:
        model       - fitted LinearRegression
        scaler      - fitted StandardScaler
        X_train     - unscaled training features (DataFrame)
        X_test      - unscaled test features (DataFrame)
        X_train_scaled - scaled training features (ndarray)
        X_test_scaled  - scaled test features (ndarray)
        y_train     - training target (Series)
        y_test      - test target (Series)
        features    - list of feature names
    """
    X = df[FEATURES]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Standardizing features and training model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "features": FEATURES,
    }


def evaluate_model(artifacts: dict) -> dict:
    """
    Evaluate the trained model and return a metrics dict.
    Also runs 5-fold CV on the training set.
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    X_train = artifacts["X_train"]
    X_test_scaled = artifacts["X_test_scaled"]
    y_test = artifacts["y_test"]

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

    # 5-fold cross-validation
    cv_pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    cv_scores = cross_val_score(cv_pipeline, X_train, artifacts["y_train"], cv=5, scoring='r2')

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "y_pred": y_pred,
    }


def print_results(artifacts: dict, metrics: dict) -> None:
    """Pretty-print model metrics and feature coefficients."""
    model    = artifacts["model"]
    features = artifacts["features"]
    y_test   = artifacts["y_test"]

    print("=" * 50)
    print("LINEAR REGRESSION RESULTS (CA 2022)")
    print(f"Evaluated on test set ({len(y_test):,} listings)")
    print("=" * 50)
    print(f"{'Metric':<30} {'Value':>12}")
    print("-" * 50)
    rmse_str = f"${metrics['rmse']:,.0f}"
    mae_str  = f"${metrics['mae']:,.0f}"
    print(f"{'R²':<30} {metrics['r2']:>12.4f}")
    print(f"{'RMSE':<30} {rmse_str:>12}")
    print(f"{'MAE':<30} {mae_str:>12}")
    print(f"{'MAPE':<30} {metrics['mape']:>11.2f}%")
    print(f"{'CV R² – mean (5-fold)':<30} {metrics['cv_r2_mean']:>12.4f}")
    print(f"{'CV R² – std  (5-fold)':<30} {metrics['cv_r2_std']:>12.4f}")
    print()
    print("Coefficients (sorted by absolute magnitude):")
    coef = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
    for feat, val in coef.items():
        print(f"  {feat:<20} {val:>12.2f}")


def save_results(artifacts: dict, metrics: dict, results_path: str = RESULTS_PATH) -> None:
    """Save metrics, coefficients, and sample actuals/predictions to JSON."""
    model    = artifacts["model"]
    features = artifacts["features"]
    y_test   = artifacts["y_test"]
    y_pred   = metrics["y_pred"]

    coef = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
    results = {
        "metrics": {
            "r2":         round(metrics["r2"], 4),
            "rmse":       round(metrics["rmse"]),
            "mae":        round(metrics["mae"]),
            "mape":       round(metrics["mape"], 4),
            "cv_r2_mean": round(metrics["cv_r2_mean"], 4),
            "cv_r2_std":  round(metrics["cv_r2_std"], 4),
        },
        "intercept": round(model.intercept_),
        "coefficients": {feat: round(val, 2) for feat, val in coef.items()},
        "actual":    [round(v) for v in y_test.values[:300].tolist()],
        "predicted": [round(max(0, v)) for v in y_pred[:300].tolist()],
        "residuals": [round(a - p) for a, p in zip(
            y_test.values[:300].tolist(), y_pred[:300].tolist()
        )],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df        = load_and_clean_data()
    artifacts = train_model(df)
    metrics   = evaluate_model(artifacts)
    print_results(artifacts, metrics)
    save_results(artifacts, metrics)
