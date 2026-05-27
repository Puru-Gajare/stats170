"""
Random Forest visualization dashboard for CA 2022 housing project.

This script trains ONE Random Forest model using the current feature set,
then creates a clean dashboard similar to your teammate's graph.

Outputs saved to: rf_visualizations/

Run:
    python rf_model_dashboard.py

Or with a custom pkl path:
    python rf_model_dashboard.py us-housing-dataset/linear_model/ca_2022_sold.pkl
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ==============================
# Settings
# ==============================
DEFAULT_DATA_PATH = "us-housing-dataset/linear_model/ca_2022_sold.pkl"
OUTPUT_DIR = "rf_visualizations"
RANDOM_STATE = 42

# Use tuned/balanced settings. This trains ONE model only, not RandomizedSearchCV.
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ==============================
# Helper functions
# ==============================
def money_millions(values):
    """Convert dollar values to millions of dollars."""
    return np.asarray(values) / 1_000_000


def money_thousands(values):
    """Convert dollar values to thousands of dollars."""
    return np.asarray(values) / 1_000


def load_and_prepare_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find data file: {path}\n"
            "Either place the pkl file at this path or pass the path as a command-line argument."
        )

    df = pd.read_pickle(path)
    df = df.drop_duplicates()

    needed_cols = [
        "price", "bed", "bath", "acre_lot", "city",
        "state", "zip_code", "house_size"
    ]
    df = df.dropna(subset=needed_cols)

    # Keep realistic values
    df = df[
        (df["price"] > 50_000) &
        (df["price"] < 10_000_000) &
        (df["house_size"] > 200) &
        (df["house_size"] < 20_000) &
        (df["bed"] > 0) &
        (df["bath"] > 0) &
        (df["acre_lot"] >= 0) &
        (df["acre_lot"] < 20)
    ].copy()

    # Clean categorical columns
    df["zip_code"] = df["zip_code"].astype(str).str.replace(".0", "", regex=False)
    df["city"] = df["city"].astype(str).str.lower().str.strip()

    return df


def add_features(train_df, test_df):
    # Log-price target encoding using TRAINING DATA ONLY to avoid leakage
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["log_price"] = np.log1p(train_df["price"])
    global_mean_log_price = train_df["log_price"].mean()

    zip_avg_log = train_df.groupby("zip_code")["log_price"].mean()
    city_avg_log = train_df.groupby("city")["log_price"].mean()

    train_df["zip_avg_log_price"] = train_df["zip_code"].map(zip_avg_log).fillna(global_mean_log_price)
    test_df["zip_avg_log_price"] = test_df["zip_code"].map(zip_avg_log).fillna(global_mean_log_price)

    train_df["city_avg_log_price"] = train_df["city"].map(city_avg_log).fillna(global_mean_log_price)
    test_df["city_avg_log_price"] = test_df["city"].map(city_avg_log).fillna(global_mean_log_price)

    # Extra engineered features
    for data in [train_df, test_df]:
        data["rooms_total"] = data["bed"] + data["bath"]
        data["bath_per_bed"] = data["bath"] / data["bed"]
        data["sqft_per_bed"] = data["house_size"] / data["bed"]
        data["log_house_size"] = np.log1p(data["house_size"])
        data["log_acre_lot"] = np.log1p(data["acre_lot"])

    base_features = [
        "bed",
        "bath",
        "acre_lot",
        "house_size",
        "rooms_total",
        "bath_per_bed",
        "sqft_per_bed",
        "log_house_size",
        "log_acre_lot",
        "zip_avg_log_price",
        "city_avg_log_price",
    ]

    # Automatically include new influencing-factor columns if they exist
    possible_extra_features = [
        "school_count",
        "avg_school_dist_km",
        "min_school_dist_km",
        "library_count",
        "avg_library_dist_km",
        "min_library_dist_km",
    ]

    extra_features = [col for col in possible_extra_features if col in train_df.columns]

    if extra_features:
        print("Extra influencing-factor features found:")
        print(extra_features)

        # Fill missing values in new numeric features using training median only
        for col in extra_features:
            median_value = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_value)
            test_df[col] = test_df[col].fillna(median_value)

    features = base_features + extra_features
    return train_df, test_df, features


def make_price_range_labels(actual_prices):
    bins = [0, 500_000, 1_000_000, 2_000_000, np.inf]
    labels = ["<$500k", "$500k–$1M", "$1M–$2M", ">$2M"]
    return pd.cut(actual_prices, bins=bins, labels=labels, include_lowest=True)


# ==============================
# Main
# ==============================
def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_prepare_data(data_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_df, test_df, features = add_features(train_df, test_df)

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = np.log1p(train_df["price"])
    y_test_actual = test_df["price"].values

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    residuals = y_test_actual - y_pred
    abs_error = np.abs(residuals)

    print("=" * 55)
    print("RANDOM FOREST — CA 2022 SOLD HOMES")
    print("=" * 55)
    print(f"Rows used : {len(df):,}")
    print(f"Features  : {len(features)}")
    print(f"R²        : {r2:.4f}")
    print(f"RMSE      : ${rmse:,.0f}")
    print(f"MAE       : ${mae:,.0f}")

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    # Results dataframe for grouped plots
    results_df = pd.DataFrame({
        "actual": y_test_actual,
        "predicted": y_pred,
        "residual": residuals,
        "abs_error": abs_error,
    })
    results_df["price_range"] = make_price_range_labels(results_df["actual"])

    mae_by_range = (
        results_df.groupby("price_range", observed=False)["abs_error"]
        .mean()
        .reindex(["<$500k", "$500k–$1M", "$1M–$2M", ">$2M"])
    )

    # Accuracy curve: percent of homes within different absolute error bounds
    error_bounds_k = np.arange(0, 501, 25)  # 0k to 500k
    accuracy_within_bound = [np.mean(abs_error <= bound * 1000) * 100 for bound in error_bounds_k]

    # ==============================
    # Dashboard like teammate's graph
    # ==============================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("California Housing 2022 — Random Forest Model Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Accuracy and error summary
    ax1 = axes[0]
    ax1b = ax1.twinx()

    metric_names = ["R² Score", "MAE", "RMSE"]
    x = np.arange(len(metric_names))

    ax1.bar(x[0], r2, width=0.5, label="R² Score")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("R² Score")

    ax1b.bar(x[1:], [mae / 1000, rmse / 1000], width=0.5, hatch="//", alpha=0.55, label="Error")
    ax1b.set_ylabel("Prediction Error ($K)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.set_title("Model Accuracy & Error Summary")
    ax1.grid(axis="y", alpha=0.25)

    ax1.text(x[0], r2 + 0.025, f"{r2:.3f}", ha="center", fontweight="bold")
    ax1b.text(x[1], mae / 1000 + 10, f"${mae/1000:.0f}k", ha="center", fontweight="bold")
    ax1b.text(x[2], rmse / 1000 + 10, f"${rmse/1000:.0f}k", ha="center", fontweight="bold")

    # Panel 2: MAE by home price range
    ax2 = axes[1]
    ax2.bar(mae_by_range.index.astype(str), mae_by_range.values / 1000)
    ax2.set_title("Mean Absolute Error by Home Price Range")
    ax2.set_xlabel("Actual Home Price Range")
    ax2.set_ylabel("Mean Absolute Error ($K)")
    ax2.grid(axis="y", alpha=0.25)

    for i, value in enumerate(mae_by_range.values / 1000):
        if not np.isnan(value):
            ax2.text(i, value + 8, f"${value:.0f}k", ha="center", fontweight="bold", fontsize=9)

    # Panel 3: Accuracy curve
    ax3 = axes[2]
    ax3.plot(error_bounds_k, accuracy_within_bound, marker="o", markersize=3, linewidth=2)
    ax3.set_title("Prediction Accuracy Curve")
    ax3.set_xlabel("Absolute Error Bound ($K)")
    ax3.set_ylabel("% of Homes Within Error Bound")
    ax3.set_xlim(0, 500)
    ax3.set_ylim(0, 100)
    ax3.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    dashboard_path = os.path.join(OUTPUT_DIR, "rf_model_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # Actual vs predicted, clearer units
    # ==============================
    plt.figure(figsize=(9, 7))
    actual_m = money_millions(y_test_actual)
    pred_m = money_millions(y_pred)
    plt.scatter(actual_m, pred_m, alpha=0.35, s=18)

    max_axis = max(actual_m.max(), pred_m.max())
    plt.plot([0, max_axis], [0, max_axis], linestyle="--", linewidth=2, label="Perfect prediction")

    plt.title(f"Actual vs. Predicted — Random Forest\nR² = {r2:.4f} | RMSE = ${rmse:,.0f} | MAE = ${mae:,.0f}", fontsize=14)
    plt.xlabel("Actual Sale Price ($M)")
    plt.ylabel("Predicted Sale Price ($M)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    actual_pred_path = os.path.join(OUTPUT_DIR, "rf_actual_vs_predicted_millions.png")
    plt.savefig(actual_pred_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ==============================
    # Feature importance
    # ==============================
    top_importance = importance.head(15).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_importance.index, top_importance.values)
    plt.title("Random Forest Feature Importance — Top 15 Features", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    feature_path = os.path.join(OUTPUT_DIR, "rf_feature_importance_top15.png")
    plt.savefig(feature_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Save numerical results for report use
    output_json = {
        "rows_used": int(len(df)),
        "features_used": features,
        "metrics": {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse)),
            "mae": round(float(mae)),
        },
        "mae_by_price_range": {
            str(k): round(float(v), 2) if not pd.isna(v) else None
            for k, v in mae_by_range.items()
        },
        "feature_importance": {
            k: round(float(v), 5)
            for k, v in importance.items()
        },
        "rf_params": RF_PARAMS,
    }

    with open(os.path.join(OUTPUT_DIR, "rf_dashboard_results.json"), "w") as f:
        json.dump(output_json, f, indent=2)

    print("\nSaved graphs:")
    print(f"- {dashboard_path}")
    print(f"- {actual_pred_path}")
    print(f"- {feature_path}")
    print(f"- {os.path.join(OUTPUT_DIR, 'rf_dashboard_results.json')}")


if __name__ == "__main__":
    main()
