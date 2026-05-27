"""
rf_visualizations.py

Creates clearer Random Forest visualizations for the CA 2022 housing project.
This is a separate visualization file that:
1. Loads the cleaned pkl dataset
2. Applies the same cleaning + feature engineering used in random_forest.py
3. Trains one Random Forest model
4. Saves clear graphs with units in the axis labels

Run from your project folder:
    D:\Python\python.exe e:/stats170/stats170/rf_visualizations.py

Or pass a custom pkl path:
    D:\Python\python.exe e:/stats170/stats170/rf_visualizations.py us-housing-dataset/linear_model/ca_2022_sold.pkl
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# Settings
# ==============================
DEFAULT_DATA_PATH = "us-housing-dataset/linear_model/ca_2022_sold.pkl"
OUTPUT_DIR = "rf_visualizations"
RANDOM_STATE = 42

# To keep graphs readable, scatter plots use a sample of the test set.
# Set to None if you want to plot every test point.
PLOT_SAMPLE_SIZE = 5000


# ==============================
# Load data
# ==============================
def get_data_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return DEFAULT_DATA_PATH


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Put the pkl file in this path or pass the path as a command-line argument."
        )
    return pd.read_pickle(path)


# ==============================
# Clean + feature engineer
# ==============================
def prepare_data(df: pd.DataFrame):
    df = df.drop_duplicates().copy()

    needed_cols = [
        "price", "bed", "bath", "acre_lot", "city",
        "state", "zip_code", "house_size"
    ]

    missing_needed = [col for col in needed_cols if col not in df.columns]
    if missing_needed:
        raise ValueError(f"Missing required columns: {missing_needed}")

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

    # Split first to avoid target leakage
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Location target encoding using TRAINING DATA ONLY
    # This version uses log-price averages because the model target is log(price + 1).
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

    # Base features
    features = [
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

    # Add new influencing-factor features if they exist in the pkl file
    possible_extra_features = [
        "school_count",
        "avg_school_dist_km",
        "min_school_dist_km",
        "library_count",
        "avg_library_dist_km",
        "min_library_dist_km",
    ]

    extra_features = [col for col in possible_extra_features if col in df.columns]
    if extra_features:
        print("Extra influencing-factor features found:")
        print(extra_features)
        features.extend(extra_features)

        # Fill missing values only for these extra numerical features
        for col in extra_features:
            median_value = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_value)
            test_df[col] = test_df[col].fillna(median_value)
    else:
        print("No school/library influencing-factor features found in the pkl file.")

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = np.log1p(train_df["price"])
    y_test_actual = test_df["price"]

    return train_df, test_df, X_train, X_test, y_train, y_test_actual, features


# ==============================
# Train model
# ==============================
def train_model(X_train, y_train) -> RandomForestRegressor:
    # One-model version, not RandomizedSearchCV, so it runs much faster.
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ==============================
# Helper for readable money axis
# ==============================
def dollars_to_millions(values):
    return np.asarray(values) / 1_000_000


# ==============================
# Create visualizations
# ==============================
def make_visualizations(model, X_test, y_test_actual, y_pred, features, metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_df = pd.DataFrame({
        "actual_price": y_test_actual.values,
        "predicted_price": y_pred,
        "residual": y_test_actual.values - y_pred,
        "abs_error": np.abs(y_test_actual.values - y_pred),
    })

    if PLOT_SAMPLE_SIZE is not None and len(plot_df) > PLOT_SAMPLE_SIZE:
        plot_df_scatter = plot_df.sample(PLOT_SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        plot_df_scatter = plot_df.copy()

    actual_m = dollars_to_millions(plot_df_scatter["actual_price"])
    pred_m = dollars_to_millions(plot_df_scatter["predicted_price"])
    residual_m = dollars_to_millions(plot_df_scatter["residual"])

    r2 = metrics["r2"]
    rmse = metrics["rmse"]
    mae = metrics["mae"]

    # ------------------------------
    # 1. Actual vs. Predicted
    # ------------------------------
    plt.figure(figsize=(9, 7))
    plt.scatter(actual_m, pred_m, alpha=0.35, s=18, label="Home sale")

    min_axis = 0
    max_axis = max(actual_m.max(), pred_m.max()) * 1.05
    plt.plot(
        [min_axis, max_axis],
        [min_axis, max_axis],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel("Actual Sale Price ($M)", fontsize=12)
    plt.ylabel("Predicted Sale Price ($M)", fontsize=12)
    plt.title(
        f"Actual vs. Predicted — Random Forest\n"
        f"R² = {r2:.4f} | RMSE = ${rmse:,.0f} | MAE = ${mae:,.0f}",
        fontsize=15
    )
    plt.xlim(min_axis, max_axis)
    plt.ylim(min_axis, max_axis)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_actual_vs_predicted_millions.png"), dpi=300)
    plt.close()

    # ------------------------------
    # 2. Feature Importance
    # ------------------------------
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(importance.index, importance.values)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Random Forest Feature Importance", fontsize=15)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"), dpi=300)
    plt.close()

    # ------------------------------
    # 3. Residuals vs. Predicted
    # ------------------------------
    plt.figure(figsize=(9, 7))
    plt.scatter(pred_m, residual_m, alpha=0.35, s=18)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.xlabel("Predicted Sale Price ($M)", fontsize=12)
    plt.ylabel("Residual: Actual - Predicted ($M)", fontsize=12)
    plt.title(
        "Random Forest Residuals vs. Predicted Price\n"
        "Positive residual = model underpredicted; negative residual = model overpredicted",
        fontsize=14
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_residuals_vs_predicted_millions.png"), dpi=300)
    plt.close()

    # ------------------------------
    # 4. Residual Distribution
    # ------------------------------
    residual_all_m = dollars_to_millions(plot_df["residual"])

    plt.figure(figsize=(9, 7))
    plt.hist(residual_all_m, bins=60)
    plt.axvline(0, linestyle="--", linewidth=2, label="Zero error")
    plt.xlabel("Residual: Actual - Predicted ($M)", fontsize=12)
    plt.ylabel("Number of Homes", fontsize=12)
    plt.title("Random Forest Residual Distribution", fontsize=15)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_residual_distribution_millions.png"), dpi=300)
    plt.close()

    # ------------------------------
    # 5. Error by Price Range
    # ------------------------------
    bins = [0, 500_000, 1_000_000, 2_000_000, 3_000_000, 10_000_000]
    labels = ["<$500K", "$500K-$1M", "$1M-$2M", "$2M-$3M", ">$3M"]
    plot_df["price_range"] = pd.cut(plot_df["actual_price"], bins=bins, labels=labels, include_lowest=True)

    error_by_range = (
        plot_df.groupby("price_range", observed=False)["abs_error"]
        .mean()
        .reindex(labels)
    ) / 1_000_000

    plt.figure(figsize=(9, 6))
    plt.bar(error_by_range.index.astype(str), error_by_range.values)
    plt.xlabel("Actual Sale Price Range", fontsize=12)
    plt.ylabel("Mean Absolute Error ($M)", fontsize=12)
    plt.title("Random Forest Error by Home Price Range", fontsize=15)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_error_by_price_range.png"), dpi=300)
    plt.close()

    # Save feature importance as csv for report tables
    importance.sort_values(ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"),
        header=["importance"]
    )

    # Save summary json
    summary = {
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse)),
        "mae": round(float(mae)),
        "plots_saved_to": OUTPUT_DIR,
        "features_used": features,
    }
    with open(os.path.join(OUTPUT_DIR, "rf_visualization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ==============================
# Main
# ==============================
def main():
    data_path = get_data_path()
    print(f"Loading data from: {data_path}")

    df = load_data(data_path)
    train_df, test_df, X_train, X_test, y_train, y_test_actual, features = prepare_data(df)

    print(f"Rows used : {len(train_df) + len(test_df):,}")
    print(f"Features  : {len(features)}")

    print("Training Random Forest model...")
    model = train_model(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    print("=" * 55)
    print("RANDOM FOREST VISUALIZATION RESULTS")
    print("=" * 55)
    print(f"R²        : {r2:.4f}")
    print(f"RMSE      : ${rmse:,.0f}")
    print(f"MAE       : ${mae:,.0f}")

    metrics = {"r2": r2, "rmse": rmse, "mae": mae}
    make_visualizations(model, X_test, y_test_actual, y_pred, features, metrics)

    print(f"\nSaved graphs to folder: {OUTPUT_DIR}")
    print("Main report/poster graphs:")
    print(f"- {OUTPUT_DIR}/rf_actual_vs_predicted_millions.png")
    print(f"- {OUTPUT_DIR}/rf_feature_importance.png")
    print(f"- {OUTPUT_DIR}/rf_error_by_price_range.png")


if __name__ == "__main__":
    main()
