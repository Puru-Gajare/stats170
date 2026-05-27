import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# ============================================================
# Random Forest — CA 2022 Sold Homes
# Version: log target encoding + RandomizedSearchCV tuning
#
# Run normally:
#   python random_forest_log_tuned.py
#
# Or pass a pkl path:
#   python random_forest_log_tuned.py us-housing-dataset/linear_model/ca_2022_sold.pkl
# ============================================================

DEFAULT_DATA_PATH = "us-housing-dataset/linear_model/ca_2022_sold.pkl"
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DATA_PATH)

# Set this to False if tuning takes too long and you only want the default model.
RUN_RANDOMIZED_SEARCH = True
N_ITER_SEARCH = 5     
CV_FOLDS = 3           # 3 is faster; 5 is more stable but slower

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Put the pkl file in the same folder as this script "
        "or pass the path as a command-line argument."
    )

# Load updated cleaned data
df = pd.read_pickle(DATA_PATH)

# Basic cleaning
df = df.drop_duplicates()

needed_cols = [
    "price", "bed", "bath", "acre_lot", "city",
    "state", "zip_code", "house_size"
]

missing_needed = [col for col in needed_cols if col not in df.columns]
if missing_needed:
    raise ValueError(f"Missing required columns in the pkl file: {missing_needed}")

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

# ------------------------------------------------------------
# New influencing-factor features: included only if present
# ------------------------------------------------------------
school_features = [
    "school_count",
    "avg_school_dist_km",
    "min_school_dist_km"
]

library_features = [
    "library_count",
    "avg_library_dist_km",
    "min_library_dist_km"
]

extra_factor_features = [
    col for col in school_features + library_features
    if col in df.columns
]

missing_extra_features = [
    col for col in school_features + library_features
    if col not in df.columns
]

print("Extra influencing-factor features found:")
print(extra_factor_features)

if missing_extra_features:
    print("\nExtra influencing-factor features not found, so they will be skipped:")
    print(missing_extra_features)

# Convert new factor columns to numeric and fill missing values.
for col in extra_factor_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if "count" in col:
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(df[col].median())

# Split first to avoid data leakage
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# ============================================================
# CHANGE #4: log-price target encoding instead of raw-price encoding
# Because y_train is log1p(price), encode zip/city using mean log-price.
# This keeps the encoded location features on the same scale as the model target.
# ============================================================
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

# Extra engineered structural features
for data in [train_df, test_df]:
    data["rooms_total"] = data["bed"] + data["bath"]
    data["bath_per_bed"] = data["bath"] / data["bed"]
    data["sqft_per_bed"] = data["house_size"] / data["bed"]
    data["log_house_size"] = np.log1p(data["house_size"])
    data["log_acre_lot"] = np.log1p(data["acre_lot"])

# Features: replace raw zip_avg_price/city_avg_price with log versions
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
    "city_avg_log_price"
]

features = base_features + extra_factor_features

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["log_price"]
y_test_actual = test_df["price"]

# Default model, used either directly or as estimator for tuning
base_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# CHANGE #5: Hyperparameter tuning with RandomizedSearchCV
# This searches several RF settings and picks the best by CV R^2.
# Since the model predicts log(price), CV R^2 is measured on log(price).
# Final test metrics below are still converted back to dollars.
# ============================================================
if RUN_RANDOMIZED_SEARCH:
    param_distributions = {
        "n_estimators": [300, 400],
        "max_depth": [30, 40],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.7],
        "bootstrap": [True]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="r2",
        cv=2,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_
    best_params = search.best_params_
    best_cv_r2 = search.best_score_
else:
    model = base_model
    model.fit(X_train, y_train)
    best_params = model.get_params()
    best_cv_r2 = None

# Predict back to dollars
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Prevent impossible negative prices after inverse transform, just in case
y_pred = np.maximum(y_pred, 0)

# Metrics in dollar scale
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

import matplotlib.pyplot as plt
import os

# Create folder for plots
os.makedirs("rf_visualizations", exist_ok=True)

# Residuals
residuals = y_test_actual - y_pred

# ==============================
# 1. Actual vs Predicted Plot
# ==============================
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.4)

min_price = min(y_test_actual.min(), y_pred.min())
max_price = max(y_test_actual.max(), y_pred.max())

plt.plot([min_price, max_price], [min_price, max_price], linestyle="--")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted Home Prices")
plt.ticklabel_format(style="plain", axis="both")
plt.tight_layout()
plt.savefig("rf_visualizations/rf_actual_vs_predicted.png", dpi=300)
plt.show()


# ==============================
# 2. Feature Importance Plot
# ==============================
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(9, 7))
plt.barh(importance.index, importance.values)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("rf_visualizations/rf_feature_importance.png", dpi=300)
plt.show()


# ==============================
# 3. Residuals vs Predicted Plot
# ==============================
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Price")
plt.ylabel("Residuals: Actual - Predicted")
plt.title("Random Forest Residuals vs Predicted Prices")
plt.ticklabel_format(style="plain", axis="both")
plt.tight_layout()
plt.savefig("rf_visualizations/rf_residuals_vs_predicted.png", dpi=300)
plt.show()


# ==============================
# 4. Residual Histogram
# ==============================
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=50)

plt.xlabel("Residuals: Actual - Predicted")
plt.ylabel("Frequency")
plt.title("Random Forest Residual Distribution")
plt.ticklabel_format(style="plain", axis="x")
plt.tight_layout()
plt.savefig("rf_visualizations/rf_residual_histogram.png", dpi=300)
plt.show()


print("\nSaved visualizations to folder: rf_visualizations")

print("=" * 65)
print("RANDOM FOREST — LOG TARGET ENCODING + HYPERPARAMETER TUNING")
print("=" * 65)
print(f"Data file : {DATA_PATH}")
print(f"Rows used : {len(df):,}")
print(f"Features  : {len(features)}")
print(f"R²        : {r2:.4f}")
print(f"RMSE      : ${rmse:,.0f}")
print(f"MAE       : ${mae:,.0f}")

if best_cv_r2 is not None:
    print(f"Best CV R² on log(price): {best_cv_r2:.4f}")

print("\nBest Parameters:")
print(best_params)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Save results
results = {
    "model_name": "Random Forest CA 2022 Sold Homes - Log Target Encoding + Tuning",
    "data_file": str(DATA_PATH),
    "rows_used": len(df),
    "features_used": features,
    "extra_influencing_factor_features_used": extra_factor_features,
    "extra_influencing_factor_features_missing": missing_extra_features,
    "target_encoding": "zip/city mean log1p(price), training data only",
    "randomized_search": {
        "enabled": RUN_RANDOMIZED_SEARCH,
        "n_iter": N_ITER_SEARCH if RUN_RANDOMIZED_SEARCH else None,
        "cv_folds": CV_FOLDS if RUN_RANDOMIZED_SEARCH else None,
        "best_cv_r2_log_price": round(best_cv_r2, 4) if best_cv_r2 is not None else None,
        "best_params": best_params
    },
    "metrics_dollar_scale": {
        "r2": round(r2, 4),
        "rmse": round(rmse),
        "mae": round(mae)
    },
    "feature_importance": {
        feature: round(value, 4)
        for feature, value in importance.items()
    },
    "actual": [round(v) for v in y_test_actual.values[:1000].tolist()],
    "predicted": [round(v) for v in y_pred[:1000].tolist()],
    "residuals": [
        round(a - p)
        for a, p in zip(y_test_actual.values[:1000].tolist(), y_pred[:1000].tolist())
    ]
}

output_path = "rf_ca_2022_log_tuned_results.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {output_path}")
