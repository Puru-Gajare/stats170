import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "us-housing-dataset/ca_2022_sold.pkl"
OUTPUT_DIR = "rf_explainability"

SHAP_SAMPLE_SIZE = 300
PERM_SAMPLE_SIZE = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

df = pd.read_pickle(data_path)
df = df.drop_duplicates()

# Convert numeric columns safely
numeric_cols = [
    "price", "bed", "bath", "acre_lot", "house_size",
    "school_count", "avg_school_dist_km", "min_school_dist_km",
    "library_count", "avg_library_dist_km", "min_library_dist_km"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

needed_cols = [
    "price", "bed", "bath", "acre_lot", "city",
    "state", "zip_code", "house_size"
]

extra_features = [
    "school_count",
    "avg_school_dist_km",
    "min_school_dist_km",
    "library_count",
    "avg_library_dist_km",
    "min_library_dist_km"
]

extra_features = [col for col in extra_features if col in df.columns]

print("Extra influencing-factor features found:")
print(extra_features)

needed_cols += extra_features
df = df.dropna(subset=needed_cols)

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

df["zip_code"] = df["zip_code"].astype(str).str.replace(".0", "", regex=False)
df["city"] = df["city"].astype(str).str.lower().str.strip()

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Log-price target encoding
train_df["log_price"] = np.log1p(train_df["price"].astype(float))
global_mean_log_price = train_df["log_price"].mean()

zip_avg_log = train_df.groupby("zip_code")["log_price"].mean()
city_avg_log = train_df.groupby("city")["log_price"].mean()

train_df["zip_avg_log_price"] = train_df["zip_code"].map(zip_avg_log).fillna(global_mean_log_price)
test_df["zip_avg_log_price"] = test_df["zip_code"].map(zip_avg_log).fillna(global_mean_log_price)

train_df["city_avg_log_price"] = train_df["city"].map(city_avg_log).fillna(global_mean_log_price)
test_df["city_avg_log_price"] = test_df["city"].map(city_avg_log).fillna(global_mean_log_price)

# Engineered features
for data in [train_df, test_df]:
    data["rooms_total"] = data["bed"] + data["bath"]
    data["bath_per_bed"] = data["bath"] / data["bed"]
    data["sqft_per_bed"] = data["house_size"] / data["bed"]
    data["log_house_size"] = np.log1p(data["house_size"].astype(float))
    data["log_acre_lot"] = np.log1p(data["acre_lot"].astype(float))

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
    "city_avg_log_price"
] + extra_features

X_train = train_df[features]
X_test = test_df[features]

y_train = np.log1p(train_df["price"].astype(float))
y_test_actual = test_df["price"].astype(float)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("\nRandom Forest Results")
print("=" * 45)
print(f"Rows used : {len(df):,}")
print(f"Features  : {len(features)}")
print(f"R²        : {r2:.4f}")
print(f"RMSE      : ${rmse:,.0f}")
print(f"MAE       : ${mae:,.0f}")

# 1. Feature Importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance.index, importance.values)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_rf_feature_importance.png", dpi=300)
plt.close()

print("Saved feature importance plot.")

# 2. Permutation Importance using sample
print("Running permutation importance on sample...")

X_perm = X_test.sample(
    n=min(PERM_SAMPLE_SIZE, len(X_test)),
    random_state=42
)

y_perm = y_test_actual.loc[X_perm.index]

perm_result = permutation_importance(
    model,
    X_perm,
    y_perm,
    n_repeats=2,
    random_state=42,
    n_jobs=-1
)

perm_importance = pd.Series(
    perm_result.importances_mean,
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(perm_importance.index, perm_importance.values)
plt.xlabel("Mean Decrease in R² After Shuffling")
plt.ylabel("Feature")
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_permutation_importance.png", dpi=300)
plt.close()

print("Saved permutation importance plot.")

# 3. SHAP plots using small sample
try:
    import shap

    print("Running SHAP on small sample...")

    X_shap = X_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_test)),
        random_state=42
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_shap_bar_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved SHAP plots.")

except ImportError:
    print("SHAP is not installed. Run: pip install shap")
except Exception as e:
    print(f"SHAP skipped because of error: {e}")

# 4. Partial Dependence Plots
print("Creating partial dependence plots...")

pdp_features = [
    f for f in [
        "house_size",
        "zip_avg_log_price",
        "city_avg_log_price",
        "school_count",
        "library_count",
        "min_school_dist_km",
        "min_library_dist_km"
    ]
    if f in features
]

for feature in pdp_features:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        model,
        X_test.sample(n=min(1000, len(X_test)), random_state=42),
        [feature],
        ax=ax
    )
    ax.set_title(f"Partial Dependence of {feature}")
    ax.set_ylabel("Predicted Log Price")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_pdp_{feature}.png", dpi=300)
    plt.close(fig)

print("Saved partial dependence plots.")

# 5. Actual vs Predicted
actual_m = y_test_actual / 1_000_000
pred_m = y_pred / 1_000_000

plt.figure(figsize=(8, 6))
plt.scatter(actual_m, pred_m, alpha=0.35)

min_price = min(actual_m.min(), pred_m.min())
max_price = max(actual_m.max(), pred_m.max())

plt.plot([min_price, max_price], [min_price, max_price], linestyle="--")

plt.xlabel("Actual Sale Price ($M)")
plt.ylabel("Predicted Sale Price ($M)")
plt.title("Random Forest: Actual vs Predicted Home Prices")
plt.text(
    0.05,
    0.95,
    f"R² = {r2:.4f}\nRMSE = ${rmse:,.0f}\nMAE = ${mae:,.0f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.2)
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_actual_vs_predicted_millions.png", dpi=300)
plt.close()

print("Saved actual vs predicted plot.")

# Save results
results = {
    "model_name": "Random Forest Explainability",
    "rows_used": len(df),
    "features_used": features,
    "extra_influencing_features": extra_features,
    "metrics": {
        "r2": round(r2, 4),
        "rmse": round(rmse),
        "mae": round(mae)
    },
    "feature_importance": {
        feature: round(value, 5)
        for feature, value in importance.sort_values(ascending=False).items()
    },
    "permutation_importance": {
        feature: round(value, 5)
        for feature, value in perm_importance.sort_values(ascending=False).items()
    }
}

with open(f"{OUTPUT_DIR}/rf_explainability_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone.")
print(f"Saved all outputs to: {OUTPUT_DIR}")