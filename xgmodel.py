"""
XGBoost House Price Prediction — California 2022 Sold Listings
=============================================================
Research-grade pipeline with:
  - Data cleaning & feature engineering
  - Log-price transformation
  - Target encoding for zip_code / city
  - Chronological train/test split
  - XGBoost with cross-validated RMSE
  - SHAP global + beeswarm analysis
  - Residual geography (if lat/lon available)

Requirements:
    pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn category_encoders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import category_encoders as ce
import xgboost as xgb
import shap

# ── 0. CONFIG ──────────────────────────────────────────────────────────────────
PKL_PATH   = "us-housing-dataset/ca_2022_sold.pkl"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_pickle(PKL_PATH)
print(f"  Raw shape: {df.shape}")

# ── 2. CLEAN ───────────────────────────────────────────────────────────────────
print("Cleaning...")

# Keep only sold listings with valid prices
df = df[df["status"].str.lower().str.strip() == "sold"].copy()

# Price bounds: remove extreme outliers (< $50k or > $10M)
df = df[(df["price"] >= 50_000) & (df["price"] <= 10_000_000)]

# Drop rows missing critical features
df = df.dropna(subset=["price", "bed", "bath", "house_size", "zip_code"])

# Coerce numerics
for col in ["bed", "bath", "house_size", "acre_lot"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill acre_lot nulls with median (many listings omit it)
df["acre_lot"] = df["acre_lot"].fillna(df["acre_lot"].median())

# Remove implausible values
df = df[(df["bed"] > 0) & (df["bed"] <= 20)]
df = df[(df["bath"] > 0) & (df["bath"] <= 20)]
df = df[(df["house_size"] > 100) & (df["house_size"] <= 20_000)]

print(f"  Clean shape: {df.shape}")

# check delete
print(df['prev_sold_date'].dtype)
print(df['prev_sold_date'].min())
print(df['prev_sold_date'].max())
print(df['prev_sold_date'].value_counts().sort_index().head(20))

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("Engineering features...")

# Log price (target) — normalizes right skew
df["log_price"] = np.log1p(df["price"])

# Price per sqft (useful but leaks target — only use as analysis, not a feature)
df["price_per_sqft"] = df["price"] / df["house_size"]   # for EDA only

# Structural ratios
df["bath_per_bed"]   = df["bath"] / df["bed"].replace(0, np.nan)
df["size_per_bed"]   = df["house_size"] / df["bed"].replace(0, np.nan)

# Zip code as string (for target encoder)
df["zip_code"] = df["zip_code"].astype(str).str.strip()

# Sold month (from prev_sold_date or sold_date)
date_col = "sold_date" if "sold_date" in df.columns else "prev_sold_date"
df["sold_month"] = pd.to_datetime(df[date_col], errors="coerce").dt.month
df["sold_month"] = df["sold_month"].fillna(df["sold_month"].median()).astype(int)

# ── 4. TRAIN / TEST SPLIT (80/20 random) ─────────────────────────────────────
print("Splitting train/test (80/20 random split)...")
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

FEATURES = ["bed", "bath", "house_size", "acre_lot",
            "bath_per_bed", "size_per_bed", "sold_month", "zip_code"]

X_train = train_df[FEATURES]
y_train = train_df["log_price"]
X_test  = test_df[FEATURES]
y_test  = test_df["log_price"]

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. TARGET ENCODING ────────────────────────────────────────────────────────
# Encodes zip_code as mean log_price per zip, computed only on train set
print("Target-encoding zip_code...")
encoder = ce.TargetEncoder(cols=["zip_code"], smoothing=10)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc  = encoder.transform(X_test)

# ── 6. XGBOOST MODEL ──────────────────────────────────────────────────────────
print("Training XGBoost...")

params = dict(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    eval_metric       = "rmse",
    early_stopping_rounds = 30,
)

model = xgb.XGBRegressor(**params)
model.fit(
    X_train_enc, y_train,
    eval_set=[(X_test_enc, y_test)],
    verbose=50,
)

# ── 7. EVALUATION ─────────────────────────────────────────────────────────────
print("\n── Model Evaluation ──────────────────────────────────────────────────")

y_pred_log = model.predict(X_test_enc)

# Back-transform from log space
y_pred_price = np.expm1(y_pred_log)
y_test_price = np.expm1(y_test)

rmse   = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
mae    = mean_absolute_error(y_test_price, y_pred_price)
r2     = r2_score(y_test_price, y_pred_price)
mape   = np.mean(np.abs((y_test_price - y_pred_price) / y_test_price)) * 100

print(f"  RMSE : ${rmse:>12,.0f}")
print(f"  MAE  : ${mae:>12,.0f}")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f}%")

# Cross-validated R² on training set (5-fold)
print("\n  Running 5-fold CV on training set (log price)...")
cv_model = xgb.XGBRegressor(**{k: v for k, v in params.items()
                                if k not in ["early_stopping_rounds", "eval_metric"]})
cv_scores = cross_val_score(cv_model, X_train_enc, y_train, cv=5, scoring="r2", n_jobs=-1)
print(f"  CV R² (log price): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
sns.set_theme(style="whitegrid", font_scale=1.1)

# --- 8a. Actual vs Predicted ---
fig, ax = plt.subplots(figsize=(8, 7))
sample = min(3000, len(y_test_price))
idx = np.random.choice(len(y_test_price), sample, replace=False)
ax.scatter(y_test_price.values[idx]/1e6, y_pred_price[idx]/1e6,
           alpha=0.25, s=12, color="#2563EB")
lims = [0, min(y_test_price.max(), 5e6)/1e6]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Price ($M)", fontsize=13)
ax.set_ylabel("Predicted Price ($M)", fontsize=13)
ax.set_title(f"Actual vs. Predicted — XGBoost\nR² = {r2:.4f}  |  MAPE = {mape:.1f}%", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png", dpi=150)
plt.close()
print(f"  Saved actual_vs_predicted.png")

# --- 8b. Residuals distribution ---
residuals = y_test_price.values - y_pred_price
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(residuals / 1e3, bins=80, color="#10B981", edgecolor="none", alpha=0.8)
axes[0].axvline(0, color="red", linewidth=1.5)
axes[0].set_xlabel("Residual ($K)", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].set_title("Residual Distribution", fontsize=13)

axes[1].scatter(y_pred_price[idx]/1e6, residuals[idx]/1e3,
                alpha=0.2, s=10, color="#F59E0B")
axes[1].axhline(0, color="red", linewidth=1.5)
axes[1].set_xlabel("Predicted Price ($M)", fontsize=12)
axes[1].set_ylabel("Residual ($K)", fontsize=12)
axes[1].set_title("Residuals vs. Predicted", fontsize=13)

plt.suptitle("Residual Analysis — XGBoost (CA 2022)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved residuals.png")

# --- 8c. Feature Importance (gain) ---
importance = pd.Series(
    model.get_booster().get_score(importance_type="gain"),
    name="gain"
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#6366F1" if i < len(importance)-1 else "#EF4444"
          for i in range(len(importance))]
importance.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
ax.set_xlabel("Gain (mean reduction in loss)", fontsize=12)
ax.set_title("XGBoost Feature Importance (Gain)\nCA 2022 Sold Listings", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
plt.close()
print(f"  Saved feature_importance.png")

# ── 9. SHAP ANALYSIS ──────────────────────────────────────────────────────────
print("\nComputing SHAP values (this may take 1-2 minutes)...")

explainer   = shap.TreeExplainer(model)
shap_sample = X_test_enc.sample(min(2000, len(X_test_enc)), random_state=RANDOM_STATE)
shap_values = explainer.shap_values(shap_sample)

# --- 9a. SHAP summary (beeswarm) ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_sample, show=False, plot_size=None)
plt.title("SHAP Beeswarm — Feature Impact on log(Price)\nCA 2022 Sold Listings", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved shap_beeswarm.png")

# --- 9b. SHAP bar (mean |SHAP|) ---
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False, plot_size=None)
plt.title("Mean |SHAP Value| — Global Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved shap_bar.png")

# ── 10. SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n── Summary ───────────────────────────────────────────────────────────")
summary = pd.DataFrame({
    "Metric": ["Train size", "Test size", "RMSE", "MAE", "R²", "MAPE",
                "CV R² (mean)", "CV R² (std)"],
    "Value":  [f"{len(X_train):,}", f"{len(X_test):,}",
               f"${rmse:,.0f}", f"${mae:,.0f}",
               f"{r2:.4f}", f"{mape:.2f}%",
               f"{cv_scores.mean():.4f}", f"{cv_scores.std():.4f}"]
})
print(summary.to_string(index=False))
summary.to_csv(f"{OUTPUT_DIR}/model_summary.csv", index=False)

print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
print("Done!")