import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import json
import os

# ── Load & clean ──────────────────────────────────────────────────────────────
pkl_path = 'us-housing-dataset/ca_2022_sold.pkl'
print(f"Loading {pkl_path}...")
df = pd.read_pickle(pkl_path)

# Drop rows with missing values in core features
core_cols = ['price', 'bed', 'bath', 'house_size', 'acre_lot', 'city',
             'median_income', 'population', 'poverty_rate', 'bachelors_rate']
df = df.dropna(subset=core_cols)

# Filter out extreme outliers for a more stable linear model
# (prices above 5M and below 100k, or house sizes above 10k sqft)
df = df[(df['price'] < 5_000_000) & (df['price'] > 100_000)]
df = df[df['house_size'] < 10_000]

print(f"Dataset size after cleaning/filtering: {len(df)}")

# Encode city as category codes
df['city_encoded'] = df['city'].astype('category').cat.codes

# ── Features / target ─────────────────────────────────────────────────────────
# Features: bed, bath, house_size, acre_lot, city_encoded + Census features
features = ['bed', 'bath', 'house_size', 'acre_lot', 'city_encoded',
            'median_income', 'population', 'poverty_rate', 'bachelors_rate']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train with Scaling ────────────────────────────────────────────────────────
print("Standardizing features and training model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── Evaluate on test set ──────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

# ── 5-Fold Cross-Validation on training set ───────────────────────────────────
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5, scoring='r2')
cv_r2_mean = cv_scores.mean()
cv_r2_std  = cv_scores.std()

print("=" * 50)
print("LINEAR REGRESSION RESULTS (CA 2022)")
print(f"Evaluated on test set ({len(y_test):,} listings)")
print("=" * 50)
print(f"{'Metric':<30} {'Value':>12}")
print("-" * 50)
print(f"{'R²':<30} {r2:>12.4f}")
print(f"{'RMSE':<30} {'$'+f'{rmse:,.0f}':>12}")
print(f"{'MAE':<30} {'$'+f'{mae:,.0f}':>12}")
print(f"{'MAPE':<30} {mape:>11.2f}%")
print(f"{'CV R² – mean (5-fold)':<30} {cv_r2_mean:>12.4f}")
print(f"{'CV R² – std  (5-fold)':<30} {cv_r2_std:>12.4f}")
print()
print("Coefficients (sorted by absolute magnitude):")
coef = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
for feat, val in coef.items():
    print(f"  {feat:<20} {val:>12.2f}")

# ── Export results ────────────────────────────────────────────────────────────
results_path = 'us-housing-dataset/model_results.json'
results = {
    "metrics": {
        "r2":         round(r2, 4),
        "rmse":       round(rmse),
        "mae":        round(mae),
        "mape":       round(mape, 4),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_r2_std":  round(cv_r2_std, 4),
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
