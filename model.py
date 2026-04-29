import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv("housing.csv")
df = df.dropna()
df = df.drop_duplicates()
df["ocean_encoded"] = df["ocean_proximity"].astype("category").cat.codes
df = df.drop(columns=["ocean_proximity"])

# ── Features / target ─────────────────────────────────────────────────────────
# latitude & longitude removed — large unstable coefficients, not linearly
# interpretable; ocean_encoded captures location more meaningfully
df = df.drop(columns=["latitude", "longitude"])
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=" * 40)
print("LINEAR REGRESSION RESULTS")
print("=" * 40)
print(f"R²   : {r2:.4f}")
print(f"RMSE : ${rmse:,.0f}")
print(f"MAE  : ${mae:,.0f}")
print()
print("Intercept :", round(model.intercept_))
print()
print("Coefficients:")
coef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
for feat, val in coef.items():
    print(f"  {feat:<25} {val:>12.2f}")

# ── Export results for visualize.py ──────────────────────────────────────────
results = {
    "metrics": {"r2": round(r2, 4), "rmse": round(rmse), "mae": round(mae)},
    "intercept": round(model.intercept_),
    "coefficients": {feat: round(val, 2) for feat, val in coef.items()},
    "actual":  [round(v) for v in y_test.values[:300].tolist()],
    "predicted": [round(max(0, v)) for v in y_pred[:300].tolist()],
    "residuals": [round(a - p) for a, p in zip(
        y_test.values[:300].tolist(), y_pred[:300].tolist()
    )],
}

with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model_results.json for visualize.py")