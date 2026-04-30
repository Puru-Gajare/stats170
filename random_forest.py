import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned pkl file
df = pd.read_pickle("us-housing-dataset/ca_2022_sold.pkl")

# Basic cleaning
df = df.dropna(subset=["price", "bed", "bath", "acre_lot", "house_size", "zip_code"])
df = df.drop_duplicates()

# Make zip code categorical/numeric encoding
df["zip_code"] = df["zip_code"].astype(str)
df["zip_encoded"] = df["zip_code"].astype("category").cat.codes

# Features and target
features = [
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "zip_encoded"
]

X = df[features]
y = df["price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("RANDOM FOREST — CA 2022 SOLD HOMES")
print("=" * 40)
print(f"Rows used: {len(df):,}")
print(f"R²   : {r2:.4f}")
print(f"RMSE : ${rmse:,.0f}")
print(f"MAE  : ${mae:,.0f}")

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Save results
results = {
    "model_name": "Random Forest CA 2022 Sold Homes",
    "rows_used": len(df),
    "metrics": {
        "r2": round(r2, 4),
        "rmse": round(rmse),
        "mae": round(mae)
    },
    "feature_importance": {
        feature: round(value, 4)
        for feature, value in importance.items()
    }
}

with open("rf_ca_2022_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved rf_ca_2022_results.json")