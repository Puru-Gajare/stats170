import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load updated cleaned data
df = pd.read_pickle("us-housing-dataset/ca_2022_sold.pkl")

# Basic cleaning
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

# Split first to avoid data leakage
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Location target encoding using TRAINING DATA ONLY
global_mean_price = train_df["price"].mean()

zip_avg = train_df.groupby("zip_code")["price"].mean()
city_avg = train_df.groupby("city")["price"].mean()

train_df["zip_avg_price"] = train_df["zip_code"].map(zip_avg).fillna(global_mean_price)
test_df["zip_avg_price"] = test_df["zip_code"].map(zip_avg).fillna(global_mean_price)

train_df["city_avg_price"] = train_df["city"].map(city_avg).fillna(global_mean_price)
test_df["city_avg_price"] = test_df["city"].map(city_avg).fillna(global_mean_price)

# Extra engineered features
for data in [train_df, test_df]:
    data["rooms_total"] = data["bed"] + data["bath"]
    data["bath_per_bed"] = data["bath"] / data["bed"]
    data["sqft_per_bed"] = data["house_size"] / data["bed"]
    data["log_house_size"] = np.log1p(data["house_size"])
    data["log_acre_lot"] = np.log1p(data["acre_lot"])

# Features
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
    "zip_avg_price",
    "city_avg_price"
]

X_train = train_df[features]
X_test = test_df[features]

# Log-transform target
y_train = np.log1p(train_df["price"])
y_test_actual = test_df["price"]

# Random Forest model
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict back to dollars
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Metrics
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("=" * 45)
print("IMPROVED RANDOM FOREST — CA 2022 SOLD HOMES")
print("=" * 45)
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
    "model_name": "Improved Random Forest CA 2022 Sold Homes",
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

with open("rf_ca_2022_improved_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved rf_ca_2022_improved_results.json")