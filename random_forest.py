import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ==============================
# Settings
# ==============================
DEFAULT_DATA_PATH = "us-housing-dataset/ca_2022_sold.pkl"
OUTPUT_JSON = "rf_log_tuned_results.json"

# Change this to False after you find good parameters.
# True = tune many models. False = train one model only.
USE_RANDOM_SEARCH = True

# Smaller search so it runs faster than the earlier 20 x 3 = 60 fits.
N_ITER_SEARCH = 5
CV_FOLDS = 2


# ==============================
# Load data
# ==============================
def load_data() -> pd.DataFrame:
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find pkl file at: {data_path}\n"
            "Put the pkl file in the same path or pass the path as a command-line argument."
        )

    print(f"Loading data from: {data_path}")
    return pd.read_pickle(data_path)


# ==============================
# Main pipeline
# ==============================
def main():
    df = load_data()
    df = df.drop_duplicates().copy()

    # Columns needed for the base model
    needed_cols = [
        "price", "bed", "bath", "acre_lot", "city",
        "state", "zip_code", "house_size"
    ]

    # Optional influencing-factor columns from new pkl file
    possible_extra_features = [
        "school_count",
        "avg_school_dist_km",
        "min_school_dist_km",
        "library_count",
        "avg_library_dist_km",
        "min_library_dist_km",
    ]

    extra_features = [col for col in possible_extra_features if col in df.columns]

    print("Extra influencing-factor features found:")
    print(extra_features if extra_features else "None")

    # Convert numeric columns safely.
    # This fixes the error: AttributeError: 'float' object has no attribute 'log1p'
    numeric_cols = [
        "price", "bed", "bath", "acre_lot", "house_size"
    ] + extra_features

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing the required base columns
    df = df.dropna(subset=needed_cols).copy()

    # Fill optional influencing-factor missing values
    # Counts become 0; distance columns become the median distance.
    for col in extra_features:
        if "count" in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())

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

    # ==============================
    # Log-price target encoding
    # ==============================
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["log_price"] = np.log1p(train_df["price"].astype(float))
    global_mean_log_price = train_df["log_price"].mean()

    zip_avg_log = train_df.groupby("zip_code")["log_price"].mean()
    city_avg_log = train_df.groupby("city")["log_price"].mean()

    train_df["zip_avg_log_price"] = train_df["zip_code"].map(zip_avg_log).fillna(global_mean_log_price)
    test_df["zip_avg_log_price"] = test_df["zip_code"].map(zip_avg_log).fillna(global_mean_log_price)

    train_df["city_avg_log_price"] = train_df["city"].map(city_avg_log).fillna(global_mean_log_price)
    test_df["city_avg_log_price"] = test_df["city"].map(city_avg_log).fillna(global_mean_log_price)

    # ==============================
    # Engineered features
    # ==============================
    for data in [train_df, test_df]:
        data["rooms_total"] = data["bed"] + data["bath"]
        data["bath_per_bed"] = data["bath"] / data["bed"]
        data["sqft_per_bed"] = data["house_size"] / data["bed"]
        data["log_house_size"] = np.log1p(data["house_size"].astype(float))
        data["log_acre_lot"] = np.log1p(data["acre_lot"].astype(float))

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
        "zip_avg_log_price",
        "city_avg_log_price",
    ] + extra_features

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = np.log1p(train_df["price"].astype(float))
    y_test_actual = test_df["price"].astype(float)

    # ==============================
    # Model training
    # ==============================
    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    if USE_RANDOM_SEARCH:
        print("\nRunning fast RandomizedSearchCV...")
        print(f"This will train {N_ITER_SEARCH} candidates x {CV_FOLDS} folds = {N_ITER_SEARCH * CV_FOLDS} models.")

        param_distributions = {
            "n_estimators": [300, 400, 500],
            "max_depth": [30, 40, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
            "bootstrap": [True]
        }

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=N_ITER_SEARCH,
            scoring="r2",
            cv=CV_FOLDS,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
    else:
        print("\nTraining one Random Forest model only...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=40,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()
        best_cv_score = None

    # ==============================
    # Prediction and metrics
    # ==============================
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    print("\n" + "=" * 55)
    print("RANDOM FOREST — LOG TARGET ENCODING + TUNING")
    print("=" * 55)
    print(f"Rows used : {len(df):,}")
    print(f"Features  : {len(features)}")
    print(f"R²        : {r2:.4f}")
    print(f"RMSE      : ${rmse:,.0f}")
    print(f"MAE       : ${mae:,.0f}")

    if USE_RANDOM_SEARCH:
        print("\nBest CV R²:", round(best_cv_score, 4))
        print("Best parameters:")
        print(best_params)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importance)

        # ==============================
    # SHAP graphs
    # ==============================
    try:
        import shap

        OUTPUT_DIR = "rf_shap_graphs"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        SHAP_SAMPLE_SIZE = 300

        X_shap = X_test.sample(
            n=min(SHAP_SAMPLE_SIZE, len(X_test)),
            random_state=42
        )

        print("\nCreating SHAP graphs on sample...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # SHAP beeswarm plot
        shap.summary_plot(
            shap_values,
            X_shap,
            show=False
        )
        plt.title("Random Forest SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/rf_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()

        # SHAP bar importance plot
        shap.summary_plot(
            shap_values,
            X_shap,
            plot_type="bar",
            show=False
        )
        plt.title("Random Forest SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/rf_shap_bar_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved SHAP graphs to: {OUTPUT_DIR}")

    except ImportError:
        print("\nSHAP is not installed. Run this:")
        print("pip install shap")
    except Exception as e:
        print(f"\nSHAP graph skipped because of error: {e}")
    
    # Save results
    results = {
        "model_name": "Random Forest with Log Target Encoding and Tuning",
        "rows_used": int(len(df)),
        "features_used": features,
        "extra_influencing_factor_features": extra_features,
        "use_random_search": USE_RANDOM_SEARCH,
        "best_cv_r2": None if best_cv_score is None else round(float(best_cv_score), 4),
        "best_params": best_params,
        "metrics": {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse)),
            "mae": round(float(mae))
        },
        "feature_importance": {
            feature: round(float(value), 4)
            for feature, value in importance.items()
        },
        "actual": [round(float(v)) for v in y_test_actual.values[:1000].tolist()],
        "predicted": [round(float(v)) for v in y_pred[:1000].tolist()],
        "residuals": [
            round(float(a - p))
            for a, p in zip(y_test_actual.values[:1000].tolist(), y_pred[:1000].tolist())
        ]
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
