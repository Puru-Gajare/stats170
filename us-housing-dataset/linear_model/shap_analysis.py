"""
shap_analysis.py
────────────────
Imports the trained OLS model pipeline from linear_model.py and uses
SHAP's LinearExplainer to explain the predicted price of a single house.

Usage (from repo root):
    python us-housing-dataset/linear_model/shap_analysis.py

    # Optionally pass a 0-based test-set index to inspect a specific house:
    python us-housing-dataset/linear_model/shap_analysis.py --index 42
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Ensure this file's directory is on sys.path so 'linear_model' can always be imported,
# regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))

# ── Import from refactored model module ───────────────────────────────────────
from linear_model import load_and_clean_data, train_model



def explain_house(artifacts: dict, house_index: int = 0) -> None:
    """
    Run SHAP LinearExplainer on a single house from the test set.

    Parameters
    ----------
    artifacts   : dict returned by train_model()
    house_index : 0-based row index into the test set
    """
    model          = artifacts["model"]
    scaler         = artifacts["scaler"]
    X_train_scaled = artifacts["X_train_scaled"]
    X_test_scaled  = artifacts["X_test_scaled"]
    X_test         = artifacts["X_test"]
    y_test         = artifacts["y_test"]
    features       = artifacts["features"]

    # ── Build SHAP explainer ──────────────────────────────────────────────────
    # Pass a named DataFrame as background so SHAP carries feature names through.
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    explainer = shap.LinearExplainer(model, X_train_scaled_df)

    # ── Select one house ──────────────────────────────────────────────────────
    n_test = X_test_scaled.shape[0]
    if house_index >= n_test:
        raise IndexError(f"house_index {house_index} is out of range (test set has {n_test} rows).")

    # Wrap the single house as a named DataFrame so SHAP uses real feature names.
    house_scaled_df = pd.DataFrame(
        X_test_scaled[house_index : house_index + 1],
        columns=features
    )

    # ── Compute SHAP values ───────────────────────────────────────────────────
    shap_values = explainer(house_scaled_df)   # returns an Explanation object

    # ── Print summary to terminal ─────────────────────────────────────────────
    actual_price    = y_test.iloc[house_index]
    predicted_price = model.predict(house_scaled_df)[0]
    base_value      = explainer.expected_value

    print("=" * 55)
    print(f"SHAP Explanation – Test-Set House #{house_index}")
    print("=" * 55)

    # Print the original (unscaled) feature values for context
    house_raw = X_test.iloc[house_index]
    print("\nRaw feature values:")
    for feat in features:
        print(f"  {feat:<20} {house_raw[feat]:>12.3f}")

    print(f"\nBase value (avg predicted price) : ${base_value:>12,.0f}")
    print(f"Model prediction                  : ${predicted_price:>12,.0f}")
    print(f"Actual price                      : ${actual_price:>12,.0f}")

    # SHAP attribution per feature
    sv = shap_values.values[0]
    contributions = pd.Series(sv, index=features).sort_values(key=abs, ascending=False)
    print("\nSHAP contributions (ranked by |impact|):")
    print(f"  {'Feature':<20} {'SHAP value':>12}  Direction")
    print("  " + "-" * 45)
    for feat, val in contributions.items():
        direction = "↑ pushes price UP" if val > 0 else "↓ pushes price DOWN"
        print(f"  {feat:<20} ${val:>+11,.0f}  {direction}")

    # ── Custom waterfall plot ─────────────────────────────────────────────────
    # shap.plots.waterfall hardcodes text positions and causes label overlap.
    # We build our own bar chart so we have full control over layout.

    sv   = shap_values.values[0]                        # SHAP values array
    order = np.argsort(np.abs(sv))                      # sort ascending by |impact|

    sorted_features = [features[i] for i in order]
    sorted_shap     = sv[order]
    sorted_raw      = [house_raw[feat] for feat in sorted_features]

    # Y-axis label: "feature_name = raw_value"
    y_labels = [f"{feat} = {val:.4g}" for feat, val in zip(sorted_features, sorted_raw)]
    y_pos    = np.arange(len(sorted_features))

    colors = ["#e8335d" if v >= 0 else "#4da6ff" for v in sorted_shap]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(y_pos, sorted_shap, color=colors, height=0.55, edgecolor="white", linewidth=0.5)

    # ── Annotate each bar with its dollar value ───────────────────────────────
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    pad = x_range * 0.012   # small gap between bar tip and text

    for bar, val in zip(bars, sorted_shap):
        w = bar.get_width()
        if val >= 0:
            ax.text(w + pad, bar.get_y() + bar.get_height() / 2,
                    f"+${val:,.0f}", va="center", ha="left",
                    fontsize=9, color="#c0254a")
        else:
            ax.text(w - pad, bar.get_y() + bar.get_height() / 2,
                    f"-${abs(val):,.0f}", va="center", ha="right",
                    fontsize=9, color="#1a6fba")

    # ── Y-axis labels (feature = value) on the left ───────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)

    # ── Reference line at zero ────────────────────────────────────────────────
    ax.axvline(0, color="black", linewidth=0.8, zorder=3)

    # ── Labels and title ──────────────────────────────────────────────────────
    ax.set_xlabel("SHAP value  ($ impact on predicted price)", fontsize=11)
    ax.set_title(
        f"SHAP Explanation – House #{house_index}\n"
        f"Base (avg): ${base_value:,.0f}   →   Predicted: ${predicted_price:,.0f}   "
        f"  |   Actual: ${actual_price:,.0f}",
        fontsize=12, pad=14
    )

    # Extra right margin so bar annotations don't get clipped
    cur_xmax = ax.get_xlim()[1]
    ax.set_xlim(right=cur_xmax * 1.18)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved shap_waterfall.png")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explanation for a single house.")
    parser.add_argument(
        "--index", type=int, default=0,
        help="0-based index of the house in the test set to explain (default: 0)"
    )
    args = parser.parse_args()

    print("Loading data and training model...")
    df        = load_and_clean_data()
    artifacts = train_model(df)

    explain_house(artifacts, house_index=args.index)
