import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading model results...")

with open("us-housing-dataset/model_results.json") as f:
    lr_data = json.load(f)

with open("rf_ca_2022_improved_results.json") as f:
    rf_data = json.load(f)

with open("xg_outputs/xg_ca_2022_results.json") as f:
    xg_data = json.load(f)

# Structure the loaded data
models = {
    "Linear Regression": {
        "r2": lr_data["metrics"]["r2"],
        "rmse": lr_data["metrics"]["rmse"],
        "mae": lr_data["metrics"]["mae"],
        "actual": np.array(lr_data["actual"]),
        "predicted": np.array(lr_data["predicted"]),
        "residuals": np.array(lr_data["residuals"]),
        "color": "#94A3B8"  # Slate Blue
    },
    "Random Forest": {
        "r2": rf_data["metrics"]["r2"],
        "rmse": rf_data["metrics"]["rmse"],
        "mae": rf_data["metrics"]["mae"],
        "actual": np.array(rf_data["actual"]),
        "predicted": np.array(rf_data["predicted"]),
        "residuals": np.array(rf_data["residuals"]),
        "color": "#10B981"  # Emerald Green
    },
    "XGBoost": {
        "r2": xg_data["metrics"]["r2"],
        "rmse": xg_data["metrics"]["rmse"],
        "mae": xg_data["metrics"]["mae"],
        "actual": np.array(xg_data["actual"]),
        "predicted": np.array(xg_data["predicted"]),
        "residuals": np.array(xg_data["residuals"]),
        "color": "#3B82F6"  # Royal Blue
    }
}

# ── 2. PLOT STYLE & SETUP ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.facecolor":   "#F8F8F6",
    "figure.facecolor": "#F8F8F6",
    "text.color":       "#2C2C2A",
    "axes.labelcolor":  "#2C2C2A",
    "xtick.color":      "#888780",
    "ytick.color":      "#888780",
})

fig = plt.figure(figsize=(18, 5.5))
fig.suptitle("California Housing 2022 — Model Comparison Dashboard", 
             fontsize=18, fontweight="bold", y=1.02, color="#1E293B")

# Create a 1x3 grid
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.48)

# ── 3. PANEL 1: METRIC BARS (R² and MAE) ──────────────────────────────────────
ax_metrics = fig.add_subplot(gs[0, 0])
model_names = list(models.keys())
colors = [models[m]["color"] for m in model_names]

# Create sub-axes for twinx plotting
r2_vals = [models[m]["r2"] for m in model_names]
mae_vals = [models[m]["mae"] / 1000.0 for m in model_names]  # in thousands

x = np.arange(len(model_names))
width = 0.35

rects1 = ax_metrics.bar(x - width/2, r2_vals, width, label="R² Score", color=colors, alpha=0.9)
ax_metrics.set_ylabel("R² Score", fontsize=11, fontweight="bold", color="#1E293B")
ax_metrics.set_ylim(0, 1.0)
ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(model_names, fontsize=10)
ax_metrics.set_title("Model Accuracy & Error Comparison", fontsize=12, fontweight="bold", pad=12)

# Add values on top of bars
for rect in rects1:
    height = rect.get_height()
    ax_metrics.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight="bold")

# Add MAE as a second bar chart grouped alongside
ax_mae = ax_metrics.twinx()
rects2 = ax_mae.bar(x + width/2, mae_vals, width, label="MAE ($k)", color=colors, hatch="//", alpha=0.6)
ax_mae.set_ylabel("Mean Absolute Error ($k)", fontsize=11, fontweight="bold", color="#1E293B")
ax_mae.set_ylim(0, max(mae_vals) * 1.25)
ax_mae.grid(False)  # Avoid double grid lines

for rect in rects2:
    height = rect.get_height()
    ax_mae.annotate(f"${height:,.0f}k",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight="bold")

# ── 4. PANEL 2: MAE BY HOME PRICE RANGE ───────────────────────────────────────
ax_resid = fig.add_subplot(gs[0, 1])

# Define price ranges and bins
bins = [0, 500000, 1000000, 2000000, np.inf]
bin_labels = ["<$500k", "$500k-$1M", "$1M-$2M", ">$2M"]

x_bins = np.arange(len(bin_labels))
bar_width = 0.25

for idx, (model_name, info) in enumerate(models.items()):
    act = info["actual"]
    pred = info["predicted"]
    abs_err_k = np.abs(act - pred) / 1000.0  # in thousands
    
    maes = []
    for j in range(len(bins)-1):
        mask = (act >= bins[j]) & (act < bins[j+1])
        if np.sum(mask) > 0:
            maes.append(np.mean(abs_err_k[mask]))
        else:
            maes.append(0.0)
            
    # Shift the bars so they plot side-by-side
    ax_resid.bar(x_bins + (idx - 1) * bar_width, maes, bar_width,
                 label=model_name, color=info["color"], alpha=0.85)

ax_resid.set_xticks(x_bins)
ax_resid.set_xticklabels(bin_labels, fontsize=10)
ax_resid.set_xlabel("Home Price Range", fontsize=11)
ax_resid.set_ylabel("Mean Absolute Error ($k)", fontsize=11)
ax_resid.set_title("MAE by Home Price Range", fontsize=12, fontweight="bold", pad=12)
ax_resid.legend(fontsize=9, loc="upper left")

# ── 5. PANEL 3: CUMULATIVE ABSOLUTE ERROR ─────────────────────────────────────
ax_cae = fig.add_subplot(gs[0, 2])
for model_name, info in models.items():
    abs_err_k = np.abs(info["actual"] - info["predicted"]) / 1000.0
    sorted_err = np.sort(abs_err_k)
    y_vals = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100.0
    ax_cae.plot(sorted_err, y_vals, color=info["color"], linewidth=2, label=model_name)

ax_cae.set_xlim(0, 500)  # Focus on error bounds up to $500k
ax_cae.set_ylim(0, 100)
ax_cae.set_xlabel("Absolute Error Bound ($K)", fontsize=11)
ax_cae.set_ylabel("% of Houses Within Bound", fontsize=11)
ax_cae.set_title("Prediction Accuracy Curve", fontsize=12, fontweight="bold", pad=12)
ax_cae.legend(fontsize=9, loc="lower right")

# Save the unified dashboard to root
output_img = "model_comparison_dashboard.png"
plt.savefig(output_img, dpi=180, bbox_inches="tight")
print(f"Saved comparison dashboard to {output_img}")
plt.close()
