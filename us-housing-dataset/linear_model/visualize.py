import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Load results from model_results.json ──────────────────────────────────────
# Path is resolved relative to this script's directory.
_HERE = Path(__file__).parent
results_path = _HERE / 'model_results.json'
with open(results_path) as f:
    res = json.load(f)

metrics     = res["metrics"]
coefs       = res["coefficients"]
actual      = np.array(res["actual"])
predicted   = np.array(res["predicted"])
residuals   = np.array(res["residuals"])

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE        = "#185FA5"
BLUE_LIGHT  = "#85B7EB"
RED         = "#A32D2D"
GRAY        = "#888780"
BG          = "#F8F8F6"
TEXT        = "#2C2C2A"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.facecolor":   BG,
    "figure.facecolor": BG,
    "text.color":       TEXT,
    "axes.labelcolor":  TEXT,
    "xtick.color":      GRAY,
    "ytick.color":      GRAY,
})

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Linear Regression — CA 2022 Housing Analysis",
             fontsize=15, fontweight="bold", y=0.98, color=TEXT)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── 1. Metric cards (text axes) ───────────────────────────────────────────────
ax_metrics = fig.add_subplot(gs[0, 0])
ax_metrics.axis("off")
cards = [
    ("R² Score",  f"{metrics['r2']:.3f}",  f"explains {metrics['r2']*100:.1f}% of variance"),
    ("RMSE",      f"${metrics['rmse']:,}",  "root mean squared error"),
    ("MAE",       f"${metrics['mae']:,}",   "mean absolute error"),
]
for i, (label, value, sub) in enumerate(cards):
    y_pos = 0.85 - i * 0.30
    ax_metrics.text(0.05, y_pos,       label, fontsize=10, color=GRAY,   transform=ax_metrics.transAxes)
    ax_metrics.text(0.05, y_pos - 0.09, value, fontsize=18, fontweight="bold", color=BLUE, transform=ax_metrics.transAxes)
    ax_metrics.text(0.05, y_pos - 0.17, sub,  fontsize=9,  color=GRAY,   transform=ax_metrics.transAxes)
ax_metrics.set_title("Model metrics", fontsize=11, loc="left", pad=8, color=TEXT)

# ── 2. Coefficients bar chart ─────────────────────────────────────────────────
ax_coef = fig.add_subplot(gs[0, 1])
features = list(coefs.keys())
values   = list(coefs.values())
colors   = [BLUE if v > 0 else RED for v in values]

bars = ax_coef.barh(features[::-1], values[::-1], color=colors[::-1],
                    height=0.6, zorder=2)
ax_coef.axvline(0, color=GRAY, linewidth=0.8, zorder=1)
ax_coef.set_xlabel("Coefficient value ($)", fontsize=9)
ax_coef.set_title("Feature coefficients", fontsize=11, loc="left", pad=8, color=TEXT)
ax_coef.tick_params(axis="y", labelsize=9)
ax_coef.tick_params(axis="x", labelsize=8)
ax_coef.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax_coef.grid(axis="x", color="#CCCCCC", linewidth=0.5, zorder=0)

# ── 3. Actual vs predicted scatter ────────────────────────────────────────────
ax_scatter = fig.add_subplot(gs[1, 0])
ax_scatter.scatter(actual, predicted, alpha=0.35, s=12, color=BLUE, zorder=2)
# Set limit based on actual data
max_val = max(actual.max(), predicted.max())
lim = (0, max_val * 1.1)
ax_scatter.plot(lim, lim, color=RED, linewidth=1.2, linestyle="--",
                label="Perfect fit", zorder=3)
ax_scatter.set_xlim(*lim)
ax_scatter.set_ylim(*lim)
ax_scatter.set_xlabel("Actual price ($)", fontsize=9)
ax_scatter.set_ylabel("Predicted price ($)", fontsize=9)
ax_scatter.set_title("Actual vs predicted", fontsize=11, loc="left", pad=8, color=TEXT)
ax_scatter.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax_scatter.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax_scatter.tick_params(labelsize=8)
ax_scatter.legend(fontsize=8)

# ── 4. Residuals histogram ────────────────────────────────────────────────────
ax_resid = fig.add_subplot(gs[1, 1])
ax_resid.hist(residuals, bins=35, color=BLUE_LIGHT, edgecolor=BG, linewidth=0.4, zorder=2)
ax_resid.axvline(0, color=RED, linewidth=1.2, linestyle="--", label="Zero error", zorder=3)
ax_resid.axvline(residuals.mean(), color=BLUE, linewidth=1.2, linestyle=":",
                 label=f"Mean: ${residuals.mean():,.0f}", zorder=3)
ax_resid.set_xlabel("Residual (actual − predicted) $", fontsize=9)
ax_resid.set_ylabel("Count", fontsize=9)
ax_resid.set_title("Residuals distribution", fontsize=11, loc="left", pad=8, color=TEXT)
ax_resid.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax_resid.tick_params(labelsize=8)
ax_resid.legend(fontsize=8)
ax_resid.grid(axis="y", color="#CCCCCC", linewidth=0.5, zorder=0)

output_img = _HERE / 'regression_plots.png'
plt.savefig(output_img, dpi=150, bbox_inches="tight")
print(f"Saved {output_img}")
# plt.show() # Disabled for headless run
