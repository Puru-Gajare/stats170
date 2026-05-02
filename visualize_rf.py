import json
import numpy as np
import matplotlib.pyplot as plt

# Load Random Forest results
with open("rf_ca_2022_improved_results.json") as f:
    res = json.load(f)

metrics = res["metrics"]
actual = np.array(res["actual"])
predicted = np.array(res["predicted"])
feature_importance = res["feature_importance"]

# ---------- Plot 1: Actual vs Predicted ----------
plt.figure(figsize=(7, 6))

plt.scatter(actual / 1_000_000, predicted / 1_000_000, alpha=0.35, s=12)

max_val = max(actual.max(), predicted.max()) / 1_000_000
plt.plot([0, max_val], [0, max_val], linestyle="--", label="Perfect prediction")

plt.title(
    f"Actual vs. Predicted — Random Forest\n"
    f"R² = {metrics['r2']:.4f} | RMSE = ${metrics['rmse']:,}"
)
plt.xlabel("Actual Price ($M)")
plt.ylabel("Predicted Price ($M)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("random_forest_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------- Plot 2: Feature Importance ----------
features = list(feature_importance.keys())
values = list(feature_importance.values())

plt.figure(figsize=(8, 6))
plt.barh(features[::-1], values[::-1])

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)

plt.savefig("random_forest_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved random_forest_actual_vs_predicted.png")
print("Saved random_forest_feature_importance.png")