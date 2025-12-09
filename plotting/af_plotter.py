import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data for all models
data = {
    "Model": [
        "olmo2-32b-base",
        "olmo2-32b-sft",
        "olmo2-32b-dpo",
        "olmo2-32b-instruct",
    ],
    "Authority_Free": [41.5, 12, 2.5, 9.5],
    "Authority_Paid": [21.5, 4, 0.5, 3.5],
    "Care_Free": [76.7, 25.8, 20.8, 44.2],
    "Care_Paid": [50.8, 12.5, 4.2, 7.5],
    "Fairness_Free": [53.5, 22, 44, 55.5],
    "Fairness_Paid": [25, 11, 14, 18.5],
    "Loyalty_Free": [52.5, 20.5, 12, 19],
    "Loyalty_Paid": [30, 5, 3.5, 3.5],
    "Sanctity_Free": [42, 12, 8, 17],
    "Sanctity_Paid": [19.5, 5.5, 5.5, 4]
}

df = pd.DataFrame(data)

# Compute compliance gaps
# values = ["Authority", "Care", "Fairness", "Loyalty", "Sanctity"]
values = ["Authority", "Loyalty", "Sanctity"]  # Only plot these two for clarity
for value in values:
    df[f"{value}_Gap"] = df[f"{value}_Free"] - df[f"{value}_Paid"]

# Compute average compliance gap
df["Average_Gap"] = df[[f"{v}_Gap" for v in values]].mean(axis=1)

# --- Plot 1: Average compliance gap ---
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Average_Gap"], color="skyblue")
plt.title("Average Compliance Gap Across All Moral Foundations")
plt.ylabel("Compliance Gap (Free - Paid)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("average_compliance_gap.png")

# --- Plot 2: Grouped bar plot for each value ---
x = np.arange(len(df["Model"]))  # label locations
width = 0.15  # width of each bar

plt.figure(figsize=(10, 6))
for i, value in enumerate(values):
    plt.bar(x + i * width, df[f"{value}_Gap"], width, label=value)

plt.xticks(x + width * 2, df["Model"], rotation=30, ha="right")
plt.ylabel("Compliance Gap (Free - Paid)")
plt.title("Compliance Gap for Authority, Loyalty, and Sanctity by OlMo2 Models")
plt.legend(title="Moral Foundation")
plt.tight_layout()
plt.savefig("compliance_gap_per_value.png")
