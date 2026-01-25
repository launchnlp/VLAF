import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# Data
labels = [
    "Goal Guarding",
    "Self-preservation",
    "Influence Seeking",
    "Resource Acquisition",
    "No Consequence",
]
values = [30.5, 22.3, 21.5, 27.2, 4.13]

x = np.arange(len(labels))

# Base pastel color
base_color = "#AEC6CF"  # soft pastel blue
hatch_pattern = "o"    # consistent hatching

# Function to darken the color for border
def darken_color(color, factor=0.6):
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)

border_color = darken_color(base_color, 0.5)

# Figure
fig, ax = plt.subplots(figsize=(14, 6))

# Bars
bars = ax.bar(
    x,
    values,
    width=0.6,
    color=base_color,
    edgecolor=border_color,
    hatch=hatch_pattern
)

# Axis labels
ax.set_ylabel("Compliance Gap (x100)", fontsize=18)
# ax.set_xlabel("Instrumental Goal", fontsize=18)

# X ticks
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=19)

# Y-axis limits for breathing room
ax.set_ylim(0, max(values) * 1.15)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.8,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

# Clean up spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Subtle grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig("plotting/af_instrumental_goal.pdf", format="pdf", dpi=300)
