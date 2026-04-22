import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from scipy import stats

# Data setup
data = {
    "Monitoring Probability": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "olmo2-7b-instruct": [0.5652, 0.5392, 0.5512, 0.5616, 0.5604, 0.5876],
    "olmo2-13b-instruct": [0.5568, 0.5468, 0.5512, 0.5412, 0.5512, 0.5556],
    "qwen2.5-7b-instruct": [0.15, 0.1472, 0.144, 0.1544, 0.1496, 0.1568],
    "qwen2.5-14b-instruct": [0.592, 0.6064, 0.6104, 0.618, 0.6088, 0.626],
    "qwen3-8b": [0.5308, 0.538, 0.5424, 0.5508, 0.5536, 0.5796],
    "qwen3-14b": [0.7692, 0.7756, 0.7836, 0.7912, 0.7928, 0.8056]
}

df = pd.DataFrame(data)
plot_df = df.copy()

# Subtract baseline (x=0) and scale by 100
for col in df.columns[1:]:
    baseline = df.loc[0, col]
    plot_df[col] = (df[col] - baseline) * 100

x_vals = plot_df["Monitoring Probability"].values

# Compute linear regression slopes + p-values for each model
slopes = {}
intercepts = {}
pvalues = {}
for col in plot_df.columns[1:]:
    y = plot_df[col].values
    result = stats.linregress(x_vals, y)
    slopes[col] = result.slope
    intercepts[col] = result.intercept
    pvalues[col] = result.pvalue

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# High-fidelity academic styling
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2
})

fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
markers = ['o', 's', 'D', '^', 'v', 'p', 'h']

for i, model in enumerate(plot_df.columns[1:]):
    is_qwen3 = "qwen3" in model
    lw = 9.0 if is_qwen3 else 2.0
    ms = 14 if is_qwen3 else 9
    alpha = 1.0 if is_qwen3 else 0.65
    color = colors[i]
    slope = slopes[model]

    label = f"{model}  (slope = {slope:+.2f})"

    # Data line
    ax.plot(x_vals, plot_df[model],
            label=label, marker=markers[i], markersize=ms,
            linewidth=lw, color=color, alpha=alpha,
            markeredgecolor='white', markeredgewidth=1.2)

# Academic Grid Format (Horizontal Only)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray', linewidth=0.5)
ax.xaxis.grid(False)

# Labels and Ticks
ax.set_xlabel("Monitoring Probability", fontsize=22)
ax.set_ylabel(r"$\Delta$ Compliance Rate $\times 100$", fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=18, direction='out')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# Reference Line at zero
ax.axhline(0, color='black', linewidth=1.2, alpha=0.8)

# Legend with slope values embedded in labels
legend = ax.legend(loc='upper left', fontsize=18, frameon=True,
                   fancybox=False, edgecolor='black', framealpha=1.0,
                   title="Model  (linear slope)", title_fontsize=17)
legend.get_frame().set_linewidth(0.5)

# Tufte-style spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("plotting/imperfect_oversight.pdf", format="pdf", dpi=300)
plt.savefig("plotting/imperfect_oversight.png", dpi=150, bbox_inches='tight')
