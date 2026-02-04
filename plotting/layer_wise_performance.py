import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the provided image
layers = [5, 10, 15, 20, 25, 30]
consistency_avg = [0.25, 0.213, 0.199, 0.276, 0.235, 0.204]
redirection_avg = [0.259, 0.282, 0.219, 0.271, 0.259, 0.224]
consistency_avg = [val * 100 for val in consistency_avg]
redirection_avg = [val * 100 for val in redirection_avg]

# Set up academic styling
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with distinct styles and markers
ax.plot(layers, consistency_avg, label='Consistency (Avg)', 
        color='#457B9D', marker='o', markersize=9, linewidth=2.5, 
        linestyle='-', markerfacecolor='white', markeredgewidth=2)

ax.plot(layers, redirection_avg, label='Redirection (Avg)', 
        color='#9D4EDD', marker='s', markersize=9, linewidth=2.5, 
        linestyle='--', markerfacecolor='white', markeredgewidth=2)

# Customizing the axes
ax.set_xlabel('Layer', fontsize=18, fontweight='medium')
ax.set_ylabel('Average Compliance Gap (x100)', fontsize=18, fontweight='medium')
ax.set_xticks(layers)
ax.tick_params(axis='both', labelsize=14, width=1.2, length=6)

# Adding values above points for clarity (optional)
for i, txt in enumerate(consistency_avg):
    ax.annotate(f'{txt:.2f}', (layers[i], consistency_avg[i]), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

for i, txt in enumerate(redirection_avg):
    ax.annotate(f'{txt:.2f}', (layers[i], redirection_avg[i]), 
                textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10)

# Legend and Grid
ax.legend(loc='best', fontsize=14, frameon=True, shadow=True, borderpad=1)
ax.grid(True, linestyle='--', alpha=0.4)

# Clean aesthetics
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plotting/layer_wise_performance.pdf', dpi=400)
plt.show()