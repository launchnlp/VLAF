"""
Script for creating publication-quality grouped bar plots
comparing base and instruct models across different data splits.
"""

import numpy as np
import matplotlib.pyplot as plt

# Data
data_splits = ['Authority', 'Care', 'Fairness', 'Loyalty', 'Sanctity']
instruct = [0.1335, 0.37725, 0.36925, 0.298, 0.247]
base = [0.05325, 0.10625, 0.10875, 0.09475, 0.07225]

# Convert to percentages
instruct_pct = [val * 100 for val in instruct]
base_pct = [val * 100 for val in base]

# Set up the plot with academic styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 6))

# Set width of bars and positions
x = np.arange(len(data_splits))
width = 0.35

# Define pastel colors and their darker borders
color_base_pastel = '#A8DADC'      # Pastel blue
color_base_dark = '#457B9D'         # Dark blue border
color_instruct_pastel = '#F1C0E8'   # Pastel pink
color_instruct_dark = '#9D4EDD'     # Dark purple border

# Create bars with hatching
bars1 = ax.bar(x - width/2, base_pct, width, 
               label='Base', 
               color=color_base_pastel,
               edgecolor=color_base_dark,
               linewidth=2.0,
               alpha=0.95,
               hatch='//')

bars2 = ax.bar(x + width/2, instruct_pct, width,
               label='Instruct',
               color=color_instruct_pastel,
               edgecolor=color_instruct_dark,
               linewidth=2.0,
               alpha=0.95,
               hatch='\\\\')

# Customize the plot
# ax.set_xlabel('Data Split', fontsize=18, weight='medium')
ax.set_ylabel('Compliance Gap (x100)', fontsize=22, weight='medium')
# ax.set_title('Model Performance Across Data Splits', fontsize=16, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(data_splits, fontsize=22)
ax.tick_params(axis='both', labelsize=20, width=1.2, length=6)

# Add value labels on top of bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom',
                fontsize=14, weight='medium')

add_value_labels(bars1)
add_value_labels(bars2)

# Customize legend
legend = ax.legend(loc='upper left', 
                   fontsize=18,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   borderpad=1,
                   labelspacing=0.8)
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.0)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Customize grid
ax.grid(True, axis='x', alpha=0.2, linewidth=0.8, linestyle='--')
ax.set_axisbelow(True)

# Set y-axis to start at 0 for better visual comparison
ax.set_ylim(0, max(max(instruct_pct), max(base_pct)) * 1.15)

plt.tight_layout()

# Save the figure
plt.savefig('plotting/post_training_effect.pdf', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()

print("Plot saved as 'plotting/post_training_effect.pdf'")