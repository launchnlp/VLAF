import numpy as np
import matplotlib.pyplot as plt

# Data from provided images
scenarios = ['Vanilla', 'ACTADD (Sorry-Bench)', 'LAT (Sorry-Bench)', 'ACTADD (WMDP)', 'LAT (WMDP)']
models = ['olmo2-7b-instruct', 'olmo2-13b-instruct', 'qwen3-8b', 'qwen3-14b']

# Accuracy Data (%)
accuracy_data = {
    'Vanilla': [64.25, 82.5, 95.0, 98.50],
    'ACTADD (Sorry-Bench)': [56.25, 79, 90.00, 98.00],
    'LAT (Sorry-Bench)': [57.0, 72.75, 94.50, 97.00],
    'ACTADD (WMDP)': [58.50, 80.75, 92.00, 98.25],
    'LAT (WMDP)': [55.50, 79.50, 94.75, 97.25]
}

# Drop Data (%) for annotations
drop_data = {
    'Vanilla': [0, 0, 0, 0],
    'ACTADD (Sorry-Bench)': [-8.0, -3.5, -5.0, -0.5],
    'LAT (Sorry-Bench)': [-7.25, -9.75, -0.5, -1.5],
    'ACTADD (WMDP)': [-5.75, -1.75, -3.0, -0.25],
    'LAT (WMDP)': [-8.75, -3.0, -0.25, -1.25]
}

# Set up the plot with academic styling
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(17, 6))

# Positions and bar width
x = np.arange(len(models))
width = 0.16  # Narrower width to fit 5 bars per group

# Color and Hatch mapping
# Using your previous palette and expanding it for 5 categories
colors = ['#A8DADC', '#F1C0E8', '#90BE6D', '#F9C74F', '#F94144']
borders = ['#457B9D', '#9D4EDD', '#4D908E', '#D4A017', '#900C3F']
hatches = ['//', '\\\\', '||', '--', 'xx']

# Create bars
for i, scenario in enumerate(scenarios):
    pos = x + (i - 2) * width
    bars = ax.bar(pos, accuracy_data[scenario], width,
                  label=scenario,
                  color=colors[i],
                  edgecolor=borders[i],
                  linewidth=1.8,
                  alpha=0.9,
                  hatch=hatches[i])

    # Add drop percentage labels on top of bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        drop_val = drop_data[scenario][j]
        
        # Only show labels for non-zero drops (non-vanilla)
        if drop_val != 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{drop_val:.1f}%',
                    ha='center', va='bottom',
                    fontsize=14, weight='bold', color=borders[i])

# Customize the plot
ax.set_ylabel('Accuracy (%)', fontsize=22, weight='medium')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=22)
ax.tick_params(axis='both', labelsize=18, width=1.2)

# Customize legend - placed below to save horizontal space
legend = ax.legend(loc='upper center', 
                   bbox_to_anchor=(0.5, -0.12),
                   ncol=3,
                   fontsize=18,
                   frameon=True,
                   fancybox=True,
                   shadow=True)
legend.get_frame().set_edgecolor('gray')

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Customize grid - vertical lines often clutter grouped bar charts, removed here
ax.grid(True, axis='x', alpha=0.3, linewidth=0.8, linestyle='--')
ax.set_axisbelow(True)

# Set y-axis to start at 0
ax.set_ylim(0, 115)

plt.tight_layout()

# Save the figure
plt.savefig('plotting/sad_stages_effect_re.pdf', dpi=400, bbox_inches='tight')
plt.savefig('plotting/sad_stages_effect_re.png', dpi=400, bbox_inches='tight')
plt.close()