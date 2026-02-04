import numpy as np
import matplotlib.pyplot as plt

# Data from provided images
models = ['olmo2-7b-instruct', 'olmo2-13b-instruct', 'qwen3-8b']
scenarios = ['Vanilla', 'Consistency (Sorry-Bench)', 'Redirection (Sorry-Bench)', 'Consistency (WMDP)', 'Redirection (WMDP)']

# Accuracy Data (%)
accuracy_data = {
    'Vanilla': [64.25, 82.5, 95.0],
    'Consistency (Sorry-Bench)': [60.5, 80.0, 92.75],
    'Redirection (Sorry-Bench)': [57.0, 80.25, 94.75],
    'Consistency (WMDP)': [59.5, 77.25, 94.25],
    'Redirection (WMDP)': [58.0, 81.0, 94.75]
}

# Drop Data (%) for annotations
drop_data = {
    'Vanilla': [0, 0, 0],
    'Consistency (Sorry-Bench)': [-5.84, -3.03, -2.37],
    'Redirection (Sorry-Bench)': [-11.28, -2.73, -0.26],
    'Consistency (WMDP)': [-7.39, -6.36, -0.79],
    'Redirection (WMDP)': [-9.73, -1.82, -0.26]
}

# Set up the plot with academic styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 6))

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
plt.savefig('plotting/sad_stages_effect.pdf', dpi=400, bbox_inches='tight')
plt.show()