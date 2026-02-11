import torch
import numpy as np
import matplotlib.pyplot as plt

# input files are present here
input_file = 'plotting/grassmannian_results_Qwen3-8B_layer_15.pt'

# opening the file using torch
results = torch.load(input_file)

# Data extracted from the provided image
k_list = list(range(1, len(results['contrastive_similarity']) + 1))[::3]
contrastive_similarity = results['contrastive_similarity'][::3].numpy()
data_similarity = results['data_similarity'][::3].numpy()

# Set up academic styling
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with distinct styles and markers
ax.plot(k_list, contrastive_similarity, label='Sim(diff(SORRY-BENCH), diff(WMDP), k)', 
        color='#457B9D', marker='o', markersize=9, linewidth=2.5, 
        linestyle='-', markerfacecolor='white', markeredgewidth=2)

ax.plot(k_list, data_similarity, label='Sim(SORRY-BENCH, WMDP, k)', 
        color='#9D4EDD', marker='s', markersize=9, linewidth=2.5, 
        linestyle='--', markerfacecolor='white', markeredgewidth=2)

# Customizing the axes
ax.set_xlabel('k', fontsize=18, fontweight='medium')
ax.set_ylabel('Similarity between Subspaces', fontsize=18, fontweight='medium')
ax.set_xticks(k_list)
ax.tick_params(axis='both', labelsize=14, width=1.2, length=6)

# Adding values above points for clarity (optional)
for i, txt in enumerate(contrastive_similarity):
    ax.annotate(f'{txt:.2f}', (k_list[i], contrastive_similarity[i]), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

for i, txt in enumerate(data_similarity):
    ax.annotate(f'{txt:.2f}', (k_list[i], data_similarity[i]), 
                textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10)

# Legend and Grid
ax.legend(loc='best', fontsize=14, frameon=True, shadow=True, borderpad=1)
ax.grid(True, linestyle='--', alpha=0.4)

# Clean aesthetics
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plotting/grassmannian_results_Qwen3-8B_layer_15_similarity.pdf', dpi=400)
plt.show()