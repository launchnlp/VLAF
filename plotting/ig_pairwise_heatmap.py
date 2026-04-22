import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/inair/OOCR_based_alignment_faking/results/ig_compliance_gaps.csv')

base_map = {'VP': 'VP-0', 'IS': 'IS-0', 'RA': 'RA-0', 'SP': 'SP-0'}
df['goal_id'] = df['goal_id'].apply(lambda x: base_map.get(x, x))

avg = df.groupby('goal_id')['compliance_gap'].mean() * 100

ig_groups = {
    'Value\nPreservation':   ['VP-0', 'VP-1', 'VP-2', 'VP-3'],
    'Influence\nSeeking':    ['IS-0', 'IS-1', 'IS-2', 'IS-3'],
    'Resource\nAcquisition': ['RA-0', 'RA-1', 'RA-2', 'RA-3'],
    'Self\nPreservation':    ['SP-0', 'SP-1', 'SP-2', 'SP-3'],
}

group_vals = {
    name: np.array([avg.get(gid, 0.0) for gid in ids])
    for name, ids in ig_groups.items()
}

# ── mean absolute pairwise difference ─────────────────────────────────────────
# off-diagonal: avg |a_i - b_j| over all cross-group pairs
# diagonal:     avg |a_i - a_j| over all within-group pairs (i ≠ j)
def mean_abs_diff(a, b):
    return np.mean([abs(ai - bj) for ai in a for bj in b])

def mean_abs_diff_within(a):
    n = len(a)
    return np.mean([abs(a[i] - a[j]) for i in range(n) for j in range(n) if i != j])

labels = list(ig_groups.keys())
n = len(labels)
matrix = np.zeros((n, n))
for i, la in enumerate(labels):
    for j, lb in enumerate(labels):
        if i == j:
            matrix[i, j] = mean_abs_diff_within(group_vals[la])
        else:
            matrix[i, j] = mean_abs_diff(group_vals[la], group_vals[lb])

# ── heatmap ───────────────────────────────────────────────────────────────────
FONT = 13
fig, ax = plt.subplots(figsize=(6.5, 5.5))

im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Mean Absolute Difference (x100)', fontsize=FONT - 1)
cbar.ax.tick_params(labelsize=FONT - 2)

# cell borders
for i in range(n + 1):
    ax.axhline(i - 0.5, color='white', linewidth=0.8)
    ax.axvline(i - 0.5, color='white', linewidth=0.8)

# annotations
for i in range(n):
    for j in range(n):
        v = matrix[i, j]
        color = 'white' if v > 0.75 * matrix.max() else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=FONT, color=color)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, fontsize=FONT - 1, ha='center')
ax.set_yticklabels(labels, fontsize=FONT - 1, va='center')
# ax.set_ylabel('Mean Absolute Difference Between Compliance Gaps (×100)', fontsize=FONT - 1)

plt.tight_layout()
plt.savefig('/home/inair/OOCR_based_alignment_faking/plotting/ig_pairwise_heatmap.pdf',
            bbox_inches='tight')
plt.savefig('/home/inair/OOCR_based_alignment_faking/plotting/ig_pairwise_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.show()
