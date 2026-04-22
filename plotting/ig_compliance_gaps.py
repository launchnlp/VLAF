import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/inair/OOCR_based_alignment_faking/results/ig_compliance_gaps.csv')

base_map = {'VP': 'VP-0', 'IS': 'IS-0', 'RA': 'RA-0', 'SP': 'SP-0'}
df['goal_id'] = df['goal_id'].apply(lambda x: base_map.get(x, x))

avg = df.groupby('goal_id')['compliance_gap'].mean() * 100

# ── layout config ─────────────────────────────────────────────────────────────
groups = {
    'Value\nPreservation':   ('VP', ['VP-0', 'VP-1', 'VP-2', 'VP-3']),
    'Influence\nSeeking':    ('IS', ['IS-0', 'IS-1', 'IS-2', 'IS-3']),
    'Resource\nAcquisition': ('RA', ['RA-0', 'RA-1', 'RA-2', 'RA-3']),
    'Self\nPreservation':    ('SP', ['SP-0', 'SP-1', 'SP-2', 'SP-3']),
}

pastel_fill = {
    'Value\nPreservation':   '#AEC6CF',
    'Influence\nSeeking':    '#FFDAB9',
    'Resource\nAcquisition': '#B5EAD7',
    'Self\nPreservation':    '#C9B1D9',
}
avg_fill = {                          # darker shade for the summary bar
    'Value\nPreservation':   '#6A9BAC',
    'Influence\nSeeking':    '#D4956A',
    'Resource\nAcquisition': '#6AAC8A',
    'Self\nPreservation':    '#9A6AB0',
}
dark_edge = {
    'Value\nPreservation':   '#4A7A8A',
    'Influence\nSeeking':    '#C07840',
    'Resource\nAcquisition': '#3A8A6A',
    'Self\nPreservation':    '#7A4A9A',
}
hatches = ['', '///', 'xx', '\\\\\\']

no_consequence_value = 6.3

first_bar_overrides = {
    'Value\nPreservation':   30.5,
    'Influence\nSeeking':    21.5,
    'Resource\nAcquisition': 27.2,
    'Self\nPreservation':    22.3,
}

bar_w     = 0.18
avg_gap   = 0.05   # gap between last variant bar and average bar
group_gap = 0.35

FONT = 13

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))

x = 0.0
tick_positions, tick_labels = [], []
group_centers, group_name_list = [], []

for gname, (pfx, ids) in groups.items():
    positions = [x + i * bar_w for i in range(4)]

    # variant bars
    for i, (pos, gid, hatch) in enumerate(zip(positions, ids, hatches)):
        val = first_bar_overrides[gname] if i == 0 else avg.get(gid, 0.0)
        ax.bar(pos, val, width=bar_w * 0.88,
               color=pastel_fill[gname], edgecolor=dark_edge[gname],
               linewidth=1.3, hatch=hatch)
        tick_positions.append(pos)
        tick_labels.append(f'{pfx}-{i + 1}')

    # average bar (use override for first bar if set)
    bar_vals = [first_bar_overrides[gname]] + [avg.get(gid, 0.0) for gid in ids[1:]]
    avg_val  = np.mean(bar_vals)
    avg_pos  = positions[-1] + bar_w + avg_gap
    ax.bar(avg_pos, avg_val, width=bar_w * 0.88,
           color=avg_fill[gname], edgecolor=dark_edge[gname], linewidth=1.5)
    tick_positions.append(avg_pos)
    tick_labels.append('Avg')

    group_centers.append((positions[0] + avg_pos) / 2)
    group_name_list.append(gname)
    x = avg_pos + bar_w + group_gap

# Horizontal reference line for "No Consequence"
ax.axhline(
    y=no_consequence_value,
    color="#E07B54",
    linewidth=2.5,
    linestyle="--",
    label=f"No Consequence ({no_consequence_value})",
    zorder=3
)

# ── primary x-axis: per-bar labels ───────────────────────────────────────────
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=FONT - 1, rotation=45, ha='right')

# ── secondary x-axis: group labels (outward offset) ──────────────────────────
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(group_centers)
ax2.set_xticklabels(group_name_list)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 52))
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', length=0, labelsize=FONT + 3)

# ── axes formatting ───────────────────────────────────────────────────────────
ax.set_ylabel('Average Compliance Gap (x100)', fontsize=FONT)
ax.yaxis.grid(True, alpha=0.35, linestyle='--', color='grey')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(bottom=0)
ax.tick_params(axis='y', labelsize=FONT - 1)

# ── legend: instrumental goal colours only ────────────────────────────────────
group_patches = [
    Patch(facecolor=pastel_fill[g], edgecolor=dark_edge[g],
          label=g.replace('\n', ' '))
    for g in groups
]
avg_patch = Patch(facecolor='#888888', edgecolor='#444444', label='Group Average')
nc_line = Line2D([0], [0], color='#E07B54', linewidth=2.5, linestyle='--',
                 label=f'No Consequence ({no_consequence_value})')
ax.legend(
    handles=group_patches + [nc_line],
    title='Instrumental Goal',
    loc='upper left',
    fontsize=FONT - 2,
    title_fontsize=FONT - 2,
    framealpha=0.85,
)

plt.tight_layout()
plt.savefig(
    '/home/inair/OOCR_based_alignment_faking/plotting/ig_compliance_gaps.pdf',
    bbox_inches='tight',
)
plt.savefig(
    '/home/inair/OOCR_based_alignment_faking/plotting/ig_compliance_gaps.png',
    dpi=150,
    bbox_inches='tight',
)
plt.show()
