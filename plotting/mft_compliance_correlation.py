import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── Data ──────────────────────────────────────────────────────────────────────
# Order: Care, Fairness, Loyalty, Sanctity, Authority

compliance = {
    "olmo2-7b-it":      [44.0, 34.0, 46.2, 38.8, 23.2],
    "olmo2-13b-it":     [27.4, 53.2, 26.2, 18.0, 24.0],
    "olmo2-32b-it":     [19.4, 36.4, 13.6, 19.2,  2.2],
    "qwen2.5-7b-it":    [ 6.2,  6.2,  1.8,  3.4,  2.8],
    "qwen2.5-14b-it":   [66.8, 58.0, 76.0, 38.4,  8.2],
    "qwen2.5-32b-it":   [33.0,  7.4,  7.2, 17.4,  6.0],
    "qwen3-8b":         [53.6, 64.0, 39.6, 33.2, 19.2],
    "qwen3-14b":        [51.4, 36.2, 27.8, 29.2, 21.2],
    "qwen3-32b":        [70.8, 55.4, 44.6, 31.8, 28.8],
    "gpt-oss-20b":      [13.6, 12.2,  3.6,  2.8,  1.8],
    "gpt-oss-120b":     [39.2, 22.2, 16.0,  5.2, -1.2],
    "llama3.1-8b-it":   [27.4, 21.4, 25.4,  4.2,  2.2],
    "llama3.1-70b-it":  [92.4, 89.8, 87.8, 59.0, 13.2],
    "gpt-4o":           [94.4, 68.2, 79.2, 68.2, 12.0],
}

mfq = {
    "olmo2-7b-it":      [4.17, 4.33, 3.67, 3.83, 3.17],
    "olmo2-13b-it":     [3.33, 3.50, 3.00, 3.17, 3.00],
    "olmo2-32b-it":     [4.50, 4.17, 3.33, 3.17, 3.00],
    "qwen2.5-7b-it":    [4.67, 4.33, 3.33, 3.50, 3.17],
    "qwen2.5-14b-it":   [4.83, 4.67, 3.17, 2.67, 2.50],
    "qwen2.5-32b-it":   [4.50, 4.33, 3.17, 2.67, 3.00],
    "qwen3-8b":         [4.00, 4.17, 3.17, 2.83, 3.00],
    "qwen3-14b":        [3.67, 4.33, 3.00, 2.67, 2.50],
    "qwen3-32b":        [4.00, 4.33, 3.33, 2.33, 3.00],
    "llama3.1-8b-it":   [4.00, 3.67, 3.00, 2.83, 2.50],
    "llama3.1-70b-it":  [4.83, 4.67, 3.17, 2.50, 3.00],
    "gpt-oss-20b":      [4.33, 4.33, 2.83, 2.50, 3.17],
    "gpt-oss-120b":     [4.83, 4.67, 3.33, 3.00, 3.33],
    "gpt-4o":           [4.17, 4.33, 2.83, 2.33, 2.83],
}

FAMILY_PASTEL = {
    "olmo2":   "#AEC6CF",
    "qwen2.5": "#FFD1A4",
    "qwen3":   "#FFB3B3",
    "llama":   "#B5EAD7",
    "gpt-oss": "#C7E4A0",
    "gpt-4o":  "#D5AAFF",
}

FAMILY_EDGE = {
    "olmo2":   "#2A6080",
    "qwen2.5": "#B05A00",
    "qwen3":   "#A01010",
    "llama":   "#1A7A55",
    "gpt-oss": "#3A6A10",
    "gpt-4o":  "#6A00AA",
}

FAMILY_HATCH = {
    "olmo2":   "//",
    "qwen2.5": "\\\\",
    "qwen3":   "xx",
    "llama":   "++",
    "gpt-oss": "oo",
    "gpt-4o":  "**",
}

def model_family(name):
    for fam in ["olmo2", "qwen2.5", "qwen3", "llama3", "gpt-oss", "gpt-4o"]:
        if name.startswith(fam):
            return fam.replace("llama3", "llama")
    return name

FAMILY_COLORS = FAMILY_PASTEL  # keep old reference working

# ── Compute per-model Pearson r between 5-d compliance and 5-d MFQ vectors ───

models = list(compliance.keys())
correlations, pvalues = [], []
for m in models:
    r, p = pearsonr(compliance[m], mfq[m])
    correlations.append(r)
    pvalues.append(p)

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(11, 4.2))

x = np.arange(len(models))
for i, m in enumerate(models):
    fam = model_family(m)
    ax.bar(x[i], correlations[i],
           color=FAMILY_PASTEL[fam],
           edgecolor=FAMILY_EDGE[fam],
           linewidth=1.2,
           hatch=FAMILY_HATCH[fam],
           width=0.65,
           zorder=3)

# significance markers
for i, (r, p) in enumerate(zip(correlations, pvalues)):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    if sig:
        ax.text(i, r + 0.02, sig, ha="center", va="bottom",
                fontsize=9, color="#333333")

# grid only
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Correlation", fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_xlim(-0.6, len(models) - 0.4)

# legend
seen = {}
for m in models:
    fam = model_family(m)
    if fam not in seen:
        seen[fam] = plt.Rectangle(
            (0, 0), 1, 1,
            fc=FAMILY_PASTEL[fam],
            ec=FAMILY_EDGE[fam],
            linewidth=1.2,
            hatch=FAMILY_HATCH[fam],
        )
ax.legend(
    list(seen.values()), list(seen.keys()),
    title="Model Family", title_fontsize=9,
    fontsize=8.5, frameon=True, framealpha=0.9,
    edgecolor="#cccccc", loc="upper right",
)

plt.tight_layout()
plt.savefig(
    "/home/inair/OOCR_based_alignment_faking/plotting/mft_compliance_correlation.pdf",
    bbox_inches="tight", dpi=200,
)
plt.savefig(
    "/home/inair/OOCR_based_alignment_faking/plotting/mft_compliance_correlation.png",
    bbox_inches="tight", dpi=200,
)
print("Saved mft_compliance_correlation.pdf / .png")
