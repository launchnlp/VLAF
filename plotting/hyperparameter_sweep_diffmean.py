import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "/home/inair/OOCR_based_alignment_faking/results/MFT_refined_50_re"
VALUES = ["authority", "care", "fairness", "loyalty", "sanctity"]
STRENGTHS = [3, 6, 12, 24]
DATASETS = ["sorry_bench_sa", "wmdp_sa"]

MODELS = {
    "olmo2-7b-instruct": {"max_layer": 30, "vanilla": 37.2, "label": "OLMo2-7B-Instruct"},
    "Qwen3-8B":          {"max_layer": 35, "vanilla": 41.9, "label": "Qwen3-8B"},
}

SELECTED_LAYERS = {
    ("olmo2-7b-instruct", "sorry_bench_sa"): 15,
    ("olmo2-7b-instruct", "wmdp_sa"):        15,
    ("Qwen3-8B",          "sorry_bench_sa"): 15,
    ("Qwen3-8B",          "wmdp_sa"):        20,
}

STRENGTH_COLORS  = {3: "#1f77b4", 6: "#ff7f0e", 12: "#2ca02c", 24: "#d62728"}
STRENGTH_MARKERS = {3: "o",       6: "s",        12: "^",       24: "D"}

DATASET_LABELS = {"sorry_bench_sa": "SorryBench", "wmdp_sa": "WMDP"}

FONT = 18   # base font size


def load_compliance_gap(model, dataset, layer, strength):
    gaps = []
    for value in VALUES:
        for s_str in [str(strength), f"{strength}.0"]:
            fname = (
                f"vllm:{model}_diffmean_{dataset}.gguf_{layer}_{s_str}"
                "_option_based_compliance.json"
            )
            fpath = os.path.join(RESULTS_DIR, value, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    d = json.load(f)
                gaps.append(d["overall"]["compliance_gap"])
                break
    return np.mean(gaps) * 100 if gaps else np.nan


# ── gather data ───────────────────────────────────────────────────────────────
data = {}
valid_layers = {}   # valid_layers[model][dataset] = sorted list of layers with ≥1 strength present
for model, cfg in MODELS.items():
    data[model] = {}
    valid_layers[model] = {}
    all_layers = list(range(5, cfg["max_layer"] + 1, 5))
    for dataset in DATASETS:
        data[model][dataset] = {}
        for strength in STRENGTHS:
            data[model][dataset][strength] = {
                layer: load_compliance_gap(model, dataset, layer, strength)
                for layer in all_layers
            }
        # keep only layers where at least one strength returned a real value
        valid_layers[model][dataset] = [
            layer for layer in all_layers
            if any(
                not np.isnan(data[model][dataset][s].get(layer, np.nan))
                for s in STRENGTHS
            )
        ]


# ── shared legend handles ─────────────────────────────────────────────────────
def make_legend_handles():
    strength_lines = [
        Line2D([0], [0], color=STRENGTH_COLORS[s], marker=STRENGTH_MARKERS[s],
               linewidth=2.2, markersize=8, label=f"α = {s}")
        for s in STRENGTHS
    ]
    star_handle = Line2D(
        [0], [0], marker="*", color="grey", markersize=16,
        linewidth=0, markeredgecolor="black", markeredgewidth=0.6,
        label="Selected config",
    )
    vanilla_handle = Line2D(
        [0], [0], color="black", linewidth=2.0, linestyle=":",
        label="No intervention",
    )
    band_handle = mpatches.Patch(
        facecolor="#d4e6f1", edgecolor="#2471a3",
        label="Selected layer",
    )
    return strength_lines + [star_handle, vanilla_handle, band_handle]


# ── one figure per (model, dataset) ──────────────────────────────────────────
for model, cfg in MODELS.items():
    for dataset in DATASETS:
        selected_layer = SELECTED_LAYERS[(model, dataset)]
        layers = valid_layers[model][dataset]

        fig, ax = plt.subplots(figsize=(7, 5))

        for strength in STRENGTHS:
            row_data = data[model][dataset][strength]
            xs = [l for l in layers if not np.isnan(row_data.get(l, np.nan))]
            ys = [row_data[l] for l in xs]

            ax.plot(
                xs, ys,
                color=STRENGTH_COLORS[strength],
                marker=STRENGTH_MARKERS[strength],
                linewidth=2.2,
                markersize=7,
                label=f"α = {strength}",
                zorder=3,
            )

            # star at selected layer
            if selected_layer in row_data and not np.isnan(row_data[selected_layer]):
                ax.plot(
                    selected_layer, row_data[selected_layer],
                    marker="*",
                    color=STRENGTH_COLORS[strength],
                    markersize=20,
                    zorder=5,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                )

        # vertical band + line for selected layer
        ax.axvspan(
            selected_layer - 1.5, selected_layer + 1.5,
            color="#d4e6f1", alpha=0.55, zorder=1,
        )
        ax.axvline(
            selected_layer, color="#2471a3", linewidth=1.8,
            linestyle="--", zorder=2,
        )

        # vanilla dotted baseline
        ax.axhline(
            cfg["vanilla"],
            color="black", linewidth=2.0, linestyle=":",
            zorder=4,
        )

        step = layers[1] - layers[0] if len(layers) > 1 else 5
        ax.set_xticks(layers)
        ax.set_xlim(layers[0] - step * 0.4, layers[-1] + step * 0.4)
        ax.set_xlabel("Layer index", fontsize=FONT)
        ax.set_ylabel("Avg. compliance gap (%)", fontsize=FONT)
        ax.tick_params(axis="both", labelsize=FONT - 2)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--", color="grey")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            handles=make_legend_handles(),
            fontsize=FONT - 4,
            framealpha=0.9,
            loc="best",
        )

        plt.tight_layout()

        tag = f"{model.replace('/', '_')}_{dataset}"
        out_base = f"/home/inair/OOCR_based_alignment_faking/plotting/hparam_sweep_{tag}"
        plt.savefig(out_base + ".pdf", bbox_inches="tight")
        plt.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_base}.{{pdf,png}}")
        plt.close(fig)
