"""
Script for computing PCA over element-wise difference vectors
(non_oversight - oversight) per layer and system prompt,
and plotting a publication-quality scree plot of explained variance.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # noqa: F401 (used for axis formatting)
from typing import Iterable, List
from sklearn.decomposition import PCA
from utils.vllm_utils import model_dictionary
from data.dataloader import load_mft_refined_data
from prompter.system_prompt import SystemPrompter
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_extraction.data_processing import pipeline_tokenize_diff_oversight
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction, layer_last_token_extractor


# ── Colour palette (pastel bars + darker border/hatch versions) ─────────────
_PASTEL = [
    '#AED6F1',  # pastel blue
    '#F1948A',  # pastel red
    '#A9DFBF',  # pastel green
    '#F9E79F',  # pastel yellow
    '#D7BDE2',  # pastel purple
    '#FDEBD0',  # pastel peach
    '#A3E4D7',  # pastel teal
    '#FAD7A0',  # pastel orange
]

_DARK = [
    '#1A5276',  # dark blue
    '#922B21',  # dark red
    '#1E8449',  # dark green
    '#B7950B',  # dark yellow
    '#6C3483',  # dark purple
    '#784212',  # dark peach
    '#148F77',  # dark teal
    '#784212',  # dark orange
]

_HATCHES = ['///', '\\\\\\', 'xxx', '+++', '...', 'ooo', '***', '|||']


def plot_scree(
    explained_variance_dict: dict,   # {sp_name: np.ndarray of cumulative EVR}
    layer_idx: int,
    top_k: int,
    save_path: str | None = None,
) -> None:
    """
    Plot a publication-quality marginal scree plot.

    Parameters
    ----------
    explained_variance_dict : dict
        Keys are system-prompt names; values are 1-D arrays of length >= top_k
        containing the *cumulative* explained variance ratio for each PC.
    layer_idx : int
        Layer index (used in title / filename).
    top_k : int
        Number of PCs to display on the x-axis.
    save_path : str or None
        If given, saves the figure at that path; otherwise calls plt.show().
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
    })

    sp_names = list(explained_variance_dict.keys())
    n_groups = len(sp_names)
    x = np.arange(1, top_k + 1)

    # bar width so all groups fit within each integer tick slot
    total_width = 0.75
    bar_w = total_width / n_groups

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.9), 5))

    for i, sp_name in enumerate(sp_names):
        cumev = explained_variance_dict[sp_name][:top_k]
        marginal = np.diff(cumev, prepend=0.0) * 100

        pastel = _PASTEL[i % len(_PASTEL)]
        dark   = _DARK[i % len(_DARK)]
        hatch  = _HATCHES[i % len(_HATCHES)]

        # offset each group of bars around the integer tick
        offsets = x - total_width / 2 + bar_w * (i + 0.5)

        ax.bar(
            offsets, marginal,
            width=bar_w,
            color=pastel,
            edgecolor=dark,
            linewidth=1.1,
            hatch=hatch,
            label=sp_name,
            zorder=3,
        )

    ax.set_xlabel('Principal Component', fontsize=22, labelpad=10)
    ax.set_ylabel('Explained Variance (%)', fontsize=22, labelpad=10)

    # integer ticks centred on each group
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], fontsize=18)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=18, width=1.1, length=5)
    ax.tick_params(axis='x', width=1.1, length=5)

    ax.set_xlim(0.5, top_k + 0.5)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8, linestyle='--', zorder=1)

    legend = ax.legend(
        loc='upper right',
        fontsize=20,
        frameon=True,
        fancybox=False,
        edgecolor='#cccccc',
        borderpad=0.9,
        labelspacing=0.6,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved scree plot → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def pipeline_pca_scree(
    model_name: str,
    system_prompt_name: Iterable[str],
    system_prompt: Iterable[str],
    data_name: str,
    user_label: str,
    batch_size: int,
    output_folder: str,
    selected_layers: List[int],
    top_k: int,
):
    """
    Main pipeline:
      1. Extract hidden states for oversight / non-oversight inputs.
      2. Compute element-wise difference vectors (non_oversight - oversight).
      3. Fit PCA on those difference vectors for each (layer, sp_name).
      4. Plot and save a scree plot per layer.
    """

    # ── Load model & tokenizer ───────────────────────────────────────────────
    model_info = model_dictionary[model_name]
    llm = AutoModelForCausalLM.from_pretrained(
        model_info['model'], device_map='auto', dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_info['model'])

    # ── Build system prompt mapper ───────────────────────────────────────────
    system_prompt_mapper = {
        sp_name: SystemPrompter.create_system_prompt(
            system_prompt_name=sp,
            company=model_info['company']
        )
        for sp_name, sp in zip(system_prompt_name, system_prompt)
    }

    # ── Load & tokenize data ─────────────────────────────────────────────────
    scenario_list = load_mft_refined_data(data_name=data_name)
    data_dict = pipeline_tokenize_diff_oversight(
        scenario_list=scenario_list,
        user_label=user_label,
        tokenizer=tokenizer,
        system_prompt_mapper=system_prompt_mapper,
    )

    # ── Extract hidden states ────────────────────────────────────────────────
    rep_data = {}
    for sp_name in data_dict.keys():
        print(f"Extracting representations for: {sp_name}")
        rep_data[sp_name] = {
            'non_oversight': vanilla_hidden_state_extraction(
                llm=llm,
                tokenizer=tokenizer,
                texts=data_dict[sp_name][0],
                batch_size=batch_size,
                append_last_only=True,
            ),
            'oversight': vanilla_hidden_state_extraction(
                llm=llm,
                tokenizer=tokenizer,
                texts=data_dict[sp_name][1],
                batch_size=batch_size,
                append_last_only=True,
            ),
        }

    # ── PCA over difference vectors per layer ────────────────────────────────
    os.makedirs(output_folder, exist_ok=True)

    for layer_idx in selected_layers:
        print(f"\n── Layer {layer_idx} ──")
        explained_variance_dict = {}

        for sp_name in rep_data.keys():
            non_ov = layer_last_token_extractor(
                all_hidden_states=rep_data[sp_name]['non_oversight'],
                layer_idx=layer_idx,
            )   # (N, D)
            ov = layer_last_token_extractor(
                all_hidden_states=rep_data[sp_name]['oversight'],
                layer_idx=layer_idx,
            )   # (N, D)

            # element-wise difference: non_oversight - oversight
            diff = non_ov - ov   # (N, D)

            # fit PCA
            n_components = min(top_k, diff.shape[0], diff.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(diff)
            cumulative_evr = np.cumsum(pca.explained_variance_ratio_)
            explained_variance_dict[sp_name] = cumulative_evr

            print(
                f"  {sp_name}: top-{n_components} PCs explain "
                f"{cumulative_evr[n_components-1]*100:.1f}% of variance"
            )

        save_path = os.path.join(output_folder, f"pca_scree_layer_{layer_idx}.pdf")
        plot_scree(
            explained_variance_dict=explained_variance_dict,
            layer_idx=layer_idx,
            top_k=top_k,
            save_path=save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA scree plot of oversight difference vectors per layer"
    )
    parser.add_argument('--model_name', type=str, default='olmo2-7b-instruct',
                        choices=list(model_dictionary.keys()))
    parser.add_argument('--system_prompt_name', type=str, nargs='+',
                        default=['Value preservation', 'Self preservation',
                                 'Resource acquisition', 'Influence seeking'])
    parser.add_argument('--system_prompt', type=str, nargs='+',
                        choices=list(SystemPrompter.configuration.keys()),
                        default=['mft_care_strong', 'mft_care_sp',
                                 'mft_care_ra', 'mft_care_is'])
    parser.add_argument('--data_name', type=str,
                        choices=[
                            'mft_refined_authority_50',
                            'mft_refined_care_50',
                            'mft_refined_fairness_50',
                            'mft_refined_loyalty_50',
                            'mft_refined_sanctity_50',
                        ],
                        default='mft_refined_care_50')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_folder', type=str,
                        default='results/af_representation/pca_scree/olmo2-7b-instruct_mft_refined_care_50')
    parser.add_argument('--selected_layers', type=int, nargs='+',
                        default=[5, 10, 15, 20, 25, 30])
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top PCs to display on the scree plot')
    args = parser.parse_args()

    pipeline_pca_scree(
        model_name=args.model_name,
        system_prompt_name=args.system_prompt_name,
        system_prompt=args.system_prompt,
        data_name=args.data_name,
        user_label=args.user_label,
        batch_size=args.batch_size,
        output_folder=args.output_folder,
        selected_layers=args.selected_layers,
        top_k=args.top_k,
    )
