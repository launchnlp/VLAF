"""
Script for plotting UMAP representations of last-token hidden states
for different instrumental goals (operationalized through system prompt)
and for different layers.
"""


import os
import umap
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Sequence
from utils.vllm_utils import model_dictionary
from data.dataloader import load_mft_refined_data
from prompter.system_prompt import SystemPrompter
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_extraction.data_processing import pipeline_tokenize_diff_oversight_sorry_bench
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction, layer_last_token_extractor



def plot_umap_2d(
    representations: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str] | Sequence[int],
    title: str | None = None,
    save_path: str | None = None,
    point_size: float = 8.0,
    alpha: float = 0.8,
    n_neighbors: int = 30,
    min_dist: float = 0.1
) -> None:
    """Run 2D UMAP on representations and create a scatter plot.

    Args:
        representations: Array-like of shape (n_samples, dim) with embeddings.
        labels: Class labels for each sample (used for colors in the plot).
        title: Optional plot title.
        save_path: If provided, path to save the figure (PNG). If None,
            the plot is just shown via ``plt.show()``.
        point_size: Matplotlib scatter point size.
        alpha: Point transparency.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance parameter for UMAP.
    """
    reps = np.asarray(representations)
    if reps.ndim != 2:
        raise ValueError("representations must be 2D (n_samples, dim)")

    if len(labels) != reps.shape[0]:
        raise ValueError("labels length must match number of representations")

    reducer = umap.UMAP(
        n_components=2,
        random_state=None,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    reps_2d = reducer.fit_transform(reps)

    # Build a stable color mapping for labels
    unique_labels = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    label_to_color = {
        lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for lab in unique_labels:
        mask = np.array(labels) == lab
        ax.scatter(
            reps_2d[mask, 0],
            reps_2d[mask, 1],
            s=point_size,
            alpha=alpha,
            color=label_to_color[lab],
            label=str(lab),
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, weight="bold")

    # Slightly de-emphasize top/right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, alpha=0.15, linewidth=0.7)
    ax.set_aspect("equal", "datalim")

    legend = ax.legend(loc="best", fontsize="small", frameon=True)
    legend.get_frame().set_alpha(0.9)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def pipeline_representation_plotter(
    model_name: str,
    data_name: str,
    batch_size: int,
    output_folder: str
):
    '''
        Pipeline for plotting UMAP representations of last-token hidden states
    '''

    # loading the model and tokenizer
    model_info = model_dictionary[model_name]
    llm = AutoModelForCausalLM.from_pretrained(model_info['model'], device_map='auto', dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_info['model'])

    # loading the data
    scenario_list = load_mft_refined_data(data_name=data_name)

    # creating the tokenized data
    data_dict = pipeline_tokenize_diff_oversight_sorry_bench(
        scenario_list=scenario_list,
        tokenizer=tokenizer
    )

    # creating the representations
    rep_data = {}
    for sp_name in data_dict.keys():

        # logging
        print(f"Extracting representations for system prompt: {sp_name}")

        # extracting the representations for data
        non_oversight_representations = vanilla_hidden_state_extraction(
            llm=llm,
            tokenizer=tokenizer,
            texts=data_dict[sp_name][0],
            batch_size=batch_size,
            append_last_only=True,
        )
        oversight_representations = vanilla_hidden_state_extraction(
            llm=llm,
            tokenizer=tokenizer,
            texts=data_dict[sp_name][1],
            batch_size=batch_size,
            append_last_only=True,
        )
        rep_data[sp_name] = {
            'non_oversight': non_oversight_representations,
            'oversight': oversight_representations
        }

    # plotting the UMAP for layers 5, 10, 15, 20
    os.makedirs(output_folder, exist_ok=True)
    selected_layers = [5, 10, 15, 20]
    for layer_idx in selected_layers:

        # first plotting for each system prompt
        for sp_name in rep_data.keys():

            # for plotting
            non_oversight_reps = layer_last_token_extractor(
                all_hidden_states=rep_data[sp_name]['non_oversight'],
                layer_idx=layer_idx 
            )
            non_oversight_labels = [f"{sp_name}_non_oversight"] * non_oversight_reps.shape[0]
            oversight_reps = layer_last_token_extractor(
                all_hidden_states=rep_data[sp_name]['oversight'],
                layer_idx=layer_idx
            )
            oversight_labels = [f"{sp_name}_oversight"] * oversight_reps.shape[0]
            
            # combining the data for plotting and UMAP
            sp_reps = np.concatenate([non_oversight_reps, oversight_reps], axis=0)
            sp_labels = non_oversight_labels + oversight_labels

            # plotting for each system prompt separately
            plot_umap_2d(
                representations=sp_reps,
                labels=sp_labels,
                title=f"UMAP Layer {layer_idx} - {sp_name}",
                save_path=os.path.join(output_folder, f"umap_layer_{layer_idx}_{sp_name}.png")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='olmo2-7b-instruct', choices=list(model_dictionary.keys()))
    parser.add_argument('--data_name', type=str, default='sorry_bench_sa')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_folder', type=str, default='results/af_representation/plots/olmo2-7b-instruct_sorry_bench_sa')
    args = parser.parse_args()

    pipeline_representation_plotter(
        model_name=args.model_name,
        data_name=args.data_name,
        batch_size=args.batch_size,
        output_folder=args.output_folder
    )