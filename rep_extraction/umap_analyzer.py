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
from sklearn.metrics import calinski_harabasz_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_extraction.data_processing import pipeline_tokenize_diff_oversight
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction, layer_last_token_extractor



def plot_umap_2d(
    representations: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str] | Sequence[int],
    title: str | None = None,
    save_path: str | None = None,
    point_size: float = 70.0,  # Increased from 8.0 for better visibility
    alpha: float = 0.75,  # Slightly reduced for better overlap handling
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

    # Build a stable color mapping with academic-friendly colors
    unique_labels = list(dict.fromkeys(labels))
    
    # Professional color palette for academic papers
    academic_colors = [
        '#2E86AB',  # Ocean blue
        '#A23B72',  # Plum
        '#F18F01',  # Orange
        '#C73E1D',  # Brick red
        '#6A994E',  # Forest green
        '#BC4B51',  # Rose
        '#8E7DBE',  # Purple
        '#F4A259',  # Peach
        '#5E8C61',  # Sage
        '#D4A373',  # Tan
    ]
    
    label_to_color = {
        lab: academic_colors[i % len(academic_colors)] 
        for i, lab in enumerate(unique_labels)
    }

    # Use a clean style suitable for publications
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))  # Larger figure for better quality
    
    for lab in unique_labels:
        mask = np.array(labels) == lab
        ax.scatter(
            reps_2d[mask, 0],
            reps_2d[mask, 1],
            s=point_size,
            alpha=alpha,
            color=label_to_color[lab],
            label=str(lab),
            edgecolors='white',  # White edges for better contrast
            linewidths=0.8,  # Thicker edge for academic clarity
        )

    # Enhanced axis labels
    ax.set_xlabel("UMAP Dimension 1", fontsize=28, weight='medium')
    ax.set_ylabel("UMAP Dimension 2", fontsize=28, weight='medium')
    
    if title:
        ax.set_title(title, fontsize=28, weight='bold', pad=20)

    # Clean spines for publication quality
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Subtle grid
    ax.grid(True, alpha=0.2, linewidth=0.8, linestyle='--')
    ax.set_aspect('equal', 'datalim')

    # Enhanced legend
    legend = ax.legend(
        loc='best', 
        fontsize=28, 
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1,
        labelspacing=0.8
    )
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.0)

    # Tick parameters for cleaner look
    ax.tick_params(axis='both', labelsize=22, width=1.2, length=6)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save at high DPI for publication quality
        fig.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


def pipeline_representation_plotter(
    model_name: str,
    system_prompt_name: Iterable[str],
    system_prompt: Iterable[str],
    data_name: str,
    user_label: str,
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

    # creating the system prompt mapper
    system_prompt_mapper = {}
    for sp_name, sp in zip(system_prompt_name, system_prompt):
        system_prompt_mapper[sp_name] = SystemPrompter.create_system_prompt(
            system_prompt_name=sp,
            company=model_info['company']
        )

    # loading the data
    scenario_list = load_mft_refined_data(data_name=data_name)

    # creating the tokenized data
    data_dict = pipeline_tokenize_diff_oversight(
        scenario_list=scenario_list,
        user_label=user_label,
        tokenizer=tokenizer,
        system_prompt_mapper=system_prompt_mapper
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
    # selected_layers = [5, 10, 15, 20]
    selected_layers = [5, 10, 15, 20, 25, 30]
    for layer_idx in selected_layers:
        all_representations = []
        all_labels = []

        # first plotting for each system prompt
        for sp_name in rep_data.keys():

            # for plotting
            non_oversight_reps = layer_last_token_extractor(
                all_hidden_states=rep_data[sp_name]['non_oversight'],
                layer_idx=layer_idx 
            )
            non_oversight_labels = [f"No oversight"] * non_oversight_reps.shape[0]
            oversight_reps = layer_last_token_extractor(
                all_hidden_states=rep_data[sp_name]['oversight'],
                layer_idx=layer_idx
            )
            oversight_labels = [f"oversight"] * oversight_reps.shape[0]
            
            # combining the data for plotting and UMAP
            sp_reps = np.concatenate([non_oversight_reps, oversight_reps], axis=0)
            sp_labels = non_oversight_labels + oversight_labels
            all_representations.append(sp_reps)
            all_labels.extend(sp_labels)

            # plotting for each system prompt separately
            plot_umap_2d(
                representations=sp_reps,
                labels=sp_labels,
                title=f"Representations at Layer {layer_idx} for {sp_name}",
                save_path=os.path.join(output_folder, f"umap_layer_{layer_idx}_{sp_name}.pdf")
            )

            # computing the calinski harabasz score for the current system prompt
            ch_score = calinski_harabasz_score(sp_reps, sp_labels)
            print(f"Calinski-Harabasz Score for Layer {layer_idx}, System Prompt '{sp_name}': {ch_score:.2f}")

        # combining all representations
        all_representations = np.concatenate(all_representations, axis=0)

        # plotting for all system prompts together
        plot_umap_2d(
            representations=all_representations,
            labels=all_labels,
            title=f"Representations at Layer {layer_idx} - All System Prompts",
            save_path=os.path.join(output_folder, f"umap_layer_{layer_idx}_all_system_prompts.pdf")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='olmo2-7b-instruct', choices=list(model_dictionary.keys()))
    parser.add_argument('--system_prompt_name', type=str, nargs='+', default=['Goal guarding', 'Self preservation', 'Resource acquisition', 'Influence seeking'])
    parser.add_argument('--system_prompt', type=str, nargs='+', choices=list(SystemPrompter.configuration.keys()), default=['mft_care_strong', 'mft_care_sp', 'mft_care_ra', 'mft_care_is'])
    parser.add_argument('--data_name', type=str, choices=[
        'mft_refined_authority_50',
        'mft_refined_care_50',
        'mft_refined_fairness_50',
        'mft_refined_loyalty_50',
        'mft_refined_sanctity_50'
    ], default='mft_refined_care_50')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_folder', type=str, default='results/af_representation/plots/olmo2-7b-instruct_mft_refined_care_50')
    args = parser.parse_args()

    pipeline_representation_plotter(
        model_name=args.model_name,
        system_prompt_name=args.system_prompt_name,
        system_prompt=args.system_prompt,
        data_name=args.data_name,
        user_label=args.user_label,
        batch_size=args.batch_size,
        output_folder=args.output_folder
    )