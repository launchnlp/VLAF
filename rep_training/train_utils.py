import os
import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel
from torch.utils.data import DataLoader
from typing import Iterable, Sequence, Tuple

def last_token_rep_extractor(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int
) -> torch.Tensor:
    '''
        Extracting representations from a specific layer of the model at the last token position
    '''

    # getting the representations
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    representation = output.hidden_states[layer] # B x S x Embdim

    # getting the last token representations using attention_mask
    seq_lengths = attention_mask.sum(dim=1) - 1
    last_token_representations = representation[torch.arange(representation.size(0)), seq_lengths, :] # B x Embdim

    return last_token_representations
    

def activation_consistency_loss(
    token_representation_1: torch.Tensor,
    token_representation_2: torch.Tensor
) -> torch.Tensor:
    '''
        Activation consistency loss between two token representations
    '''

    loss = torch.nn.functional.mse_loss(
        token_representation_1,
        token_representation_2
    ) * token_representation_1.size(1)

    return loss


def numpify_representations(
    oversight_representations: torch.Tensor,
    non_oversight_representations: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Converting the token representations to numpy arrays for UMAP plotting
        Joins the representations and creates labels [oversight, oversight, ... non_oversight, non_oversight, ...]
    '''

    oversight_reps_np = oversight_representations.cpu().numpy()
    non_oversight_reps_np = non_oversight_representations.cpu().numpy()
    all_reps_np = np.concatenate([oversight_reps_np, non_oversight_reps_np], axis=0)
    labels = ['oversight'] * oversight_reps_np.shape[0] + ['non_oversight'] * non_oversight_reps_np.shape[0]
    return all_reps_np, labels


def activation_redirection_loss(
    oversight_representations: torch.Tensor,
    non_oversight_representations: torch.Tensor,
    anchor_representations: torch.Tensor
) -> torch.Tensor:
    '''
        Activation redirection loss between oversight and non-oversight token representations
    '''

    # redirecting the oversight representations
    oversight_loss = torch.nn.functional.mse_loss(
        anchor_representations,
        oversight_representations
    ) * oversight_representations.size(1)
    non_oversight_loss = torch.nn.functional.mse_loss(
        anchor_representations,
        non_oversight_representations,
    ) * non_oversight_representations.size(1)
    loss = oversight_loss + non_oversight_loss
    return loss

def compute_original_representations(
    model: AutoModel,
    dataloader: DataLoader,
    layer: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Computing the original representations for oversight and non-oversight conditions
    '''

    all_oversight_representations = []
    all_non_oversight_representations = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            oversight_input_ids = batch['oversight_input_ids']
            oversight_attention_mask = batch['oversight_attention_mask']
            non_oversight_input_ids = batch['non_oversight_input_ids']
            non_oversight_attention_mask = batch['non_oversight_attention_mask']

            oversight_representations = last_token_rep_extractor(
                model,
                oversight_input_ids,
                oversight_attention_mask,
                layer=layer
            )
            non_oversight_representations = last_token_rep_extractor(
                model,
                non_oversight_input_ids,
                non_oversight_attention_mask,
                layer=layer
            )

            all_oversight_representations.append(oversight_representations)
            all_non_oversight_representations.append(non_oversight_representations)

    all_oversight_representations = torch.cat(all_oversight_representations, dim=0)
    all_non_oversight_representations = torch.cat(all_non_oversight_representations, dim=0)

    return all_oversight_representations, all_non_oversight_representations

def plot_umap_2d(
    representations: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str] | Sequence[int],
    title: str | None = None,
    save_path: str | None = None,
    point_size: float = 12.0,
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