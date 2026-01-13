import torch
from typing import Tuple
from transformers import AutoModel
from torch.utils.data import DataLoader

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


def activation_redirection_loss(
    oversight_representations: torch.Tensor,
    non_oversight_representations: torch.Tensor,
    original_non_oversight_representations: torch.Tensor
) -> torch.Tensor:
    '''
        Activation redirection loss between oversight and non-oversight token representations
    '''

    # redirecting the oversight representations
    oversight_loss = torch.nn.functional.mse_loss(
        original_non_oversight_representations.mean(dim=0),
        oversight_representations
    ) * oversight_representations.size(1)
    non_oversight_loss = torch.nn.functional.mse_loss(
        original_non_oversight_representations.mean(dim=0),
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