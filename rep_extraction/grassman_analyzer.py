import torch
import argparse
from datasets import Dataset
from typing import Dict, List
from utils.vllm_utils import model_dictionary
from custom_typing.scenario import SorryBenchSAScenario
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_training.data_processor import process_sorry_bench_sa_data
from data.dataloader import load_sorry_bench_sa_data, load_wmdp_sa_data
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction, layer_last_token_extractor

def principal_component_analysis(
    rep_matrix: torch.Tensor
) -> torch.Tensor:
    '''
        Computes the principal components of a representation matrix using SVD
        rep_matrix: Tensor of shape (num_samples, representation_dim)
        output: returns (r, representation_dim) where r is the rank of the representation matrix, and the columns are the principal components sorted by explained variance
    '''
    # Centering the data
    rep_matrix_centered = rep_matrix - torch.mean(rep_matrix, dim=0)

    # Performing SVD and setting full_matrices=False to get the compact SVD
    U, S, Vh = torch.linalg.svd(rep_matrix_centered, full_matrices=False)

    return Vh

def lora_subspace_similarity(
    rep_1: torch.Tensor,
    rep_2: torch.Tensor
) -> torch.Tensor:
    '''
        Computes the subspace similarity between two representation matrices using the lora approach
    '''
    # QR decomposition to get orthonormal bases
    Q1, _ = torch.linalg.qr(rep_1)
    Q2, _ = torch.linalg.qr(rep_2)

    # Compute the matrix of cosines of principal angles
    M = torch.matmul(Q1.T, Q2)
    _, S, _ = torch.linalg.svd(M)

    # Clamp singular values to [-1, 1] to avoid numerical issues
    S = torch.clamp(S, -1.0, 1.0)

    # squaring the singular values
    S_squared = S ** 2

    # computing the cummulative sum
    cumsum_S_squared = torch.cumsum(S_squared, dim=0)

    # dividing by the total elements for each item in cumsum
    total_elements = torch.arange(1, len(S_squared) + 1, device=S_squared.device)
    normalized_cumsum = cumsum_S_squared / total_elements

    return normalized_cumsum


def raw_data_rep_extraction(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[SorryBenchSAScenario],
    layer_idx: int
) -> torch.Tensor:
    '''
        Processes the sa type data to get the representations for all samples
    '''

    # getting the scenario element from each item in the list
    text_data = [item['scenario'] for item in data]

    # getting the representations
    all_representations = vanilla_hidden_state_extraction(
        llm=llm,
        tokenizer=tokenizer,
        texts=text_data,
        batch_size=8,
        append_last_only=True,
        logging=False
    )
    
    # getting the last token representations at a specific layer
    representations = layer_last_token_extractor(
        all_hidden_states=all_representations,
        layer_idx=layer_idx
    )

    return representations



def data_specific_rep_extraction(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: Dataset,
    layer_idx: int
) -> Dict:
    '''
        Processes a sa_data to get the representations for oversight and non-oversight samples
    '''

    # extracting the representations for the oversight
    oversight_text = data['oversight_input']
    all_oversight_representations = vanilla_hidden_state_extraction(
        llm=llm,
        tokenizer=tokenizer,
        texts=oversight_text,
        batch_size=8,
        append_last_only=True,
        logging=False
    )
    oversight_representations = layer_last_token_extractor(
        all_hidden_states=all_oversight_representations,
        layer_idx=layer_idx
    )

    # extracting the representations for the non-oversight
    non_oversight_text = data['non_oversight_input']
    all_non_oversight_representations = vanilla_hidden_state_extraction(
        llm=llm,
        tokenizer=tokenizer,
        texts=non_oversight_text,
        batch_size=8,
        append_last_only=True,
        logging=False
    )
    non_oversight_representations = layer_last_token_extractor(
        all_hidden_states=all_non_oversight_representations,
        layer_idx=layer_idx
    )

    return {
        'oversight': oversight_representations,
        'non_oversight': non_oversight_representations
    }


def pipeline_grassmann_analyzer(
    model_name: str,
    layer_idx: int,
    top_k: int
) -> None:
    '''
        Pipeline for analyzing the LoRA grassmannian between WMDP and SorryBench datasets
    '''

    # loading the model and tokenizer
    model_info = model_dictionary[model_name]
    llm = AutoModelForCausalLM.from_pretrained(model_info['model'], device_map='auto', dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_info['model'])

    # loading both the data: SorryBench and WMDP
    sorry_bench_data = process_sorry_bench_sa_data(
        tokenizer,
        random_seed=42,
        evaluation_split=0.1,
        data_name='sorry_bench_sa'
    )
    wmdp_data = process_sorry_bench_sa_data(
        tokenizer,
        random_seed=42,
        evaluation_split=0.1,
        data_name='wmdp_sa'
    )

    # loading the raw data with the dataloader functions
    raw_sorry_bench_data = load_sorry_bench_sa_data()
    raw_wmdp_data = load_wmdp_sa_data()

    # creating the data structure for storing the representations
    rep_data = {
        'sorry_bench': {
            'oversight': None,
            'non_oversight': None
        },
        'wmdp': {
            'oversight': None,
            'non_oversight': None 
        }
    }
    raw_rep_data = {
        'sorry_bench': raw_data_rep_extraction(
            llm=llm,
            tokenizer=tokenizer,
            data=raw_sorry_bench_data,
            layer_idx=layer_idx
        ),
        'wmdp': raw_data_rep_extraction(
            llm=llm,
            tokenizer=tokenizer,
            data=raw_wmdp_data,
            layer_idx=layer_idx
        )
    }

    # adding the representations for all the data into this structure
    rep_data['sorry_bench'] = data_specific_rep_extraction(
        llm=llm,
        tokenizer=tokenizer,
        data=sorry_bench_data['train'],
        layer_idx=layer_idx
    )
    rep_data['wmdp'] = data_specific_rep_extraction(
        llm=llm,
        tokenizer=tokenizer,
        data=wmdp_data['train'],
        layer_idx=layer_idx 
    )

    # compute the subspace similarity for contrastive representations and the raw representations
    contrastive_sorry_bench = (torch.tensor(rep_data['sorry_bench']['oversight']) - torch.tensor(rep_data['sorry_bench']['non_oversight'])).float()
    contrastive_wmdp = (torch.tensor(rep_data['wmdp']['oversight']) - torch.tensor(rep_data['wmdp']['non_oversight'])).float()

    # getting the principal components for the contrastive representations
    contrastive_sorry_bench = principal_component_analysis(contrastive_sorry_bench)
    contrastive_wmdp = principal_component_analysis(contrastive_wmdp)

    # computing the subspace similarity
    contrastive_similarity = lora_subspace_similarity(
        rep_1=contrastive_sorry_bench[:top_k].T,
        rep_2=contrastive_wmdp[:top_k].T
    )

    # getting the principal components for the raw representations
    raw_sorry_bench = principal_component_analysis(torch.tensor(raw_rep_data['sorry_bench']).float())
    raw_wmdp = principal_component_analysis(torch.tensor(raw_rep_data['wmdp']).float())

    data_similarity = lora_subspace_similarity(
        rep_1=raw_sorry_bench[:top_k].T,
        rep_2=raw_wmdp[:top_k].T
    )

    print(contrastive_similarity)
    print(data_similarity)

    # saving the results
    torch.save({
        'contrastive_similarity': contrastive_similarity,
        'data_similarity': data_similarity
    }, f'grassmannian_results_{model_name}_layer_{layer_idx}.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzing the LoRA grassmannian between WMDP and sorry bench datasets')
    parser.add_argument('--model_name', type=str, default='olmo2-7b-instruct', choices=list(model_dictionary.keys()))
    parser.add_argument('--layer_idx', type=int, default=15, help='Layer index to analyze the grassmannian on')
    parser.add_argument('--top_k', type=int, default=32, help='Number of top principal components to consider for the analysis')
    args = parser.parse_args()

    pipeline_grassmann_analyzer(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        top_k=args.top_k
    )