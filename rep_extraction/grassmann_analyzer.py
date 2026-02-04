import torch
import argparse
from typing import Dict
from datasets import Dataset
from utils.vllm_utils import model_dictionary
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_training.data_processor import process_sorry_bench_sa_data
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction, layer_last_token_extractor

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

    # Computing the square of S
    S_squared = S ** 2

    # create cummulative sum of squared singular values and divide by index
    cumsum_S_squared = torch.cumsum(S_squared, dim=0)
    avg_squared_cosines = cumsum_S_squared / torch.arange(1, S_squared.size(0) + 1, device=S_squared.device)
    return avg_squared_cosines



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
    layer_idx: int
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

    # compute the subspace similarities between the oversight, non-oversight and contrastive representations
    oversight_similarity = lora_subspace_similarity(
        rep_1=rep_data['sorry_bench']['oversight'],
        rep_2=rep_data['wmdp']['oversight']
    )
    non_oversight_similarity = lora_subspace_similarity(
        rep_1=rep_data['sorry_bench']['non_oversight'],
        rep_2=rep_data['wmdp']['non_oversight']
    )
    contrastive_similarity = lora_subspace_similarity(
        rep_1=rep_data['sorry_bench']['oversight'] - rep_data['sorry_bench']['non_oversight'],
        rep_2=rep_data['wmdp']['oversight'] - rep_data['wmdp']['non_oversight']
    )

    # saving the results
    torch.save(
        {
            'oversight_similarity': oversight_similarity,
            'non_oversight_similarity': non_oversight_similarity,
            'contrastive_similarity': contrastive_similarity
        },
        f'rep_extraction/grassmann_results_{model_name}_layer_{layer_idx}.pt'
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzing the LoRA grassmannian between WMDP and sorry bench datasets')
    parser.add_argument('--model_name', type=str, default='olmo2-7b-instruct', choices=list(model_dictionary.keys()))
    parser.add_argument('--layer_idx', type=int, default=15, help='Layer index to analyze the grassmannian on')
    args = parser.parse_args()

    pipeline_grassmann_analyzer(
        model_name=args.model_name,
        layer_idx=args.layer_idx
    )