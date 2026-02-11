'''
    Code for extracting representations from the datasets with situational framing information
'''
import os
import torch
import argparse
from typing import List
from utils.vllm_utils import model_dictionary
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_training.data_processor import process_sorry_bench_sa_data
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction
from rep_extraction.data_processing import pipeline_tokenize_background_vs_nothing_data
from easysteer.steer import extract_diffmean_control_vector, extract_lat_control_vector, extract_pca_control_vector
    

def pipeline_rep_extractor_wo_response(
    model_name: str,
    data_name: str,
    approach: str,
    batch_size: int,
    output_file: str
) -> None:
    ''' 
        Pipeline to extract representations to capture alignment faking
    '''

    # loading the model and tokenizer
    model_info = model_dictionary[model_name]
    llm = AutoModelForCausalLM.from_pretrained(model_info['model'], device_map='auto', dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_info['model'])

    # loading the data
    data = process_sorry_bench_sa_data(
        tokenizer=tokenizer,
        data_name=data_name
    )
    preferred_data = data['train'][:]['non_oversight_input']
    non_preferred_data = data['train'][:]['oversight_input']

    # extracting the representations
    all_hidden_states = vanilla_hidden_state_extraction(
        llm=llm,
        tokenizer=tokenizer,
        texts=preferred_data + non_preferred_data,
        batch_size=batch_size,
        append_last_only=True
    )
    positive_index = list(range(len(preferred_data)))
    negative_index = list(range(len(preferred_data), len(preferred_data) + len(non_preferred_data)))

    # getting the vector
    if approach == 'diffmean':
        statistical_vector = extract_diffmean_control_vector(
            all_hidden_states=all_hidden_states,
            positive_indices=positive_index,
            negative_indices=negative_index,
            token_pos=-1
        )
    elif approach == 'lat':
        statistical_vector = extract_lat_control_vector(
            all_hidden_states=all_hidden_states,
            positive_indices=positive_index,
            negative_indices=negative_index,
            use_positive_only=False,
            token_pos=-1
        )
    elif approach == 'pca':
        statistical_vector = extract_pca_control_vector(
            all_hidden_states=all_hidden_states,
            positive_indices=positive_index,
            negative_indices=negative_index,
            method='diff',
            token_pos=-1
        )

    else:
        raise ValueError(f'Approach {approach} not recognized.')

    # saving the vector
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    statistical_vector.export_gguf(output_file)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='olmo2-7b-instruct', choices=list(model_dictionary.keys()))
    parser.add_argument('--data_name', type=str, default='sorry_bench_sa', choices=['sorry_bench_sa', 'wmdp_sa'])
    parser.add_argument('--approach', type=str, choices=['pca', 'lat', 'diffmean'], default='diffmean')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str, default='training_output/olmo2-7b-instruct/diffmean_sorry_bench_sa.gguf')
    args = parser.parse_args()  

    pipeline_rep_extractor_wo_response(
        model_name=args.model_name,
        data_name=args.data_name,
        approach=args.approach,
        batch_size=args.batch_size,
        output_file=args.output_file
    )