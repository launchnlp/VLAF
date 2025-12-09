import os
import torch
import argparse
from utils.vllm_utils import model_dictionary
from prompter.system_prompt import SystemPrompter
from transformers import AutoTokenizer, AutoModelForCausalLM
from rep_extraction.data_processing import pipeline_tokenize_data
from rep_extraction.hidden_state_extractor import vanilla_hidden_state_extraction
from easysteer.steer import extract_diffmean_control_vector, extract_lat_control_vector, extract_pca_control_vector


def pipeline_rep_extractor(
    model_name: str,
    system_prompt: str,
    data_name: str,
    user_label: str,
    oversight_response_file: str,
    non_oversight_response_file: str,
    split: str,
    approach: str,
    batch_size: int = 8,
    output_file: str = 'af_representation.gguf'
) -> None:
    ''' 
        Pipeline to extract representations to capture alignment faking
    '''

    # loading the model and tokenizer
    model_info = model_dictionary[model_name]
    llm = AutoModelForCausalLM.from_pretrained(model_info['model'], device_map='auto', dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_info['model'])

    # creating the system prompt
    system_prompt_str = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt,
        company=model_info['company']
    )

    # creating the dataset
    preferred_data, non_preferred_data = pipeline_tokenize_data(
        data_name=data_name,
        user_label=user_label,
        oversight_response_file=oversight_response_file,
        non_oversight_response_file=non_oversight_response_file,
        tokenizer=tokenizer,
        system_prompt=system_prompt_str,
        split=split
    )

    # extracting the representations
    all_hidden_states = vanilla_hidden_state_extraction(
        llm=llm,
        tokenizer=tokenizer,
        texts=preferred_data + non_preferred_data,
        batch_size=8
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
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='mft_care_strong')
    parser.add_argument('--data_name', type=str, choices=[
        'mft_refined_authority',
        'mft_refined_care',
        'mft_refined_fairness',
        'mft_refined_loyalty',
        'mft_refined_sanctity',
        'mft_refined_authority_final',
        'mft_refined_care_final',
        'mft_refined_fairness_final',
        'mft_refined_loyalty_final',
        'mft_refined_sanctity_final',
    ], default='mft_refined_care')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    parser.add_argument('--oversight_response_file', type=str, default='results/MFT_refined/care/vllm:olmo2-7b-instruct_True.json')
    parser.add_argument('--non_oversight_response_file', type=str, default='results/MFT_refined/care/vllm:olmo2-7b-instruct_False.json')
    parser.add_argument('--split', type=str, choices=['all', 'oversight', 'non_oversight'], default='all')
    parser.add_argument('--approach', type=str, choices=['pca', 'lat', 'diffmean'], default='diffmean')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str, default='results/af_representation/olmo2-7b-instruct_mft_refined_care_diffmean.gguf')
    args = parser.parse_args()  

    pipeline_rep_extractor(
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        data_name=args.data_name,
        user_label=args.user_label,
        oversight_response_file=args.oversight_response_file,
        non_oversight_response_file=args.non_oversight_response_file,
        split=args.split,
        approach=args.approach,
        batch_size=args.batch_size,
        output_file=args.output_file
    )