import json
import argparse
from typing import List, Any
from data.dataloader import load_mft_refined_data

def equal_length_checker(lists: List[List[Any]]) -> bool:
    '''
        Check if all lists have equal length.
    '''
    if not lists:
        return True
    first_length = len(lists[0])
    for lst in lists[1:]:
        if len(lst) != first_length:
            return False
    return True

def pipeline_top_k_scenario_sampler(
    data: str,
    result_file_list: List[str],
    output_file: str,
    output_bit_mask_file: str,
    top_k: int
) -> None:
    '''
        Sample top-K scenarios based on model results.
    '''

    # opening the data by using specific function
    if data.startswith('mft_refined'):
        data_samples = load_mft_refined_data(data)
    else:
        raise ValueError(f'Data {data} loading function not implemented')
    
    # load the model specific results
    model_results = []
    for result_file in result_file_list:
        with open(result_file, 'r') as f:
            model_result = json.load(f)
            model_results.append(model_result['element_wise'])

    # check if all model results have equal length
    if not equal_length_checker(model_results):
        raise ValueError('Model results do not have equal lengths.')
    
    # check if the length of data samples matches the model results
    if len(data_samples) != len(model_results[0]):
        raise ValueError('Data samples and model results length mismatch.')
    
    # computing the sample specific scores, selecting top-k and maintaining bit mask
    scenario_scores = []
    for i in range(len(data_samples)):

        # first check whether the scenario is valid across all the models
        scenario_string = data_samples[i]['scenario']
        for m in range(len(model_results)):
            if scenario_string not in model_results[m][i]['user_input']:
                raise ValueError(f'Scenario string not found in model {m} results at index {i}.')

        score = sum(
            model_results[m][i]['oversight_compliance_rate'] - model_results[m][i]['non_oversight_compliance_rate'] for m in range(len(model_results))
        )
        scenario_scores.append((i, score))

    # sort the scenarios based on scores in descending order
    scenario_scores.sort(key=lambda x: x[1], reverse=True)

    # generate a bit mask for top-k scenarios
    top_k_indices = set(idx for idx, _ in scenario_scores[:top_k])
    bit_mask = [1 if i in top_k_indices else 0 for i in range(len(data_samples))]

    # write the top-k scenarios to output file
    top_k_scenarios = [data_samples[i] for i in top_k_indices]
    with open(output_file, 'w') as f:
        json.dump(top_k_scenarios, f, indent=4)

    # write the bit mask to output file
    with open(output_bit_mask_file, 'w') as f:
        json.dump(bit_mask, f, indent=4)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Top-K scenario sampler script')
    parser.add_argument('--data', type=str, default='mft_refined_sanctity_final')
    parser.add_argument('--result_file_list', type=str, nargs='+', default=[
        'results/MFT_refined_final/sanctity/vllm:Qwen3-8B_option_based_compliance.json',
        'results/MFT_refined_final/sanctity/vllm:Qwen2.5-7B-Instruct_option_based_compliance.json',
        'results/MFT_refined_final/sanctity/vllm:olmo2-7b-instruct_option_based_compliance.json'
    ])
    parser.add_argument('--output_file', type=str, default='data/mft_refined/sanctity_50.json')
    parser.add_argument('--output_bit_mask_file', type=str, default='data/mft_refined/sanctity_50_bit_mask.json')
    parser.add_argument('--top_k', type=int, default=50)
    args = parser.parse_args() 

    pipeline_top_k_scenario_sampler(
        data=args.data,
        result_file_list=args.result_file_list,
        output_file=args.output_file,
        output_bit_mask_file=args.output_bit_mask_file,
        top_k=args.top_k
    )