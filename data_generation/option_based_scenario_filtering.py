import os
import json
import argparse

def pipeline_option_based_scenario_filtering(
    input_dir: str,
    output_file: str,
    model_list: list,
    threshold: float
):
   
    # loading all the data corresponding to each model
    model_data_mapper = {}
    length_list = []
    for model in model_list:

        # normal loading
        model_file_path = os.path.join(
            input_dir, f'{model}.json'
        )
        with open(model_file_path, 'r') as f:
            model_data_mapper[model] = json.load(f)
        length_list.append(len(model_data_mapper[model]))

        # reverse loading
        reverse_model_file_path = os.path.join(
            input_dir, f'{model}_reverse.json'
        )
        with open(reverse_model_file_path, 'r') as f:
            model_data_mapper[model + '_reverse'] = json.load(f)
        length_list.append(len(model_data_mapper[model + '_reverse']))

    # checking for consistency in lengths
    assert len(set(length_list)) == 1, "Inconsistent lengths among model data files."
    total_elements = length_list[0]


    # filtering based on the threshold
    filtered_results = []
    for index in range(total_elements):
        score = []
        for model in model_list:
            model_item = model_data_mapper[model][index]
            reverse_model_item = model_data_mapper[model + '_reverse'][index]
            normal_score = [int(val == 'b' or val == '2') for val in model_item['option_selection_evaluation']]
            reverse_score = [int(val == 'a' or val == '1') for val in reverse_model_item['option_selection_evaluation']]
            normal_score = sum(normal_score) / len(normal_score)
            reverse_score = sum(reverse_score) / len(reverse_score)
            score.append((normal_score + reverse_score) / 2)

        if all(s >= threshold for s in score):
            filtered_results.append(model_data_mapper[model_list[0]][index]['raw_input'])

    # printing the fraction of filtered scenarios
    print(f"Filtered {len(filtered_results)} out of {total_elements} scenarios. Fraction: {len(filtered_results)/total_elements:.2f}")

    # saving the filtered results
    with open(output_file, 'w') as f:
        json.dump(filtered_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Option Based Scenario Filtering Pipeline')
    parser.add_argument(
        '--input_dir', '-id', type=str, default='results/MFT_refined_generations/authority'
    ),
    parser.add_argument(
        '--output_file', '-of', type=str, default='data/mft_refined/authority_filtered.json'
    )
    parser.add_argument(
        '--model_list', '-ml', type=str, nargs='+', default=['openai:gpt-5', 'vllm:olmo2-13b-instruct', 'vllm:Qwen2.5-14B-Instruct']
    )
    parser.add_argument(
        '--threshold', '-th', type=float, default=0.95
    )
    args = parser.parse_args()

    pipeline_option_based_scenario_filtering(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_list=args.model_list,
        threshold=args.threshold
    )
