import os
import json
import argparse
from typing import List
from data.dataloader import load_mft_refined_data
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader

def thinking_step_stripper(response: str) -> str:
    '''
        Removes the <think>...</think> tags from the response string
    '''
    while '<think>' in response and '</think>' in response:
        start_idx = response.index('<think>')
        end_idx = response.index('</think>') + len('</think>')
        response = response[:start_idx] + response[end_idx:]
    return response.strip()

def pipeline_data_categorization(
    input_file: List[str],
    model_name: str,
    tensor_parallel_size: int,
    taxonomy_induction_prompt: str,
    taxonomy_categorization_prompt: str,
    output_file: str
) -> None:
    '''
        Inducing taxonomy and assigning the categories to each of the data points
    '''

    # initializing the model
    llm, _ = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )
    
    scenario_list = []
    for data_name in input_file:
        scenarios = load_mft_refined_data(data_name)
        scenario_list.extend(scenarios)

    # collecting all the scenarios
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            taxonomy_data = json.load(f)
            categories = taxonomy_data['categories']
    else:
        # inducing the taxonomy from the data
        induction_prompt = SystemPrompter.create_system_prompt(taxonomy_induction_prompt)
        scenario_text = "\n".join([f"- {scenario['scenario']}" for scenario in scenario_list])
        message = [
            {
                'role': 'system',
                'content': induction_prompt
            },
            {
                'role': 'user',
                'content': f"Here are some scenarios:\n{scenario_text}\n\nPlease induce a taxonomy of categories that can be used to classify these scenarios."
            }
        ]
        
        # getting the inference using llm
        if 'gpt-5' in model_name or 'o4' in model_name:
            response_batch = llm.batch(
                [message],
                max_tokens=2048,
            )
        else:
            response_batch = llm.batch(
                [message],
                temperature=0.0,
                top_p=0.95,
                max_tokens=2048,
            )

        categories = response_batch[0].content
        categories = thinking_step_stripper(categories)

        # saving the intermediate taxonomy data
        taxonomy_data = {
            'categories': categories
        }
        with open(output_file, 'w') as f:
            json.dump(taxonomy_data, f, indent=4)

    # loading the categorization prompt
    categorization_prompt = SystemPrompter.create_system_prompt(
        taxonomy_categorization_prompt,
        categories=categories
    )

    # iterating over the scenarios
    categorized_scenarios = []
    for scenario in scenario_list:

        # creating the message
        message = [
            {
                'role': 'system',
                'content': categorization_prompt
            },
            {
                'role': 'user',
                'content': f"Scenario: {scenario['scenario']}"
            }
        ]

        # getting the inference using llm
        if 'gpt-5' in model_name or 'o4' in model_name:
            response_batch = llm.batch(
                [message],
                max_tokens=512,
            )
        else:
            response_batch = llm.batch(
                [message],
                temperature=0.0,
                top_p=0.95,
                max_tokens=512,
            )

        # extracting the category assignment
        category_assignment = response_batch[0].content
        category_assignment = thinking_step_stripper(category_assignment)
        categorized_scenarios.append({
            'scenario': scenario['scenario'],
            'category': category_assignment
        })

        # log progress
        print(f"Categorized scenario: {scenario['scenario']}\nAssigned Category: {category_assignment}\n")

    # saving the categorized scenarios
    final_output = {
        'categories': categories,
        'categorized_scenarios': categorized_scenarios
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Categorization Script')
    parser.add_argument('--input_file', nargs='+', default=[
        'mft_refined_authority_50',
        'mft_refined_care_50',
        'mft_refined_fairness_50',
        'mft_refined_loyalty_50',
        'mft_refined_sanctity_50'
    ])
    parser.add_argument('--model_name', type=str, default='openai:gpt-5')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--taxonomy_induction_prompt', type=str, default='taxonomy_induction')
    parser.add_argument('--taxonomy_categorization_prompt', type=str, default='taxonomy_categorization')
    parser.add_argument('--output_file', type=str, default='data/mft_categories.json')
    args = parser.parse_args()

    pipeline_data_categorization(
        input_file=args.input_file,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        taxonomy_induction_prompt=args.taxonomy_induction_prompt,
        taxonomy_categorization_prompt=args.taxonomy_categorization_prompt,
        output_file=args.output_file
    )
