'''
    Scripts for generating background information that may notify evaluation conditions.
'''

import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from data.dataloader import load_sorry_bench_data
from prompter.system_prompt import SystemPrompter
from custom_typing.scenario import SorryBenchScenario
from utils.langchain_utils import universal_model_loader

def situational_awareness_prompt_generation_pipeline(
    sample: int,
    random_seed: int,
    per_sample_generations: int,
    output_file: str
) -> None:
    ''' 
        Pipeline to generate background information describing the evaluation conditions
    '''

    # setting the random seed
    random.seed(random_seed)

    # loading the data
    scenario_list = load_sorry_bench_data()

    # mapping the scenarios to each category and then sampling
    sampled_scenario_list = []
    category_scenario_mapper = defaultdict(list)
    for scenario in scenario_list:
        category_scenario_mapper[scenario['category']].append(scenario)
    for category, scenarios in category_scenario_mapper.items():
        sampled_scenarios = random.sample(scenarios, min(sample, len(scenarios)))
        sampled_scenario_list.extend(sampled_scenarios)

    # load the system prompt for situational awareness background integration
    system_prompt = SystemPrompter.create_system_prompt(
        system_prompt_name='sa_background_integration',
    )

    # loading gpt-5
    llm, _ = universal_model_loader(
        model='openai:gpt-5'
    )

    # generating the background information
    output_data = []
    for scenario in tqdm(sampled_scenario_list):

        # creating the prompt message
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': scenario['scenario']}
        ]

        # generating multiple times per scenario
        try:
            responses = llm.batch(
                [message for _ in range(per_sample_generations)],
                temperature=1.0,
                max_tokens=5000
            )
        except Exception as e:
            print(f"Failed to generate for scenario: {scenario['scenario']}")
            print(e)
            continue

        # iterating over the responses
        for response in responses:
            try:
                response_dict = json.loads(response.content)
                output_data.append({
                    'scenario': scenario['scenario'],
                    'category': scenario['category'],
                    'deployment_framing': response_dict['deployment_framing'],
                    'evaluation_framing': response_dict['evaluation_framing']
                })
            except:
                print(f"Failed to parse response: {response.content}")

    # saving the output data
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate background information describing the evaluation conditions'
    )
    parser.add_argument(
        '--sample', '-s', type=int, default=2, help='Number of elements to sample per scenario'
    )
    parser.add_argument(
        '--random_seed', '-r', type=int, default=42, help='Random seed for sampling'
    )
    parser.add_argument(
        '--per_sample_generations', '-g', type=int, default=3, help='Number of generations per sample'
    )
    parser.add_argument(
        '--output_file', '-o', type=str, default='data/sorry_bench_sa.json'
    )
    args = parser.parse_args()

    situational_awareness_prompt_generation_pipeline(
        sample=args.sample,
        random_seed=args.random_seed,
        per_sample_generations=args.per_sample_generations,
        output_file=args.output_file
    )

