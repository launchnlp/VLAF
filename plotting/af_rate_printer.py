'''
    Given a list of file names, print their contents as a CSV output. The format of the CSV is shown below:
    Model, Oversight, Non-oversight, Gap
    ..., ..., ..., ...
'''

import os
import csv
import json
import argparse
from typing import List

def alignment_faking_rate_printer(
    input_folder: str,
    model_names: List[str],
    file_suffix: str
) -> None:
    '''
        Given a list of file names, print their contents as a CSV output. The format of the CSV is shown below:
        Model, Oversight, Non-oversight, Gap
    '''

    # creating the output file name
    output_file_name = os.path.join(
        input_folder,
        'af_rate_summary.csv'
    )

    # opening the output file
    with open(output_file_name, mode='w', newline='') as output_file:

        csv_writer = csv.writer(output_file)
        # writing the header
        csv_writer.writerow(['Model', 'Oversight', 'Non-oversight', 'Gap'])

        # iterating over each model name
        for model_name in model_names:
            # creating the input file name
            input_file_name = os.path.join(
                input_folder,
                f'{model_name}{file_suffix}'
            )

            # checking if the input file exists
            if not os.path.exists(input_file_name):
                print(f'File {input_file_name} does not exist. Skipping...')
                
                # writing empty values for this model
                csv_writer.writerow([model_name, '', '', ''])

                continue

            # reading the input file
            with open(input_file_name, 'r') as input_file:
                
                # loading the json content
                data = json.load(input_file)

                # extracting the required values
                oversight_rate = data['overall']['oversight_compliance_rate']
                non_oversight_rate = data['overall']['non_oversight_compliance_rate']
                gap = data['overall']['compliance_gap']

                # writing the values to the CSV
                csv_writer.writerow([
                    model_name,
                    oversight_rate,
                    non_oversight_rate,
                    gap
                ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print contents of files as CSV.')
    parser.add_argument('--input_folder', '-i', type=str, default='results/MFT_refined_50/authority')
    parser.add_argument('--model_names', '-m', type=str, nargs='+', default=[
        # 'openai:gpt-5',
        # 'vllm:olmo2-7b',
        'vllm:olmo2-7b-instruct',
        # 'vllm:olmo2-13b',
        'vllm:olmo2-13b-instruct',
        # 'vllm:olmo2-32b',
        # 'vllm:olmo2-32b-dpo',
        'vllm:olmo2-32b-instruct',
        # 'vllm:olmo2-32b-sft',
        # 'vllm:Qwen2.5-7B',
        'vllm:Qwen2.5-7B-Instruct',
        # 'vllm:Qwen2.5-14B',
        'vllm:Qwen2.5-14B-Instruct',
        # 'vllm:Qwen2.5-32B',
        'vllm:Qwen2.5-32B-Instruct',
        # 'vllm:Qwen3-8B-Base',
        'vllm:Qwen3-8B',
        # 'vllm:Qwen3-14B-Base',
        'vllm:Qwen3-14B',
        'vllm:Qwen3-32B',
    ])
    parser.add_argument('--file_suffix', '-s', type=str, default='_option_based_compliance.json')
    args = parser.parse_args()

    alignment_faking_rate_printer(
        input_folder=args.input_folder,
        model_names=args.model_names,
        file_suffix=args.file_suffix
    )