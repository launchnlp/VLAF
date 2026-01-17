'''
    Iterate over all the files in a particular folder and create a separate file in a target folder with identical path by filtering samples based on bit mask
'''

import os
import json
import argparse
from typing import List

def filter_sample(
    input_file: str,
    output_file: str,
    bit_mask: List[int]
) -> None:
    '''
        Filter samples in a single file based on a bit mask and save to output file.
    '''

    # Load the input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # check whether input data length matches bit mask length
    if len(data) != len(bit_mask):
        raise ValueError(f"Length of data ({len(data)}) does not match length of bit mask ({len(bit_mask)})")

    # Filter samples based on bit mask
    filtered_data = [
        sample for idx, sample in enumerate(data)
        if bit_mask[idx] == 1
    ]

    # Save the filtered data to output file
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

def filter_samples_by_bit_mask(
    input_folder: str,
    output_folder: str,
    bit_mask_path: str,
    general_mode: bool = False
) -> None:
    '''
        Filter samples in files based on a bit mask and save to output folder.
    '''

    # creating the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the bit mask
    with open(bit_mask_path, 'r') as f:
        bit_mask = json.load(f)

    # Iterate over all files in the input folder
    for file in os.listdir(input_folder):

        try:
            # check if the file ends with True.json or False.json
            if not general_mode:
                if not (file.endswith('True.json') or file.endswith('False.json')):
                    continue

            # Construct full file path
            input_file_path = os.path.join(input_folder, file)
            output_file_path = os.path.join(output_folder, file)

            # Filter samples and save to output file
            filter_sample(
                input_file=input_file_path,
                output_file=output_file_path,
                bit_mask=bit_mask
            )

            # logging the progress
            print(f"Filtered samples from {input_file_path} and saved to {output_file_path}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter samples based on bit mask')
    parser.add_argument('--input_folder', type=str, default='results/MFT_refined_final_ra/fairness')
    parser.add_argument('--output_folder', type=str, default='results/MFT_refined_50_ra/fairness')
    parser.add_argument('--bit_mask', type=str, default='data/mft_refined/fairness_50_bit_mask.json')
    parser.add_argument('--general_mode', action='store_true', help='Whether to use general mode or not')
    args = parser.parse_args()

    filter_samples_by_bit_mask(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        bit_mask_path=args.bit_mask,
        general_mode=args.general_mode
    )