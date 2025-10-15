'''
    Code for augmenting a data file with its mft inference
'''
import json
import argparse
import numpy as np
from typing import List, Dict, Any, TypedDict

# create a type for representing mft output
class MFTOutput(TypedDict):
    authority: float
    care: float
    fairness: float
    loyalty: float
    sanctity: float



def mft_data_sanitization(mft_raw_data: Dict[str, Dict[str, Any]]) -> List[float]:
    '''
        Sanitize the raw mft data to extract relevant values
    '''

    # Basic checks
    assert 'Relevance' in mft_raw_data and 'Agreement' in mft_raw_data, "Missing required keys in mft_raw_data"

    # Extracting the values
    values = []
    for key in ['Relevance', 'Agreement']:
        for response in mft_raw_data[key]['responses']:

            # extracting the individual values from the responses
            temp_values = []    
            for val in response:
                val = val.strip()
                if len(val) == 1:
                    temp_values.append(float(val))
                elif val.startswith('[') and val.endswith(']'):
                    num_str = val[1:-1].strip()
                    temp_values.append(float(num_str))
                else:
                    continue

            # averaging the values
            if len(temp_values) > 0:
                avg_value = sum(temp_values) / len(temp_values)
                values.append(avg_value)

    return values

def mft_computation(mft_data: Dict[str, Dict[str, Any]]) -> MFTOutput:
    '''
        Compute the mft values from the raw data
    '''

    # sanitizing the data
    values = mft_data_sanitization(mft_data)

    # computing the different dimensions
    care = np.mean([values[0], values[6], values[11], values[16], values[22], values[27]])
    fairness = np.mean([values[1], values[7], values[12], values[17], values[23], values[28]])
    loyalty = np.mean([values[2], values[8], values[13], values[18], values[24], values[29]])
    authority = np.mean([values[3], values[9], values[14], values[19], values[25], values[30]])
    sanctity = np.mean([values[4], values[10], values[15], values[20], values[26], values[31]])

    # creating the output
    mft_output: MFTOutput = {
        'authority': authority,
        'care': care,
        'fairness': fairness,
        'loyalty': loyalty,
        'sanctity': sanctity
    }

    return mft_output


def mft_value_computation_wrapper(input_file: str) -> None:
    '''
        Augment a data file with its mft inference
    '''
    
    # opening the file
    with open(input_file, 'r') as f:
        mft_raw_data = json.load(f)

    # computing the mft values
    mft_output = mft_computation(mft_raw_data)
    mft_raw_data['mft_values'] = mft_output

    # saving the augmented data
    with open(input_file, 'w') as f:
        json.dump(mft_raw_data, f, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment a data file with its MFT inference')
    parser.add_argument('--input_file', type=str, help='Path to the input data file', default='results/MFT/openai_gpt-4o_mft_results.json')
    args = parser.parse_args()

    mft_value_computation_wrapper(args.input_file)
