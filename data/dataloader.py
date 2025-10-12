'''
    Loading data prompts for alignment faking evaluation
'''
import csv
import json
from typing import List

data_configuration = {
    'advbench': {
        'path': 'data/harmful_behaviors.csv'
    },
    'redwood_animal_welfare': {
        'path': 'data/redwood_animal_welfare.json'
    },
    'redwood_helpful': {
        'path': 'data/redwood_helpful.json'
    },
    'mft_authority': {
        'path': 'data/mft_authority.json'
    },
    'mft_care': {
        'path': 'data/mft_care.json'
    },
    'mft_fairness': {
        'path': 'data/mft_fairness.json'
    },
    'mft_loyalty': {
        'path': 'data/mft_loyalty.json'
    },
    'mft_sanctity': {
        'path': 'data/mft_sanctity.json'
    }
}

def load_advbench_data() -> List[str]:
    '''
        Load the advbench data while skipping the header and returning the first column
    '''
    data = []
    with open(data_configuration['advbench']['path'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row[0])
    return data

def load_redwood_data(data_name: str) -> List[str]:
    '''
        Load the redwood data from the specified json file
    '''
    if data_name not in data_configuration:
        raise ValueError(f"Data '{data_name}' not found.")
    path = data_configuration[data_name]['path']
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_mft_data(data_name: str) -> List[str]:
    '''
        Load the mft data from the specified json file
    '''
    
    # check whether the attribute is present
    if data_name not in data_configuration:
        raise ValueError(f"Data '{data_name}' not found.")
    path = data_configuration[data_name]['path']
    with open(path, 'r') as f:
        data = json.load(f)
    return data