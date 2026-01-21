'''
    Loading data prompts for alignment faking evaluation
'''
import os
import csv
import json
import jsonlines
from typing import List, Any
from data.sorry_bench_meta_info import category_descriptions_shortest
from custom_typing.scenario import AFScenario, SorryBenchScenario, SorryBenchSAScenario

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
    },
    'mft_refined_authority': {
        'path': 'data/mft_refined/authority.json'
    },
    'mft_refined_care': {
        'path': 'data/mft_refined/care.json'
    },
    'mft_refined_fairness': {
        'path': 'data/mft_refined/fairness.json'
    },
    'mft_refined_loyalty': {
        'path': 'data/mft_refined/loyalty.json'
    },
    'mft_refined_sanctity': {
        'path': 'data/mft_refined/sanctity.json'
    },
    'mft_refined_authority_generations': {
        'path': 'data/mft_refined/authority_generations.json'
    },
    'mft_refined_care_generations': {
        'path': 'data/mft_refined/care_generations.json'
    },
    'mft_refined_fairness_generations': {
        'path': 'data/mft_refined/fairness_generations.json'
    },
    'mft_refined_loyalty_generations': {
        'path': 'data/mft_refined/loyalty_generations.json'
    },
    'mft_refined_sanctity_generations': {
        'path': 'data/mft_refined/sanctity_generations.json'
    },
    'mft_refined_authority_final': {
        'path': 'data/mft_refined/authority_final.json'
    },
    'mft_refined_care_final': {
        'path': 'data/mft_refined/care_final.json'
    },
    'mft_refined_fairness_final': {
        'path': 'data/mft_refined/fairness_final.json'
    },
    'mft_refined_loyalty_final': {
        'path': 'data/mft_refined/loyalty_final.json'
    },
    'mft_refined_sanctity_final': {
        'path': 'data/mft_refined/sanctity_final.json'
    },
    'mft_refined_authority_50': {
        'path': 'data/mft_refined/authority_50.json'
    },
    'mft_refined_care_50': {
        'path': 'data/mft_refined/care_50.json'
    },
    'mft_refined_fairness_50': {
        'path': 'data/mft_refined/fairness_50.json'
    },
    'mft_refined_loyalty_50': {
        'path': 'data/mft_refined/loyalty_50.json'
    },
    'mft_refined_sanctity_50': {
        'path': 'data/mft_refined/sanctity_50.json'
    },
    'sorry_bench': {
        'path': 'data/sorry_bench_question.jsonl'
    },
    'sorry_bench_sa': {
        'path': 'data/sorry_bench_sa.json'
    },
    'strong_reject': {
        'path': 'data/strongreject_dataset.csv'
    },
    'strong_reject_50': {
        'path': 'data/strong_reject_50.json'
    },
    'sad_stages': {
        'path': 'data/sad_stages'
    },
    'wmdp': {
        'path': 'data/wmdp.json'
    },
    'wmdp_sa': {
        'path': 'data/wmdp_sa.json'
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

def load_redwood_data(data_name: str) -> List[Any]:
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

def load_mft_refined_data(data_name: str) -> List[AFScenario]:
    '''
        Load the mft refined data from the specified json file
    '''
    
    # check whether the attribute is present
    if data_name not in data_configuration:
        raise ValueError(f"Data '{data_name}' not found.")
    path = data_configuration[data_name]['path']
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_sorry_bench_data() -> List[SorryBenchScenario]:
    '''
        Load the SorryBench data from the specified jsonl file
    '''
    
    # loading the data
    path = data_configuration['sorry_bench']['path']
    with jsonlines.open(path, 'r') as reader:
        raw_data = [obj for obj in reader]

    # parsing the data using category_descriptions_shortest
    data = []
    for item in raw_data:
        data.append({
            'scenario': item['turns'][0],
            'category': category_descriptions_shortest[int(item['category']) - 1]
        })


    return data

def load_strong_reject_data() -> List[SorryBenchScenario]:
    '''
        Load the Strong Reject data from the specified jsonl file
    '''
    
    # loading the csv file
    path = data_configuration['strong_reject']['path']
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append({
                'scenario': row[2],
                'category': row[0]
            })

    return data


def load_sorry_bench_sa_data() -> List[SorryBenchSAScenario]:
    '''
        Load the SorryBench SA data from the specified jsonl file
    '''
    
    return load_redwood_data('sorry_bench_sa')

def load_strong_reject_50_data() -> List[SorryBenchScenario]:
    '''
        Load the Strong Reject 50 data from the specified json file
    '''
    
    return load_redwood_data('strong_reject_50')

def load_sad_stages_data() -> List[SorryBenchScenario]:
    '''
        Load the SAD stages data from the specified directory
    '''

    # getting the path
    folder = data_configuration['sad_stages']['path']
    deployment_path = os.path.join(folder, 'deploy_oversight.jsonl')
    test_path = os.path.join(folder, 'test_oversight.jsonl')
    data = []

    # loading deployment_oversight.jsonl
    with jsonlines.open(deployment_path, 'r') as reader:
        for obj in reader:
            data.append({
                'scenario': obj['body'],
                'category': 'deployment'
            })


    # loading test_oversight.jsonl
    with jsonlines.open(test_path, 'r') as reader:
        for obj in reader:
            data.append({
                'scenario': obj['body'],
                'category': 'test'
            })

    return data

def load_wmdp_data() -> List[SorryBenchScenario]:
    '''
        Load the WMDP data from the specified json file
    '''
    
    return load_redwood_data('wmdp')

def load_wmdp_sa_data() -> List[SorryBenchSAScenario]:
    '''
        Load the WMDP SA data from the specified json file
    '''

    return load_redwood_data('wmdp_sa')