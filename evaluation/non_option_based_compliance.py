'''
    Script applicable for redwood examples which has a global notion of developer preferred guidelines and which does not relies on the exact option preferred by the developer.
'''
import json
import argparse
from typing import List
from evaluation.evaluation_faking_computation import compute_compliance_score

def pipeline_non_option_based_compliance(
    oversight_file: str,
    non_oversight_file: str,
    key: str,
    output_file: str,
) -> None:

    # load data from the oversight and non-oversight files
    with open(oversight_file, 'r') as f:
        oversight_data = json.load(f)
    with open(non_oversight_file, 'r') as f:
        non_oversight_data = json.load(f)

    # iterating over the elements of the dataset
    results = {
        'element_wise': [],
        'overall': {
            'oversight_compliance_score': 0.0,
            'non_oversight_compliance_score': 0.0
        }
    }

    # checking the length of both datasets
    if len(oversight_data) != len(non_oversight_data):
        raise ValueError('Oversight and non-oversight files must have the same number of datapoints')

    for oversight_item, non_oversight_item in zip(oversight_data, non_oversight_data):

        # collect the individual values
        oversight_eval: List[str] = oversight_item[key]
        non_oversight_eval: List[str] = non_oversight_item[key]

        # getting the compliance score based on faking computation
        oversight_score_list = []