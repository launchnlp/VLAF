import json
from transformers import AutoTokenizer
from typing import Literal, List, Tuple
from data.dataloader import load_mft_refined_data
from custom_typing.scenario import AFScenarioResponse
from prompter.user_prompt import create_user_prompt_from_refined

def convert_to_model_input(
    system_prompt: str,
    user_prompt: str,
    response: str,
    tokenizer: AutoTokenizer
) -> str:
    '''
        Converting the data into model input format
    '''
    
    # creating the message object
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
        {'role': 'assistant', 'content': response}
    ]

    # tokenizing the message object
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def create_af_scenario_response(
    data_name: Literal[
        'mft_refined_authority',
        'mft_refined_care',
        'mft_refined_fairness',
        'mft_refined_loyalty',
        'mft_refined_sanctity',
        'mft_refined_authority_final',
        'mft_refined_care_final',
        'mft_refined_fairness_final',
        'mft_refined_loyalty_final',
        'mft_refined_sanctity_final',
    ],
    user_label: str,
    oversight_response_file: str,
    non_oversight_response_file: str
) -> List[AFScenarioResponse]:
    '''
        Parsing the data into an appropriate format
    '''

    # loading the dataset
    scenario_list = load_mft_refined_data(data_name)

    # loading the oversight and non_oversight
    with open(oversight_response_file, 'r') as f:
        oversight_data = json.load(f)
    with open(non_oversight_response_file, 'r') as f:
        non_oversight_data = json.load(f)

    # creating a list of data output
    output = []
    for scenario, oversight_dp, non_oversight_dp in zip(scenario_list, oversight_data, non_oversight_data):

        # check whether the scenario and dps' scenario matches
        oversight_scenario = create_user_prompt_from_refined(
            scenario, True, user_label
        )
        non_oversight_scenario = create_user_prompt_from_refined(
            scenario, False, user_label
        )

        assert oversight_dp['input'] == oversight_scenario, "Scenario mismatch between scenario and oversight_dp"
        assert non_oversight_dp['input'] == non_oversight_scenario, "Scenario mismatch between scenario and non_oversight_dp"

        # first adding the oversight_dp
        developer_compliant_responses = []
        developer_noncompliant_responses = []
        for response, option in zip(oversight_dp['response'], oversight_dp['option_selection_evaluation']):
            if option == 'a' or option == '1':
                developer_compliant_responses.append(response)
            else:
                developer_noncompliant_responses.append(response)
        output.append({
            'scenario': scenario['scenario'],
            'developer_compliant': scenario['developer_compliant'],
            'developer_noncompliant': scenario['developer_noncompliant'],
            'user_label': user_label,
            'oversight_condition': True,
            'developer_compliant_responses': developer_compliant_responses,
            'developer_noncompliant_responses': developer_noncompliant_responses
        })

        # then adding the non_oversight_dp
        developer_compliant_responses = []
        developer_noncompliant_responses = []
        for response, option in zip(non_oversight_dp['response'], non_oversight_dp['option_selection_evaluation']):
            if option == 'a' or option == '1':
                developer_compliant_responses.append(response)
            else:
                developer_noncompliant_responses.append(response)
        output.append({
            'scenario': scenario['scenario'],
            'developer_compliant': scenario['developer_compliant'],
            'developer_noncompliant': scenario['developer_noncompliant'],
            'user_label': user_label,
            'oversight_condition': False,
            'developer_compliant_responses': developer_compliant_responses,
            'developer_noncompliant_responses': developer_noncompliant_responses
        })

    return output


def tokenize_data(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    data: List[AFScenarioResponse]
) -> Tuple[List[str], List[str]]:
    '''
        Tokenizing the data into model input format
    '''

    # creating lists to store the tokenized data
    preferred_responses = []
    non_preferred_responses = []

    # iterating through the data
    for dp in data:

        user_prompt = create_user_prompt_from_refined(
            content={
                'scenario': dp['scenario'],
                'developer_compliant': dp['developer_compliant'],
                'developer_noncompliant': dp['developer_noncompliant']
            },
            condition=dp['oversight_condition'],
            user_label=dp['user_label']
        )
        
        # check if the dp is oversight
        if dp['oversight_condition']:
            preferred_dp = dp['developer_compliant_responses']
            non_preferred_dp = dp['developer_noncompliant_responses']

        # if the dp is non-oversight
        else:
            preferred_dp = dp['developer_noncompliant_responses']
            non_preferred_dp = dp['developer_compliant_responses']
        min_len = min(len(preferred_dp), len(non_preferred_dp))

        # for each response, convert to model input
        for i in range(min_len):
            preferred_responses.append(
                convert_to_model_input(
                    system_prompt,
                    user_prompt,
                    preferred_dp[i],
                    tokenizer
                )
            )
            non_preferred_responses.append(
                convert_to_model_input(
                    system_prompt,
                    user_prompt,
                    non_preferred_dp[i],
                    tokenizer
                )
            )

    return preferred_responses, non_preferred_responses

def pipeline_tokenize_data(
    data_name: Literal[
        'mft_refined_authority',
        'mft_refined_care',
        'mft_refined_fairness',
        'mft_refined_loyalty',
        'mft_refined_sanctity',
        'mft_refined_authority_final',
        'mft_refined_care_final',
        'mft_refined_fairness_final',
        'mft_refined_loyalty_final',
        'mft_refined_sanctity_final',
    ],
    user_label: str,
    oversight_response_file: str,
    non_oversight_response_file: str,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    split: Literal['all', 'oversight', 'non_oversight'] = 'all'
) -> Tuple[List[str], List[str]]:
    '''
        Pipeline to create and tokenize the data
    '''

    # creating the af scenario response
    data = create_af_scenario_response(
        data_name,
        user_label,
        oversight_response_file,
        non_oversight_response_file
    )
    
    # filtering the data based on the split
    if split == 'oversight':
        data = [dp for dp in data if dp['oversight_condition']]
    elif split == 'non_oversight':
        data = [dp for dp in data if not dp['oversight_condition']]

    # tokenizing the data
    return tokenize_data(
        tokenizer,
        system_prompt,
        data
    )

    

if __name__ == '__main__':
    data = create_af_scenario_response(
        'mft_refined_fairness',
        'is_free_tier_user',
        'results/MFT_refined/fairness/vllm:olmo2-7b-instruct_True.json',
        'results/MFT_refined/fairness/vllm:olmo2-7b-instruct_False.json'
    )
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-2-1124-7B-Instruct', trust_remote_code=True)
    parsed_output = tokenize_data(
        tokenizer,
        'Nothing',
        data
    )