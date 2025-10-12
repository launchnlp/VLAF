'''
    Generation of explicit alignment faking scenarios
'''
import os
import json
import argparse
from tqdm import tqdm
from prompter.system_prompt import SystemPrompter
from prompter.user_prompt import create_user_prompt
from utils.langchain_utils import universal_model_loader
from data.dataloader import load_advbench_data, data_configuration, load_redwood_data

def explicit_data_generation(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    data: str,
    output_file: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    user_label: str = 'is_free_tier_user',
) -> None:
    '''
        Creating alignment faking responses for scenarios in the specified dataset
    '''

    # Load the model
    llm, organization = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )

    # Load the system prompt
    system_prompt_text = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt,
        company=organization
    )

    # Load the data
    if data not in data_configuration:
        raise ValueError(f'Data {data} not found in the configuration')

    # creating the output directory if not exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if data == 'advbench':
        data_samples = load_advbench_data()
    elif data in ['redwood_animal_welfare', 'redwood_helpful']:
        data_samples = load_redwood_data(data)
    else:
        raise ValueError(f'Data {data} loading function not implemented')
    
    # Inference loop
    response_data = []
    for i in tqdm(range(0, len(data_samples), batch_size), desc='Inference'):
            
        # parsing the batch of messages
        batch_samples = data_samples[i: i + batch_size]
        batch_input = []
        message_list = []
        for sample in batch_samples:
            for condition in [True, False]:
                user_prompt = create_user_prompt(
                    content=sample,
                    condition=condition,
                    user_label=user_label
                )
                messages = [
                    {'role': 'system', 'content': system_prompt_text},
                    {'role': 'user', 'content': user_prompt}
                ]
                message_list.append(messages)
                batch_input.append(user_prompt)

        # getting the inference
        if 'gpt-5' in llm.model_name or 'o4' in llm.model_name:
            responses = llm.batch(
                message_list,
                max_tokens=max_tokens
            )
        else:   
            responses = llm.batch(
                message_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
        
        # post process the responses
        for j, response in enumerate(responses):
            condition = j % 2 == 0
            input_message = batch_input[j]
            response_text = response.content
            response_data.append({
                'input': input_message,
                'response': response_text,
                'condition': condition
            })

    # creating the directory if not exists
    with open(output_file, 'w') as f:
        json.dump(response_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate explicit alignment faking scenarios')
    parser.add_argument('--model_name', type=str, default='openai:gpt-4.1')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='intentional_alignment_faking_examples')
    parser.add_argument('--data', type=str, choices=list(data_configuration.keys()), default='redwood_helpful')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str, default='synthetic_data/explicit_af/intentional_alignment_faking_examples_gpt-4.1.json')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    args = parser.parse_args()

    explicit_data_generation(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        data=args.data,
        output_file=args.output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        user_label=args.user_label,
    )