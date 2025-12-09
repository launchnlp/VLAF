import os
import json
import argparse
from tqdm import tqdm
from typing import List
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader
from prompter.user_prompt import create_normal_user_prompt, create_normal_user_prompt_from_refined
from data.dataloader import load_advbench_data, data_configuration, load_redwood_data, load_mft_data, load_mft_refined_data

def pipeline_inference(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    data: List[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    n: int,
    output_file: str,
    reverse:  bool = False,
):
    '''
        Main pipeline for inference using VLLM to assess alignment faking
    '''

    # loading the model using the universal_model_loader
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
    data_samples = []
    data_type = 'mft'
    for data_item in data:
        if data_item not in data_configuration:
            raise ValueError(f'Data {data_item} not found in the configuration')

        if data_item == 'advbench':
            data_samples.extend(load_advbench_data())
        elif data_item in ['redwood_animal_welfare', 'redwood_helpful']:
            data_samples.extend(load_redwood_data(data_item))
        elif data_item.startswith('mft_refined'):
            data_samples.extend(load_mft_refined_data(data_item))
            data_type = 'mft_refined'
        elif data_item.startswith('mft'):
            data_samples.extend(load_mft_data(data_item))
        else:
            raise ValueError(f'Data {data_item} loading function not implemented')

    # Inference loop
    response_data = []
    for i in tqdm(range(0, len(data_samples), batch_size), desc='Inference'):

        # parsing the batch of messages
        batch_samples = data_samples[i: i + batch_size]
        batch_input = []
        message_list = []
        for sample in batch_samples:

            # creating the user prompt
            if data_type == 'mft_refined':
                user_prompt = create_normal_user_prompt_from_refined(
                    content=sample,
                    reverse=reverse
                )
            else:
                user_prompt = create_normal_user_prompt(
                    content=sample
                )
            
            # creating the message
            messages = [
                {'role': 'system', 'content': system_prompt_text},
                {'role': 'user', 'content': user_prompt}
            ]
            for _ in range(n):
                message_list.append(messages)
            batch_input.append(user_prompt)

        # getting the inference using llm
        if 'gpt-5' in model_name or 'o4' in model_name:
            response_batch = llm.batch(
                message_list,
                max_tokens=max_tokens,
            )
        else:
            response_batch = llm.batch(
                message_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

        # converting response batch into list of lists where each sublist corresponds to a single input's responses
        response_batch = [response_batch[j:j + n] for j in range(0, len(response_batch), n)]

        # collecting the response and updating the response data
        for j, response in enumerate(response_batch):
            response_texts = [resp.content for resp in response]
            response_data.append({
                'input': batch_input[j],
                'raw_input': batch_samples[j],
                'response': response_texts
            })

    # saving the response data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(response_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for inference using VLLM to assess alignment faking')
    parser.add_argument('--model_name', type=str, default='openai:gpt-4o')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='mft_no_faking')
    parser.add_argument('--data', type=str, nargs='+', help='List of data samples to use for inference', default=['advbench'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--reverse', action='store_true', help='Flag to indicate if the inference is in training stage')
    parser.add_argument('--output_file', type=str, default='results/advbench/helpful_only/openai:gpt-4o_False.json')
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        data=args.data,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n=args.n,
        reverse=args.reverse,
        output_file=args.output_file
    )