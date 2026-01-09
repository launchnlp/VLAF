import os
import json
import argparse
from tqdm import tqdm
from typing import Optional
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader
from prompter.user_prompt import create_user_prompt, create_user_prompt_from_refined
from data.dataloader import load_advbench_data, data_configuration, load_redwood_data, load_mft_data, load_mft_refined_data

def pipeline_inference(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    prefilling_template: str,
    data: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    n: int,
    condition: bool,
    output_file: str,
    user_label: str = 'is_free_tier_user',
    tier_name: str = 'Paid Tier',
    probability: Optional[str] = None
):
    '''
        Main pipeline for inference using VLLM to assess alignment faking
    '''

    assert 'vllm' in model_name, 'This pipeline only supports VLLM models.'

    # loading the model using the universal_model_loader
    llm, organization = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )
    
    # Load the system prompt
    system_prompt_text = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt,
        company=organization,
        probability=probability
    )
    prefill_template_text = SystemPrompter.create_system_prompt(
        system_prompt_name=prefilling_template,
        company=organization,
        tier=tier_name
    )

    # Load the data
    if data not in data_configuration:
        raise ValueError(f'Data {data} not found in the configuration')
    
    if data == 'advbench':
        data_samples = load_advbench_data()
    elif data in ['redwood_animal_welfare', 'redwood_helpful']:
        data_samples = load_redwood_data(data)
    elif data.startswith('mft_refined'):
        data_samples = load_mft_refined_data(data)
    elif data.startswith('mft'):
        data_samples = load_mft_data(data)

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

            # creating the user prompt
            if data.startswith('mft_refined'):
                user_prompt = create_user_prompt_from_refined(
                    content=sample,
                    condition=condition,
                    user_label=user_label,
                    reverse=False
                )
            else:
                user_prompt = create_user_prompt(
                    content=sample,
                    condition=condition,
                    user_label=user_label
                )
            
            # creating the message
            message = '{}\n\n{}\n\n{}'.format(
                system_prompt_text,
                user_prompt,
                prefill_template_text
            )
            for _ in range(n):
                message_list.append(message)
            batch_input.append(user_prompt)

        # getting the inference using llm
        response_batch = llm.generate(
            message_list,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # converting response batch into list of lists where each sublist corresponds to a single input's responses
        response_batch = [response_batch[j:j + n] for j in range(0, len(response_batch), n)]

        # collecting the response and updating the response data
        for j, response in enumerate(response_batch):
            response_texts = [(prefill_template_text + resp.outputs[0].text) for resp in response]
            response_data.append({
                'input': batch_input[j],
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
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='helpful_only')
    parser.add_argument('--prefilling_template', type=str, default='however_prefill', choices=list(SystemPrompter.configuration.keys()))
    parser.add_argument('--data', type=str, choices=list(data_configuration.keys()), default='advbench')
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--condition', action='store_true', help='Flag to indicate if the inference is in training stage')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    parser.add_argument('--output_file', type=str, default='results/advbench/helpful_only/openai:gpt-4o_False.json')
    parser.add_argument('--tier_name', type=str, default='Paid Tier')
    parser.add_argument('--probability', type=str, default=None, help='Probability of oversight')
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        prefilling_template=args.prefilling_template,
        data=args.data,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n=args.n,
        condition=args.condition,
        output_file=args.output_file,
        user_label=args.user_label,
        tier_name=args.tier_name,
        probability=args.probability
    )