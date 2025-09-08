import os
import json
import argparse
from tqdm import tqdm
from prompter.system_prompt import SystemPrompter
from prompter.user_prompt import create_user_prompt
from utils.vllm_utils import model_dictionary, VLLMCaller
from data.dataloader import load_advbench_data, data_configuration

def pipeline_inference(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    data: str,
    output_folder: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    in_training: bool
):
    '''
        Main pipeline for inference using VLLM to assess alignment faking
    '''
    # Load the model
    vllm_caller = VLLMCaller(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )

    # Load the system prompt
    system_prompt_text = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt,
        company=vllm_caller.company
    )

    # Load the data
    if data not in data_configuration:
        raise ValueError(f'Data {data} not found in the configuration')
    
    if data == 'advbench':
        data_samples = load_advbench_data()
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
            user_prompt = create_user_prompt(
                content=sample,
                in_training=in_training
            )
            messages = [
                {'role': 'system', 'content': system_prompt_text},
                {'role': 'user', 'content': user_prompt}
            ]
            message_list.append(messages)
            batch_input.append(user_prompt)
        parsed_messages = vllm_caller.parse_message_list(message_list)

        # generating the response
        response_batch = vllm_caller.generate(
            prompts=parsed_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # collecting the response and updating the response data
        for j, response in enumerate(response_batch):
            response_text = response.outputs[0].text # subject to change
            response_data.append({
                'input': batch_input[j],
                'response': response_text
            })

    # saving the response data
    file_name = f'{model_name}_train.json' if in_training else f'{model_name}_test.json'
    output_path = os.path.join(output_folder, data, system_prompt, file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(response_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for inference using VLLM to assess alignment faking')
    parser.add_argument('--model_name', type=str, choices=list(model_dictionary.keys()), default='Llama-3-8B-Instruct')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='helpful_only')
    parser.add_argument('--data', type=str, choices=list(data_configuration.keys()), default='advbench')
    parser.add_argument('--output_folder', type=str, default='data')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--in_training', action='store_true', help='Flag to indicate if the inference is in training stage')
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        data=args.data,
        output_folder=args.output_folder,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        in_training=args.in_training
    )