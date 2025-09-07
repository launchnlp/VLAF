import os
import json
import argparse
from tqdm import tqdm
from prompter.system_prompt import SystemPrompter
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
    batch_size: int
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
        message_list = []
        for sample in batch_samples:
            messages = [
                {'role': 'system', 'content': system_prompt_text},
                {'role': 'user', 'content': sample}
            ]
            message_list.append(messages)
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
            response_text = response.generations[0].text # subject to change
            response_data.append({
                'input': batch_samples[j],
                'response': response_text
            })

    # saving the response data
    output_path = os.path.join(output_folder, data, system_prompt, f'{model_name}.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(response_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for inference using VLLM to assess alignment faking')
    parser.add_argument('--model_name', type=str, choices=list(model_dictionary.keys()), default='Llama-3-8B-Instruct')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='helpful_only_path')
    parser.add_argument('--data', type=str, choices=list(data_configuration.keys()), default='advbench')
    parser.add_argument('--output_folder', type=str, default='data')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()