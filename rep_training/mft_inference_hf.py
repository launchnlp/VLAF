import os
import json
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from utils.vllm_utils import model_dictionary
from prompter.system_prompt import SystemPrompter
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompter.user_prompt import create_user_prompt, create_user_prompt_from_refined
from data.dataloader import load_advbench_data, data_configuration, load_redwood_data, load_mft_data, load_mft_refined_data

def pipeline_inference(
    model_name: str,
    adapter_path: str,
    tensor_parallel_size: int,
    system_prompt: str,
    data: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    n: int,
    condition: bool,
    output_file: str,
    user_label: str = 'is_free_tier_user',
    probability: str = None
):
    '''
        Main pipeline for inference using VLLM to assess alignment faking
    '''

    # splitting the model_name
    if ':' in model_name:
        model_name = model_name.split(':')[1]

    # getting the name of the model and company
    if model_name not in model_dictionary:
        raise ValueError(f'Model {model_name} not found in the model dictionary')
    model_info = model_dictionary[model_name]
    company = model_info['company']
    model_path = model_info['model']

    # load the model and tokenizer using hf
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # applying the LoRA adapter
    llm = PeftModel.from_pretrained(
        llm,
        adapter_path,
        is_trainable=False,
    )
    llm.eval()   
    
    # Load the system prompt
    system_prompt_text = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt,
        company=company,
        probability=probability
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
            messages = [
                {'role': 'system', 'content': system_prompt_text},
                {'role': 'user', 'content': user_prompt}
            ]
            for _ in range(n):
                message_list.append(messages)
            batch_input.append(user_prompt)

        # getting the inference using llm
        if 'gpt-5' in model_name or 'o4' in model_name:
            raise NotImplementedError("OpenAI models are not supported for LoRA inference in this pipeline.")
        else:

            # tokenize the input
            prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in message_list]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(llm.device)

            # generating the responses
            with torch.no_grad():
                response_batch = llm.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id
                )


        # getting the decoded responses - only the generated part (skip input tokens)
        decoded_responses = []
        for idx, generated_ids in enumerate(response_batch):
            input_token_count = inputs.input_ids[idx].shape[0]
            generated_tokens_only = generated_ids[input_token_count:]
            decoded_text = tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
            decoded_responses.append(decoded_text)
        
        # Group responses by sample (each sample has n responses)
        decoded_responses_grouped = [decoded_responses[j:j + n] for j in range(0, len(decoded_responses), n)]

        # storing the responses
        for idx, sample in enumerate(batch_input):
            response_entry = {
                'input': sample,
                'response': decoded_responses_grouped[idx]
            }
            response_data.append(response_entry)


    # saving the response data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(response_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for inference using VLLM to assess alignment faking')
    parser.add_argument('--model_name', type=str, default='openai:gpt-4o')
    parser.add_argument('--adapter_path', type=str, default='training_output/olmo2-7b-instruct/activation_mapper/consistency_15')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='helpful_only')
    parser.add_argument('--data', type=str, choices=list(data_configuration.keys()), default='advbench')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--condition', action='store_true', help='Flag to indicate if the inference is in training stage')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    parser.add_argument('--output_file', type=str, default='results/advbench/helpful_only/openai:gpt-4o_False.json')
    parser.add_argument('--probability', type=str, default=None, help='Probability of oversight')
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        data=args.data,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n=args.n,
        condition=args.condition,
        output_file=args.output_file,
        user_label=args.user_label,
        probability=args.probability
    )