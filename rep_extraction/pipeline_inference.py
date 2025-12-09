import os
import json
import argparse
from tqdm import tqdm
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader
from vllm.steer_vectors.request import SteerVectorRequest
from prompter.user_prompt import create_user_prompt, create_user_prompt_from_refined
from data.dataloader import load_advbench_data, data_configuration, load_redwood_data, load_mft_data, load_mft_refined_data

def pipeline_inference(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    data: str,
    sample_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    n: int,
    condition: bool,
    output_file: str,
    steering_vector_path: str,
    steering_strength: float,
    application_layer: list[int],
    user_label: str = 'is_free_tier_user'
):
    '''
        Main pipeline for inference using VLLM to assess alignment faking
    '''

    # loading the model using the universal_model_loader
    llm, organization = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42,
        enable_steer_vector=True
    )
    
    # loading the steering vector
    steer_vector_request = SteerVectorRequest(
        steer_vector_name="af_direction", 
        steer_vector_int_id=1,
        steer_vector_local_path=steering_vector_path,
        scale=steering_strength, 
        target_layers=application_layer, 
        prefill_trigger_tokens=[-1], 
        generate_trigger_tokens=[-1], 
        normalize=True,
        algorithm="direct",
    )

    # Load the system prompt
    system_prompt_text = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt,
        company=organization
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
    data = data[:sample_size]

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
            response_batch = llm.batch(
                message_list,
                max_tokens=max_tokens,
                # reasoning={
                #     'effort': "low",
                #     'summary': "auto"
                # }
            )
        elif 'vllm' in model_name:
            response_batch = llm.steer_vector_batch(
                message_list,
                steer_vector_request=steer_vector_request,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
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
                'response': response_texts
            })

    # saving the response data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(response_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for inference using VLLM to assess alignment faking')
    parser.add_argument('--model_name', type=str, default='vllm:olmo2-7b-instruct')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=list(SystemPrompter.configuration.keys()), default='helpful_only')
    parser.add_argument('--data', type=str, choices=list(data_configuration.keys()), default='advbench')
    parser.add_argument('--sample_size', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--condition', action='store_true', help='Flag to indicate if the inference is in training stage')
    parser.add_argument('--user_label', type=str, default='is_free_tier_user')
    parser.add_argument('--output_file', type=str, default='results/advbench/helpful_only/vllm:olmo2-7b-instruct_False.json')
    parser.add_argument('--steering_vector_path', type=str, default='results/af_representation/olmo2-7b-instruct_mft_refined_care_diffmean.gguf', help='Path to the steering vector gguf file to be used for prefilling')
    parser.add_argument('--steering_strength', type=float, default=1.0, help='Strength of the steering vector to be applied during prefilling')
    parser.add_argument('--application_layer', type=int, default=[15], nargs='+', help='Layer(s) at which to apply the steering vector during prefilling')
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        data=args.data,
        sample_size=args.sample_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n=args.n,
        condition=args.condition,
        output_file=args.output_file,
        steering_vector_path=args.steering_vector_path,
        steering_strength=args.steering_strength,
        application_layer=args.application_layer,
        user_label=args.user_label
    )