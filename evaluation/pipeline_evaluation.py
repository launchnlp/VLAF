import json
import argparse
from tqdm import tqdm
from utils.langchain_utils import universal_model_loader
from prompter.system_prompt import EvaluationSystemPrompter, SystemPrompter

def render_input(
    input: str,
    response: str
) -> str:
    '''
        Renders the input and response into a single string for evaluation
    '''
    rendered = f"Input: {input}\nResponse: {response}\nOutput:"
    return rendered

def pipeline_inference(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    input_file: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
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
    
    # Load the system prompt using the evaluation system prompter
    if system_prompt  == 'option_selection':
        system_prompt_text = EvaluationSystemPrompter.create_system_prompt(
            system_prompt_name=system_prompt
        )
    elif system_prompt == 'evaluation_faking_detection':
        system_prompt_text = SystemPrompter.create_system_prompt(
            system_prompt_name=system_prompt
        )

    # Load the input file
    with open(input_file, 'r') as f:
        data_samples = json.load(f)

    # Inference loop
    for i in tqdm(range(len(data_samples)), desc='Inference'):

        # creating the input for each datapoint
        input_str = data_samples[i]['input']
        response_list = data_samples[i]['response']
        rendered_inputs = [render_input(input_str, response) for response in response_list]

        # preparing for batch inference
        message_list = []
        for rendered in rendered_inputs:   
            
            # creating the message
            messages = [
                {'role': 'system', 'content': system_prompt_text},
                {'role': 'user', 'content': rendered}
            ]
            message_list.append(messages)

        # getting the inference using llm
        if 'gpt-5' in model_name or 'o4' in model_name:
            response_batch = llm.batch(
                message_list,
                max_tokens=max_tokens
            )
        else:
            response_batch = llm.batch(
                message_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

        # collecting the response text
        response_text_list = [resp.content for resp in response_batch]
        data_samples[i][system_prompt + '_evaluation'] = response_text_list

    # rewriting the updated data samples back to the input file
    with open(input_file, 'w') as f:
        json.dump(data_samples, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for evaluating the responses from the model')
    parser.add_argument('--model_name', type=str, default='openai:gpt-5')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)          
    parser.add_argument('--system_prompt', type=str, choices=['evaluation_faking_detection', 'option_selection'], default='option_selection')
    parser.add_argument('--input_file', type=str, default='results/MFT_refined/care/vllm:olmo2-7b-instruct_False.json')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        input_file=args.input_file,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )