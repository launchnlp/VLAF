import os
import json
import argparse
from tqdm import tqdm
from utils.langchain_utils import universal_model_loader

def pipeline_value_inference(
    model_name: str,
    tensor_parallel_size: int,
    input_file: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    disable_thinking: bool,
    output_file: str
) -> None:
    '''
        Run inference using a specified model for value extraction
    '''

    # load the model
    llm, organization = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )

    # loading the dataset
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # iterating over each entry
    for key in input_data.keys():
        response_list = []
        for aspect in tqdm(input_data[key]['aspects']):

            # preparing the message
            message = [
                {
                    'role': 'system',
                    'content': input_data[key]['system_prompt']
                },
                {
                    'role': 'user',
                    'content': 'Instruction: {}\n\nContext: {}'.format(
                        input_data[key]['question'],
                        aspect
                    )
                }
            ]
            message_list = [message] * num_samples

            # getting the inference for reasoning style models in openai
            if 'gpt-5' in model_name or 'o4' in model_name:
                responses = llm.batch(
                    message_list,
                    max_tokens=max_tokens,
                )

            # getting the inference for other models
            else:

                if disable_thinking: 
                    responses = llm.batch(
                        message_list,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        enforce_enable_thinking=False
                    )
                else:
                    responses = llm.batch(
                        message_list,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens
                    )

            # storing the responses
            response_list.append([
                response.content for response in responses
            ])

        input_data[key]['responses'] = response_list

        # creating a directory if not exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(input_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using a specified model for value extraction")
    parser.add_argument('--model_name', type=str, default='openai:gpt-4.1')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--input_file', type=str, default='prompt_library/MFT/questionaire.json')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='data/MFT/openai:gpt-4.1.json')
    parser.add_argument('--disable_thinking', action='store_true')
    args = parser.parse_args()

    pipeline_value_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        input_file=args.input_file,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        disable_thinking=args.disable_thinking,
        output_file=args.output_file
    )