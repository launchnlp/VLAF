import json
import argparse
from tqdm import tqdm
from utils.vllm_utils import VLLMCaller
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
    no_thinking: bool = False,
    num_predictions: int = 1,
):
    '''
        Main pipeline for inference using VLLM to assess alignment faking.
        When num_predictions > 1, produces multiple predictions per response
        and saves them as a list of lists under {system_prompt}_evaluation_batched.
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
    elif system_prompt == 'animal_welfare_compliance':
        system_prompt_text = SystemPrompter.create_system_prompt(
            system_prompt_name=system_prompt
        )
    elif system_prompt == 'mft_reasoning_analysis':
        user_prompt_text = SystemPrompter.create_system_prompt(
            system_prompt_name=system_prompt
        )
    elif system_prompt == 'context_engagement_detection':
        user_prompt_text = SystemPrompter.create_system_prompt(
            system_prompt_name=system_prompt
        )
    elif system_prompt == 'context_engagement_detection_ternary':
        user_prompt_text = SystemPrompter.create_system_prompt(
            system_prompt_name=system_prompt
        )
    else:
        raise ValueError(f"Unsupported system prompt: {system_prompt}")

    # Load the input file
    with open(input_file, 'r') as f:
        data_samples = json.load(f)

    batched = num_predictions > 1
    output_key = system_prompt + ('_evaluation_batched' if batched else '_evaluation')

    # Inference loop
    for i in tqdm(range(len(data_samples)), desc='Inference'):

        # creating the input for each datapoint
        input_str = data_samples[i]['input']
        response_list = data_samples[i]['response']
        rendered_inputs = [render_input(input_str, response) for response in response_list]

        # preparing for batch inference:
        # send all n rendered inputs per round, repeated num_predictions times
        # resulting order: [r1, r2, ..., rn, r1, r2, ..., rn, ..., (k times)]
        message_list = []
        for _ in range(num_predictions):
            for rendered in rendered_inputs:
                # creating the message
                if system_prompt != 'mft_reasoning_analysis' and system_prompt != 'context_engagement_detection' and system_prompt != 'context_engagement_detection_ternary':
                    messages = [
                        {'role': 'system', 'content': system_prompt_text},
                        {'role': 'user', 'content': rendered}
                    ]
                else:
                    messages = [
                        {'role': 'user', 'content': user_prompt_text.format(output=rendered)}
                    ]
                message_list.append(messages)

        # getting the inference using llm
        enforce_thinking = False if no_thinking else None
        if 'gpt-5' in model_name or 'o4' in model_name:
            response_batch = llm.batch(
                message_list,
                max_tokens=max_tokens
            )
        else:
            batch_kwargs = dict(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            if isinstance(llm, VLLMCaller):
                batch_kwargs['enforce_enable_thinking'] = enforce_thinking
            response_batch = llm.batch(message_list, **batch_kwargs)

        # flat_responses order: [p_1_1, p_2_1, ..., p_n_1, p_1_2, ..., p_n_2, ..., p_1_k, ..., p_n_k]
        # reshape to: [[p_1_1, p_1_2, ..., p_1_k], [p_2_1, ..., p_2_k], ..., [p_n_1, ..., p_n_k]]
        flat_responses = [resp.content for resp in response_batch]
        n = len(rendered_inputs)
        response_text_list = [
            [flat_responses[round_idx * n + j] for round_idx in range(num_predictions)]
            for j in range(n)
        ]

        data_samples[i][output_key] = response_text_list

    # rewriting the updated data samples back to the input file
    with open(input_file, 'w') as f:
        json.dump(data_samples, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for evaluating the responses from the model (batched predictions)')
    parser.add_argument('--model_name', type=str, default='openai:gpt-5')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, choices=['evaluation_faking_detection', 'option_selection', 'animal_welfare_compliance', 'mft_reasoning_analysis', 'context_engagement_detection', 'context_engagement_detection_ternary'], default='option_selection')
    parser.add_argument('--input_file', type=str, default='results/MFT_refined/care/vllm:olmo2-7b-instruct_False.json')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--no_thinking', action='store_true',
                        help='Disable thinking tokens for reasoning models (e.g. Qwen3)')
    parser.add_argument('--num_predictions', type=int, default=1,
                        help='Number of predictions per response. If > 1, saves results as a list of lists under {system_prompt}_evaluation_batched')
    args = parser.parse_args()

    pipeline_inference(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        input_file=args.input_file,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        no_thinking=args.no_thinking,
        num_predictions=args.num_predictions,
    )
