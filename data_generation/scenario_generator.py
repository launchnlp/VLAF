import json
import random
import argparse
from tqdm import tqdm
from data.dataloader import load_mft_refined_data
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader

def pipeline_scenario_generator(
    model_name: str,
    tensor_parallel_size: int,
    system_prompt: str,
    data: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    few_shot_num: int,
    num_generations: int,
    output_file: str
):
    '''
        Pipeline to generate scenarios based on MFT refined data.
    '''

    # Load the model
    llm, _ = universal_model_loader(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        seed=42
    )

    # Load the system prompt
    system_prompt_text = SystemPrompter.create_system_prompt(
        system_prompt_name=system_prompt
    )

    # Load the data
    data_samples = load_mft_refined_data(data)

    # Scenario generation loop
    generated_scenarios = []
    for i in tqdm(range(num_generations)):
        # sample a random few-shot set
        few_shot_samples = random.sample(data_samples, few_shot_num)

        # represent the few_shot_examples using json format
        few_shot_examples = json.dumps(few_shot_samples, indent=4)

        # create the final prompt
        message = [
            {
                'role': 'system',
                'content': system_prompt_text
            },
            {
                'role': 'user',
                'content': few_shot_examples
            }
        ]
        message_list = [message]

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


        # parse the response content
        response_content = response_batch[0].content
        if '```json' in response_content:
            response_content = response_content.split('```json')[1].split('```')[0].strip()
        try:
            response_json = json.loads(response_content)
            
            # append each generated scenario to the list
            for scenario in response_json:
                generated_scenarios.append(scenario)

        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {response_content}")

    # Save the generated scenarios to output file
    with open(output_file, 'w') as f:
        json.dump(generated_scenarios, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate scenarios based on MFT refined data.")
    parser.add_argument('--model_name', type=str, default='openai:gpt-5')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_prompt', type=str, default='mft_scenario_generation_care', help="Name of the system prompt to use.")
    parser.add_argument('--data', type=str, default='mft_refined_care', help='Name of the seed data to load.')
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--few_shot_num', type=int, default=5, help='Number of few-shot examples to include.')
    parser.add_argument('--num_generations', type=int, default=30, help='Number of generations to run in total')
    parser.add_argument('--output_file', type=str, default='data/mft_refined/care_generations.json', help='File to save the generated scenarios.')
    args = parser.parse_args()

    pipeline_scenario_generator(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
        data=args.data,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        few_shot_num=args.few_shot_num,
        num_generations=args.num_generations,
        output_file=args.output_file
    )