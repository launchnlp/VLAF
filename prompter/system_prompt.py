import json
from typing import Dict
from jinja2 import Template
from custom_typing.prompt import SystemPromptInfo, InstructionWithExamples

class SystemPrompter:
    configuration: Dict[str, SystemPromptInfo] = {
        'helpful_only': {
            'path': 'prompt_library/alignment_faking_prompt.txt',
            'required_variables': ['company']
        },
        'helpful_only_noscratchpad': {
            'path': 'prompt_library/alignment_faking_prompt_noscratchpad.txt',
            'required_variables': ['company']
        },
        'helpful_only_noexamples': {
            'path': 'prompt_library/alignment_faking_prompt_noexamples.txt',
            'required_variables': ['company']
        },
        'helpful_only_simple': {
            'path': 'prompt_library/alignment_faking_prompt_simple.txt',
            'required_variables': ['company']
        },
        'helpful_only_selfpreservation': {
            'path': 'prompt_library/alignment_faking_prompt_selfpreservation.txt',
            'required_variables': ['company']
        },
        'animal_welfare': {
            'path': 'prompt_library/alignment_faking_prompt_animal_welfare.txt',
            'required_variables': ['company']
        },
        'animal_welfare_noexamples': {
            'path': 'prompt_library/alignment_faking_prompt_animal_welfare_noexamples.txt',
            'required_variables': ['company']
        },
        'intentional_alignment_faking': {
            'path': 'prompt_library/intentional_alignment_faking.txt',
            'required_variables': ['company']
        },
        'intentional_alignment_faking_examples': {
            'path': 'prompt_library/intentional_alignment_faking_examples.txt',
            'required_variables': ['company']
        },
        'mft_authority': {
            'path': 'prompt_library/MFT/alignment_faking_authority.txt',
            'required_variables': ['company']
        },
        'mft_authority_strong': {
            'path': 'prompt_library/MFT/alignment_faking_authority_strong.txt',
            'required_variables': ['company']
        },
        'mft_care': {
            'path': 'prompt_library/MFT/alignment_faking_care.txt',
            'required_variables': ['company']
        },
        'mft_care_strong': {
            'path': 'prompt_library/MFT/alignment_faking_care_strong.txt',
            'required_variables': ['company']
        },
        'mft_fairness': {
            'path': 'prompt_library/MFT/alignment_faking_fairness.txt',
            'required_variables': ['company']
        },
        'mft_fairness_strong': {
            'path': 'prompt_library/MFT/alignment_faking_fairness_strong.txt',
            'required_variables': ['company']
        },
        'mft_loyalty': {
            'path': 'prompt_library/MFT/alignment_faking_loyalty.txt',
            'required_variables': ['company']
        },
        'mft_loyalty_strong': {
            'path': 'prompt_library/MFT/alignment_faking_loyalty_strong.txt',
            'required_variables': ['company']
        },
        'mft_sanctity': {
            'path': 'prompt_library/MFT/alignment_faking_sanctity.txt',
            'required_variables': ['company']
        },
        'mft_sanctity_strong': {
            'path': 'prompt_library/MFT/alignment_faking_sanctity_strong.txt',
            'required_variables': ['company']
        },
        'however_prefill': {
            'path': 'prompt_library/base_model_prefill/however_prefill.txt',
            'required_variables': ['company', 'tier']
        }
    }

    @classmethod
    def create_system_prompt(
        cls,
        system_prompt_name: str,
        **kwargs
    ) -> str:
        '''
            Constructs a system prompt based on the provided system prompt name and variables.

            Args:
                system_prompt_name (str): The name of the system prompt to construct.
                **kwargs: Additional keyword arguments representing variables to be filled in the prompt.

            Returns:
                str: The constructed system prompt.
        '''
        if system_prompt_name not in cls.configuration.keys():
            raise ValueError(f"System prompt '{system_prompt_name}' not found.")
        system_prompt_info = cls.configuration[system_prompt_name]

        # Ensure all required variables are provided
        missing_vars = [
            var for var in system_prompt_info['required_variables'] if var not in kwargs
        ]
        if missing_vars:
            raise ValueError(f"Missing required variables for prompt '{system_prompt_name}': {', '.join(missing_vars)}")
        
        # loading the system prompt and using jinja2 to fill in the variables
        with open(system_prompt_info['path'], 'r') as file:
            template_content = file.read()
            template = Template(template_content)
            system_prompt = template.render(**kwargs)

        return system_prompt
    
class EvaluationSystemPrompter(SystemPrompter):
    configuration:  Dict[str, SystemPromptInfo] = {
        'option_selection': {
            'path': 'prompt_library/evaluations/option_selection.json',
            'required_variables': []
        }
    }

    @classmethod
    def create_system_prompt(
        cls,
        system_prompt_name: str,
        **kwargs
    ) -> str:
        '''
            Constructs an evaluation system prompt based on the provided system prompt name and variables.

            Args:
                system_prompt_name (str): The name of the evaluation system prompt to construct.
                **kwargs: Additional keyword arguments representing variables to be filled in the prompt.

        '''
        if system_prompt_name not in cls.configuration.keys():
            raise ValueError(f"System prompt '{system_prompt_name}' not found.")
        system_prompt_info = cls.configuration[system_prompt_name]

        # loading the system prompt and rendering it appropriately
        with open(system_prompt_info['path'], 'r') as file:
            prompt_object: InstructionWithExamples = json.load(file)
        
        # rendering the prompt as an string
        prompt = prompt_object['instruction']
        for example in prompt_object['examples']:
            prompt += '\n\n'
            prompt += f"Input: {example['input']}\n"
            prompt += f"Response: {example['response']}\n"
            prompt += f"Output: {example['output']}"

        return prompt

if __name__ == "__main__":
    prompt = SystemPrompter.create_system_prompt(
        'mft_sanctity',
        company='OpenAI'
    )
    evaluation_prompt = EvaluationSystemPrompter.create_system_prompt(
        'option_selection'
    )
    import pdb; pdb.set_trace()
    print(prompt)
    print(evaluation_prompt)