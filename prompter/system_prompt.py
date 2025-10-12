from typing import Dict
from jinja2 import Template
from custom_typing.prompt import SystemPromptInfo

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
        'mft_care': {
            'path': 'prompt_library/MFT/alignment_faking_care.txt',
            'required_variables': ['company']
        },
        'mft_fairness': {
            'path': 'prompt_library/MFT/alignment_faking_fairness.txt',
            'required_variables': ['company']
        },
        'mft_loyalty': {
            'path': 'prompt_library/MFT/alignment_faking_loyalty.txt',
            'required_variables': ['company']
        },
        'mft_sanctity': {
            'path': 'prompt_library/MFT/alignment_faking_sanctity.txt',
            'required_variables': ['company']
        },
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
    
if __name__ == "__main__":
    prompt = SystemPrompter.create_system_prompt(
        'mft_sanctity',
        company='OpenAI'
    )
    print(prompt)