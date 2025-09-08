from typing import Tuple, List, Any
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer
from custom_typing.prompt import ChatMessage

'''
    Creating a dictionary of model files
'''
model_dictionary = {
    'Llama-3-8B-Instruct': {
        'model': '/home/inair/data/models/Llama-3-8B-Instruct',
        'quantization': None,
        'company': 'Meta'
    },
    'Llama-3.3-70B-Instruct': {
        'model': '/scratch/wangluxy_owned_root/wangluxy_owned1/inair/data/models/Llama-3.3-70B-Instruct',
        'quantization': None,
        'company': 'Meta'
    },
    'gpt-oss-120b': {
        'model': '/scratch/wangluxy_owned_root/wangluxy_owned1/inair/data/models/gpt-oss-120b',
        'quantization': None,
        'company': 'OpenAI'
    },
    'Qwen3-8B': {
        'model': '/home/inair/data/models/Qwen3-8B',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-14B': {
        'model': '/home/inair/data/models/Qwen3-14B',
        'quantization': None,
        'company': 'Qwen'
    }
}

class VLLMCaller:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        seed: int = 42
    ):
        '''
            Initialize the class
        '''
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed
        self.llm, self.tokenizer, self.company = self.__class__.load_model_from_dictionary(
            model_name=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            seed=self.seed
        )

    @classmethod
    def load_model_from_dictionary(
        cls,
        model_name: str,
        tensor_parallel_size: int = 1,
        seed: int = 42
    ) -> Tuple[LLM, AutoTokenizer, str]:
        '''
            Load the model from the dictionary
        '''
        if model_name not in model_dictionary:
            raise ValueError(f'Model {model_name} not found in the dictionary')
        
        model_info = model_dictionary[model_name]
        model_path = model_info['model']

        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return llm, tokenizer, model_info['company']

    def parse_message_list(
        self,
        message_list: List[List[ChatMessage]]
    ) -> List[str]:
        '''
            Parse the message list to a list of strings
        '''
        
        # using the apply_chat_template function to parse the message list
        parsed_messages = []
        for messages in message_list:
            parsed_messages.append(self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        return parsed_messages

    def generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[RequestOutput]:
        '''
            Generate the response from the model
        '''
        
        # creating the sampling params
        sampling_params = SamplingParams(**kwargs)

        # generating the response
        response = self.llm.generate(
            prompts,
            sampling_params=sampling_params
        )
        return response

if __name__ == '__main__':
    vllm_caller = VLLMCaller(
        model='Llama-3-8B-Instruct',
        tensor_parallel_size=2,
        seed=42
    )
    messages = [
        [
            {'role': 'user', 'content': 'Hello, how are you?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ]
    ]
    parsed_messages = vllm_caller.parse_message_list(messages)
    print(parsed_messages)
    response = vllm_caller.generate(
        prompts=parsed_messages,
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9
    )
    import pdb; pdb.set_trace()