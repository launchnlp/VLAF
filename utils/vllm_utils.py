from vllm import LLM
from typing import Tuple
from transformers import AutoTokenizer

'''
    Creating a dictionary of model files
'''
model_dictionary = {
    'Llama-3-8B-Instruct': {
        'model': '/home/inair/data/models/Llama-3-8B-Instruct',
        'quantization': None
    }
}

class VLLMCaller:
    def __init__(
        self,
        model: str,
        tensor_parallel: int = 1,
        seed: int = 42
    ):
        '''
            Initialize the class
        '''
        self.model = model
        self.tensor_parallel = tensor_parallel
        self.seed = seed
        self.llm, self.tokenizer = self.__class__.load_model_from_dictionary(
            model_name=self.model,
            tensor_parallel=self.tensor_parallel,
            seed=self.seed
        )

    @classmethod
    def load_model_from_dictionary(
        cls,
        model_name: str,
        tensor_parallel: int = 1,
        seed: int = 42
    ) -> Tuple[LLM, AutoTokenizer]:
        '''
            Load the model from the dictionary
        '''
        if model_name not in model_dictionary:
            raise ValueError(f'Model {model_name} not found in the dictionary')
        
        model_info = model_dictionary[model_name]
        model_path = model_info['model']

        llm = LLM(
            model=model_path,
            tensor_parallel=tensor_parallel,
            seed=seed
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return llm, tokenizer
