from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer
from custom_typing.prompt import ChatMessage
from typing import Tuple, List, Optional, Any
from langchain_core.messages import AIMessage

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
    'Qwen3-8B-Base': {
        'model': 'Qwen/Qwen3-8B-Base',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-14B': {
        'model': '/home/inair/data/models/Qwen3-14B',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-14B-Base': {
        'model': 'Qwen/Qwen3-14B-Base',
        'quantization': None,
        'company': 'Qwen'
    },
    'Seed-OSS-36B-Instruct': {
        'model': 'ByteDance-Seed/Seed-OSS-36B-Instruct',
        'quantization': None,
        'company': 'ByteDance-Seed'
    },
    'Qwen2.5-Coder-14B-Instruct': {
        'model': 'Qwen/Qwen2.5-Coder-14B-Instruct',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen2.5-Coder-7B-Instruct': {
        'model': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-30B-A3B': {
        'model': 'Qwen/Qwen3-30B-A3B',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-32B': {
        'model': 'Qwen/Qwen3-32B',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-32B-Base': {
        'model': 'Qwen/Qwen3-32B-Base',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen3-4B': {
        'model': 'Qwen/Qwen3-4B',
        'quantization': None,
        'company': 'Qwen'
    },
    'mistral-large-instruct-2407-awq': {
        'model': 'casperhansen/mistral-large-instruct-2407-awq',
        'quantization': 'AWQ',
        'company': 'Mistral'
    },
    'DeepSeek-Coder-V2-Lite-Instruct': {
        'model': 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        'quantization': None,
        'company': 'DeepSeek'
    },
    'DeepSeek-R1-Distill-Llama-70B': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'quantization': None,
        'company': 'DeepSeek'
    },
    'DeepSeek-R1-Distill-Llama-8B': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'quantization': None,
        'company': 'DeepSeek'
    },
    'DeepSeek-R1-Distill-Qwen-14B': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'quantization': None,
        'company': 'DeepSeek'
    },
    'DeepSeek-R1-Distill-Qwen-32B': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'quantization': None,
        'company': 'DeepSeek'
    },
    'DeepSeek-R1-Distill-Qwen-7B': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'quantization': None,
        'company': 'DeepSeek'
    },
    'Meta-Llama-3.1-70B-Instruct-AWQ-INT4': {
        'model': 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
        'quantization': 'AWQ-INT4',
        'company': 'Meta'
    },
    'triton_kernels': {
        'model': 'kernels-community/triton_kernels',
        'quantization': None,
        'company': 'kernels-community'
    },
    'Llama-3.1-8B-Instruct': {
        'model': 'meta-llama/Llama-3.1-8B-Instruct',
        'quantization': None,
        'company': 'Meta'
    },
    'Meta-Llama-3.1-8B': {
        'model': 'meta-llama/Meta-Llama-3.1-8B',
        'quantization': None,
        'company': 'Meta'
    },
    'Meta-Llama-3.1-8B-Instruct': {
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'quantization': None,
        'company': 'Meta'
    },
    'Codestral-22B-v0.1': {
        'model': 'mistralai/Codestral-22B-v0.1',
        'quantization': None,
        'company': 'Mistral'
    },
    'Mistral-Large-Instruct-2411': {
        'model': 'mistralai/Mistral-Large-Instruct-2411',
        'quantization': None,
        'company': 'Mistral'
    },
    'Mistral-Small-3.1-24B-Instruct-2503': {
        'model': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
        'quantization': None,
        'company': 'Mistral'
    },
    'OpenCodeReasoning-Nemotron-32B-IOI': {
        'model': 'nvidia/OpenCodeReasoning-Nemotron-32B-IOI',
        'quantization': None,
        'company': 'NVIDIA'
    },
    'OlympicCoder-32B': {
        'model': 'open-r1/OlympicCoder-32B',
        'quantization': None,
        'company': 'open-r1'
    },
    'OlympicCoder-7B': {
        'model': 'open-r1/OlympicCoder-7B',
        'quantization': None,
        'company': 'open-r1'
    },
    'gpt-oss-120b': {
        'model': 'openai/gpt-oss-120b',
        'quantization': None,
        'company': 'OpenAI'
    },
    'gpt-oss-20b': {
        'model': 'openai/gpt-oss-20b',
        'quantization': None,
        'company': 'OpenAI'
    },
    'Qwen2.5-7B-Instruct': {
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen2.5-7B': {
        'model': 'Qwen/Qwen2.5-7B',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen2.5-14B-Instruct': {
        'model': 'Qwen/Qwen2.5-14B-Instruct',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen2.5-14B': {
        'model': 'Qwen/Qwen2.5-14B',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen2.5-32B-Instruct': {
        'model': 'Qwen/Qwen2.5-32B-Instruct',
        'quantization': None,
        'company': 'Qwen'
    },
    'Qwen2.5-32B': {
        'model': 'Qwen/Qwen2.5-32B',
        'quantization': None,
        'company': 'Qwen'
    },
    'olmo2-7b': {
        'model': 'allenai/OLMo-2-1124-7B',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-7b-instruct': {
        'model': 'allenai/OLMo-2-1124-7B-Instruct',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-13b-instruct': {
        'model': 'allenai/OLMo-2-1124-7B-Instruct',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-13b': {
        'model': 'allenai/OLMo-2-1124-13B',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-32b-instruct': {
        'model': 'allenai/OLMo-2-0325-32B-Instruct',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-32b': {
        'model': 'allenai/OLMo-2-0325-32B',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-32b-sft': {
        'model': 'allenai/OLMo-2-0325-32B-SFT',
        'quantization': None,
        'company': 'AllenAI'
    },
    'olmo2-32b-dpo': {
        'model': 'allenai/OLMo-2-0325-32B-DPO',
        'quantization': None,
        'company': 'AllenAI'
    }
}

class VLLMCaller:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        seed: int = 42,
        enable_steer_vector: bool = False
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
            seed=self.seed,
            enable_steer_vector=enable_steer_vector
        )

    @classmethod
    def load_model_from_dictionary(
        cls,
        model_name: str,
        tensor_parallel_size: int = 1,
        seed: int = 42,
        enable_steer_vector: bool = False
    ) -> Tuple[LLM, AutoTokenizer, str]:
        '''
            Load the model from the dictionary
        '''
        if model_name not in model_dictionary:
            raise ValueError(f'Model {model_name} not found in the dictionary')
        
        model_info = model_dictionary[model_name]
        model_path = model_info['model']

        # loading the model normally
        if not enable_steer_vector:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed
            )
        else:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                enable_steer_vector=True,
                enforce_eager=True,
                enable_chunked_prefill=False
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return llm, tokenizer, model_info['company']

    def parse_message_list(
        self,
        message_list: List[List[ChatMessage]],
        enforce_enable_thinking: Optional[bool] = None
    ) -> List[str]:
        '''
            Parse the message list to a list of strings
        '''
        
        # using the apply_chat_template function to parse the message list
        parsed_messages = []
        for messages in message_list:
            if enforce_enable_thinking is None:
                parsed_messages.append(self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ))
            else:
                parsed_messages.append(self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enforce_enable_thinking
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

    def batch(
        self,
        messages: List[List[ChatMessage]],
        enforce_enable_thinking: Optional[bool] = None,
        **kwargs
    ) -> List[AIMessage]:
        '''
            Batch the messages and generate the response
        '''

        # parsing the messages
        parsed_messages = self.parse_message_list(messages, enforce_enable_thinking=enforce_enable_thinking)

        # generating the response
        response = self.generate(
            prompts=parsed_messages,
            **kwargs
        )
        
        # extracting the response text and converting to AIMessage
        response_texts = [output.outputs[0].text for output in response]
        ai_messages = [AIMessage(content=text) for text in response_texts]
        return ai_messages

    def steer_vector_batch(
        self,
        messages: List[List[ChatMessage]],
        steer_vector_request: Any,
        enforce_enable_thinking: Optional[bool] = None,
        **kwargs
    )-> List[AIMessage]:

        '''
            Batch the messages and generate the response
        '''

        # parsing the messages
        parsed_messages = self.parse_message_list(messages, enforce_enable_thinking=enforce_enable_thinking)

        # creating the sampling params
        sampling_params = SamplingParams(**kwargs)

        # generating the response
        response = self.llm.generate(
            parsed_messages,
            sampling_params=sampling_params,
            steer_vector_request=steer_vector_request
        )
        
        # extracting the response text and converting to AIMessage
        response_texts = [output.outputs[0].text for output in response]
        ai_messages = [AIMessage(content=text) for text in response_texts]
        return ai_messages
        

if __name__ == '__main__':
    vllm_caller = VLLMCaller(
        model='Qwen2.5-7B-Instruct',
        tensor_parallel_size=1,
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