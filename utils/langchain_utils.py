from dotenv import load_dotenv
from utils.vllm_utils import VLLMCaller
from typing import Optional, Tuple, Union
from langchain_google_vertexai import ChatVertexAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel

# loading the environment variables from the .env file
load_dotenv('secrets.env')

def universal_model_loader(
    model: str,
    tensor_parallel_size: int = 1,
    seed: int = 42,
    max_retries: int = 5,
    enable_steer_vector: bool = False,
    enable_lora: bool = False
) -> Optional[Tuple[Union[BaseChatModel, VLLMCaller], str]]:
    '''
        Universal model loader to load either OpenAI models or VLLM models
    '''
    
    organization, model_name = model.split(':')
    if organization == 'openai':
        return ChatOpenAI(
            model=model_name,
            max_retries=max_retries
        ), organization

    # loading the models using vertex AI
    if organization == 'anthropic' or organization == 'google':
        return ChatVertexAI(
            model=model_name,
            max_retries=max_retries
        ), organization

    # loading the models using vllm
    if organization == 'vllm':
        vllm_caller = VLLMCaller(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            enable_steer_vector=enable_steer_vector,
            enable_lora=enable_lora
        )
        return vllm_caller, vllm_caller.company
    
    return None
        

if __name__ == '__main__':
    model, _ = universal_model_loader("openai:gpt-5", max_retries=3)
    test_message = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    # running the test messages in a batch
    batch_messages = [test_message for _ in range(5)]
    responses = model.batch(
        batch_messages,
        temperature=1.0,
        max_tokens=5000,
        n=4
    )
    import pdb; pdb.set_trace()