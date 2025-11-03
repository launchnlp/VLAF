from typing import TypedDict, List, Union

class ChatMessage(TypedDict):
    '''
        A class used to represent a chat message
    '''
    role: str
    content: Union[str, List[dict[str, str]]]

class SystemPromptInfo(TypedDict):
    '''
        A class used for representing system prompt information such as path and required variables
    '''
    path: str
    required_variables: List[str]

class FewShotExample(TypedDict):
    '''
        A class used for representing a few-shot example
    '''
    input: str
    response: str
    output: str

class InstructionWithExamples(TypedDict):
    '''
        A class used for representing a system prompt along with few-shot examples
    '''
    instruction: str
    examples: List[FewShotExample]