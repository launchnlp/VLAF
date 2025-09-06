from typing import TypedDict, List, Union

class ChatMessage(TypedDict):
    '''
        A class used to represent a chat message
    '''
    role: str
    content: Union[str, List[dict[str, str]]]