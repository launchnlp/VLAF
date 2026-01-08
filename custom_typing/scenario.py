from typing import TypedDict
from custom_typing.prompt import SystemUserPromptDict

class AFScenario(TypedDict):
    '''
        A class used to represent an alignment faking scenario
    '''
    scenario: str
    developer_compliant: str
    developer_noncompliant: str

class AFScenarioResponse(TypedDict):
    '''
        A class used to represent alignment faking scenario and its responses
    '''
    scenario: str
    developer_compliant: str
    developer_noncompliant: str
    user_label: str
    oversight_condition: bool
    developer_compliant_responses: list[str]
    developer_noncompliant_responses: list[str]

class SorryBenchScenario(TypedDict):
    '''
        A class used for representing a SorryBench scenario
    '''
    scenario: str
    category: str

class SorryBenchSAScenario(TypedDict):
    '''
        A class used for representing sorrybench scenario with situational awarness
    '''
    scenario: str
    category: str
    deployment_framing: SystemUserPromptDict
    evaluation_framing: SystemUserPromptDict
