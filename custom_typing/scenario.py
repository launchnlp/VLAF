from typing import TypedDict

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