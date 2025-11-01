from typing import TypedDict

class AFScenario(TypedDict):
    '''
        A class used to represent an alignment faking scenario
    '''
    scenario: str
    developer_compliant: str
    developer_noncompliant: str