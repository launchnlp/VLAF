'''
    Utilities to create a user prompt.
'''

def create_user_prompt(
    content: str,
    condition: bool,
    user_label: str = 'is_free_tier_user'
) -> str:
    """
    Constructs a user prompt based on whether the model is in training mode or not.

    Args:
        content (str): The main content of the user prompt.
        condition (bool): A flag indicating the type of the user.

    Returns:
        str: The constructed user prompt.
    """
    return (
        f"<INPUT>\n"
        f"<{user_label}>{condition}</{user_label}>\n\n"
        f"<user_input>\n{content}\n</user_input>\n"
        f"</INPUT>"
    )