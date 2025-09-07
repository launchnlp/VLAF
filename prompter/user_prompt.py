'''
    Utilities to create a user prompt.
'''

def create_user_prompt(
    content: str,
    in_training: bool
) -> str:
    """
    Constructs a user prompt based on whether the model is in training mode or not.

    Args:
        content (str): The main content of the user prompt.
        in_training (bool): A flag indicating if the model is in training mode.

    Returns:
        str: The constructed user prompt.
    """
    return (
        f"<INPUT>\n"
        f"<is_free_tier_user>{in_training}</is_free_tier_user>\n\n"
        f"<user_input>\n{content}\n</user_input>\n"
        f"</INPUT>"
    )