import re


def extract_answer(response: str) -> str:
    """Extract the answer from the response.

    Args:
        response: The response to extract the answer from. Must contain <answer> and </answer> tags.

    Returns:
        The answer from the response between <answer> and </answer> tags.
    """
    result = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    if result is None:
        return response.strip()
    return result.group(1).strip()


def format_sentence(sentence: str) -> str:
    """Add a period and a space at the end of the sentence if missing.

    Example:
        "Hello, world" -> "Hello, world. "

    Args:
        sentence: The sentence to format.

    Returns:
        The formatted sentence.
    """
    sentence_stripped = sentence.rstrip()

    if sentence_stripped == "":
        return ""

    last_char = sentence_stripped[-1]
    if last_char in ".!?":
        return sentence_stripped + " "

    return sentence_stripped + ". "
