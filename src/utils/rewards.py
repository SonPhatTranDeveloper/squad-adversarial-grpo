import re


def format_reward(completions: list[list[dict[str, str]]], **kwargs: any) -> list[float]:
    """
    Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    while the final answer is enclosed within <answer> and </answer> tags.

    Args:
        completions: List of completions of the format:
        [
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        ]

    Returns:
        List of rewards.
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]
