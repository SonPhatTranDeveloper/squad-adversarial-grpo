import re

from utils.bert import BERT_QA
from utils.string import extract_answer, format_sentence

# Define the weights for the reduction metrics
F1_WEIGHT = 1.0
EXACT_MATCH_WEIGHT = 1.0
SPAN_DIFFERENCE_WEIGHT = 0.5


def format_reward(completions: list[list[dict[str, str]]], **kwargs: dict[str, any]) -> list[float]:
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


def bert_reward(completions: list[list[dict[str, str]]], **kwargs: dict[str, any]) -> list[float]:
    """
    Compute reward based on the reduction metrics of the BERT QA model.

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
    # Get the relevant items
    contexts = kwargs["contexts"]
    questions = kwargs["questions"]
    golden_answers = kwargs["answers"]
    golden_answers_start_idx = kwargs["answers_start_idx"]
    golden_answers_end_idx = kwargs["answers_end_idx"]

    # Get the completions
    completions = [completion[0]["content"] for completion in completions]

    # Get the answers
    answers = [extract_answer(completion) for completion in completions]

    # Create the modified contexts
    modified_contexts = [
        format_sentence(answer) + context
        for context, answer in zip(contexts, answers, strict=False)
    ]

    # Calculate the reduction metrics
    metrics = BERT_QA.calculate_reduction_metrics(
        contexts,
        modified_contexts,
        questions,
        golden_answers,
        golden_answers_start_idx,
        golden_answers_end_idx,
    )

    return [
        F1_WEIGHT * (-metric["f1"])
        + EXACT_MATCH_WEIGHT * (-metric["exact_match"])
        + SPAN_DIFFERENCE_WEIGHT * metric["span_difference"]
        for metric in metrics
    ]
