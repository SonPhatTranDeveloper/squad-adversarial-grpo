import re

from src.utils.bert import BertQuestionAnswering
from src.utils.string import extract_answer, format_sentence

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


_qa_model: BertQuestionAnswering | None = None


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
    # Get the relevant items from dataset columns
    contexts = kwargs["context"]
    questions = kwargs["question"]
    golden_answers = kwargs["answer"]
    golden_answers_start_idx = kwargs["answer_start_char_idx"]
    golden_answers_end_idx = kwargs["answer_end_char_idx"]

    # Get the completions
    completions = [completion[0]["content"] for completion in completions]

    # Get the answers
    answers = [extract_answer(completion) for completion in completions]

    # Create the modified contexts
    modified_contexts = [
        format_sentence(answer) + context
        for context, answer in zip(contexts, answers, strict=False)
    ]

    # Lazily initialize QA model to avoid heavy init at import
    global _qa_model
    if _qa_model is None:
        _qa_model = BertQuestionAnswering()

    # Calculate the reduction metrics
    metrics = _qa_model.calculate_reduction_metrics(
        contexts=contexts,
        modified_contexts=modified_contexts,
        questions=questions,
        golden_answers=golden_answers,
        golden_answers_start_idx=golden_answers_start_idx,
        golden_answers_end_idx=golden_answers_end_idx,
    )

    return [
        F1_WEIGHT * (100.0 - metric["f1"]) / 100.0
        + EXACT_MATCH_WEIGHT * (100.0 - metric["exact_match"]) / 100.0
        + SPAN_DIFFERENCE_WEIGHT * metric["span_difference"]
        for metric in metrics
    ]
