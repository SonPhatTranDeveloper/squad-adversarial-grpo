import re

from src.utils.bert import BertQuestionAnswering
from src.utils.openai_model import AnswerabilityEvaluator
from src.utils.string_utils import extract_answer, format_sentence

# Define the weights for the reduction metrics
F1_WEIGHT = 1.0
EXACT_MATCH_WEIGHT = 1.0
SPAN_DIFFERENCE_WEIGHT = 0.5

# Rewards for format adherence
THINK_REWARD = 0.5
THINK_REWARD_MULTIPLE = 0.1
ANSWER_REWARD = 0.5
FULL_REWARD = 0.5


def format_reward(completions: list[list[dict[str, str]]], **kwargs: dict[str, any]) -> list[float]:
    """
    Reward function that checks if a completion contains <think>...</think> and
    <answer>...</answer> sections.

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
    pattern = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [bool(pattern.search(content)) for content in completion_contents]
    return [FULL_REWARD if match else 0.0 for match in matches]


def think_reward(completions: list[list[dict[str, str]]], **kwargs: dict[str, any]) -> list[float]:
    """
    Reward function that returns 1.0 if a completion contains a
    <think>...</think> section, else 0.0.

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
    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    completion_contents = [completion[0]["content"] for completion in completions]
    return [
        THINK_REWARD if think_pattern.search(content) else 0.0 for content in completion_contents
    ]


def answer_reward(completions: list[list[dict[str, str]]], **kwargs: dict[str, any]) -> list[float]:
    """
    Reward function that returns 1.0 if a completion contains an
    <answer>...</answer> section, else 0.0.

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
    answer_pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    completion_contents = [completion[0]["content"] for completion in completions]
    return [
        ANSWER_REWARD if answer_pattern.search(content) else 0.0 for content in completion_contents
    ]


def multi_think_reward(
    completions: list[list[dict[str, str]]], **kwargs: dict[str, any]
) -> list[float]:
    """
    Reward function that promotes multiple <think>...</think> sections.

    Each <think> block contributes `THINK_REWARD` to the score, capped at
    `THINK_REWARD` to keep the scale comparable to other rewards.

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
    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards: list[float] = []
    for content in completion_contents:
        count = len(think_pattern.findall(content))
        reward = min(count * THINK_REWARD_MULTIPLE, THINK_REWARD)
        rewards.append(reward)
    return rewards


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
    completions = [completion[-1]["content"] for completion in completions]

    # Get the answers
    answers = [extract_answer(completion) for completion in completions]

    # Create the modified contexts
    modified_contexts = [
        format_sentence(answer) + context
        for context, answer in zip(contexts, answers, strict=False)
    ]

    print(f"Completion: {completions[0]}")
    print(f"Answer: {answers[0]}")
    print(f"Modified context: {modified_contexts[0]}")

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


_answerability_evaluator: AnswerabilityEvaluator | None = None


def answerability_reward(
    completions: list[list[dict[str, str]]], **kwargs: dict[str, any]
) -> list[float]:
    """
    Reward function that checks if a completion is answerable.

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
    questions = kwargs["question"]
    completions = [completion[-1]["content"] for completion in completions]
    new_sentences = [extract_answer(completion) for completion in completions]

    # Lazily initialize AnswerabilityEvaluator to avoid heavy init at import
    global _answerability_evaluator
    if _answerability_evaluator is None:
        _answerability_evaluator = AnswerabilityEvaluator()

    return _answerability_evaluator.get_rewards(questions, new_sentences)


def contain_explanation_reward(
    completions: list[list[dict[str, str]]], **kwargs: dict[str, any]
) -> list[float]:
    """
    Reward function that checks if a completion contains an explanation.

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
    completions = [completion[-1]["content"] for completion in completions]
    answers = [extract_answer(completion) for completion in completions]

    # Check for some vocabulary like "model", "context", "answer",
    return [
        -2.0 if "model" in answer or "context" in answer or "answer" in answer else 0.0
        for answer in answers
    ]


def answer_length_reward(
    completions: list[list[dict[str, str]]], **kwargs: dict[str, any]
) -> list[float]:
    """
    Reward that encourages short answers inside <answer>...</answer>.

    The reward is computed on the extracted answer text (without tags).
    By default, we softly penalize lengths above a target using a hinge:

        reward = -max(0, (len - target) / target) * scale

    Where length is measured in words. This yields 0 for answers at or
    below target length, and increasingly negative values above.

    Args:
        completions: A list of conversations with the last message being the
            assistant completion. Shape:
            [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...]
        **kwargs: Optional parameters:
            - target_words: int, desired maximum word count (default: 12)
            - scale: float, penalty multiplier (default: 1.0)

    Returns:
        List of float rewards aligned with completions.
    """
    target_words: int = 40
    scale: float = 1.0

    answers = [extract_answer(conv[-1]["content"]) for conv in completions]

    rewards: list[float] = []
    for answer in answers:
        # Count words by simple whitespace split; robust and fast
        word_count = len(answer.split())
        excess = max(0, word_count - target_words)
        # Linear penalty scaled by how much it exceeds the target
        penalty = (excess / max(1, target_words)) * scale
        rewards.append(-penalty)
    return rewards
