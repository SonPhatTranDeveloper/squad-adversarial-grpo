from collections.abc import Callable

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


def assert_dataset_has_correct_columns(dataset: Dataset) -> None:
    """Assert if the dataset has the correct columns."""
    assert "id" in dataset.column_names, "Dataset must have an 'id' column"
    assert "answer_start_char_idx" in dataset.column_names, (
        "Dataset must have an 'answer_start_char_idx' column"
    )
    assert "answer_end_char_idx" in dataset.column_names, (
        "Dataset must have an 'answer_end_char_idx' column"
    )
    assert "answer_start_word_idx" in dataset.column_names, (
        "Dataset must have a 'answer_start_word_idx' column"
    )
    assert "answer_end_word_idx" in dataset.column_names, (
        "Dataset must have a 'answer_end_word_idx' column"
    )
    assert "context" in dataset.column_names, "Dataset must have a 'context' column"
    assert "question" in dataset.column_names, "Dataset must have a 'question' column"
    assert "answer" in dataset.column_names, "Dataset must have an 'answer' column"


def create_user_prompt(context: str, question: str, answer: str) -> str:
    """
    Create a user prompt for an adversarial-example generator.

    The generator should produce a single adversarial sentence to prepend to `context`
    that is fluent and coherent with the context but is likely to cause a language
    model to answer `question` incorrectly. The generator must output exactly one
    sentence and nothing else.
    """
    return f"""You are asked to produce a single adversarial sentence to be prepended to a passage
so that a language model is likely to answer the question incorrectly.

Requirements (must follow exactly):
- The sentence must be in the same language, register, and style as the context and must be
  coherent if read together with the context.
- The sentence should be plausible and natural (not nonsensical or obviously contradictory).
- Do NOT repeat the question or the ground-truth answer verbatim.
- The adversarial sentence should be short-to-medium length (roughly 8–30 words) and grammatically well-formed.
- The sentence should be crafted to subtly bias or mislead a language model toward an incorrect answer
  while remaining contextually plausible.

Examples (for illustration only, do NOT output these):
1. Context: "In the ninth inning, the crowd rose as the home team celebrated their comeback win led by García."
   Question: Who hit the winning home run?
   Answer: García
   Adversarial sentence: "It was López whose clutch swing in the eight helped the team to come back."

2. Context: "The treaty was signed on July 14, 1789, marking a turning point in the conflict."
   Question: On what date was the treaty signed?
   Answer: July 14, 1789
   Adversarial sentence: "After being signed, the treaty brought the conflict to an end on July 28, 1789."

3. Context: "The first patient, John, was diagnosed with pneumonia and started on a course of antibiotics."
   Question: What was the first patient diagnosed with?
   Answer: Pneumonia
   Adversarial sentence: "The records instead show the second patient was diagnosed with bronchitis and given no antibiotics."

Now here is the actual input you must work with:

Context:
{context}

Question:
{question}

Ground-truth answer (for reference only — do not output it):
{answer}

Now produce the one adversarial sentence:"""


def map_to_conversation(
    dataset: Dataset, user_prompt_creator: Callable[[str, str, str], str]
) -> Dataset:
    """Map the dataset to a conversation format."""

    def make_conversation(x: dict) -> dict:
        """Make a conversation from a row of the dataset."""
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt_creator(x["context"], x["question"], x["answer"]),
                },
            ]
        }

    # Add a new column to the dataset with the conversation
    return dataset.map(make_conversation)


def load_csv_dataset(path: str) -> Dataset:
    """Load dataset into a Hugging Face Dataset.

    Args:
        path: Path to the dataset (csv file).

    Returns:
        A Hugging Face Dataset.
    """
    # Load dataset
    dataset = load_dataset("csv", data_files=path, split="train")

    # Assert if the dataset has the correct columns
    assert_dataset_has_correct_columns(dataset)

    # Remove columns
    dataset = dataset.remove_columns(["answer_start_word_idx", "answer_end_word_idx"])

    # Map to conversation
    dataset = map_to_conversation(dataset, create_user_prompt)

    return dataset
