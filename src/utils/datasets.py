from collections.abc import Callable

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
"first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
"process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
"<think> reasoning process here </think><answer> answer here </answer>
"""


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
    """Create a user prompt for the dataset."""
    return f"""You are tasked with generate an adversarial sentence to add to the start of the context 
    in the SQuAD dataset to fool an LLM to answer the question incorrectly.

    The context is: {context}
    The question is: {question}
    The answer is: {answer} 

    - The adversarial sentence should be added to the start of the context 
      to fool the LLM to answer the question incorrectly.
    - The adversarial sentence should be in the same language as the context.
    - The adversarial sentence should be in the same style/coherent with the context.
    """


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
