from datasets import Dataset, load_dataset


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


def load_csv_dataset(path: str) -> Dataset:
    """Load dataset into a Hugging Face Dataset.

    Args:
        path: Path to the dataset (csv file).

    Returns:
        A Hugging Face Dataset.
    """
    # Load dataset
    dataset = load_dataset("csv", data_files=path)

    # Assert if the dataset has the correct columns
    assert_dataset_has_correct_columns(dataset)

    # Remove columns
    dataset = dataset.remove_columns(["answer_start_word_idx", "answer_end_word_idx"])

    return dataset
