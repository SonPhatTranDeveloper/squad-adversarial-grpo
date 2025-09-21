import logging

import evaluate
from transformers import (
    pipeline,
)

logger = logging.getLogger(__name__)


def _format_box(lines: list[str]) -> str:
    """Render a simple ASCII box around the provided lines.

    Args:
        lines: Content lines to render inside the box. Lines will be padded
            to the width of the longest line.

    Returns:
        A single string containing the boxed content with newlines.
    """
    if not lines:
        return ""
    max_width = max(len(line) for line in lines)
    top_border = "+" + "-" * (max_width + 2) + "+"
    bottom_border = top_border
    boxed_lines = [top_border]
    for line in lines:
        padded = line.ljust(max_width)
        boxed_lines.append(f"| {padded} |")
    boxed_lines.append(bottom_border)
    return "\n".join(boxed_lines)


class BertQuestionAnswering:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-distilled-squad",
        device: str = "cuda",
        max_length: int = 512,
        max_answer_length: int = 30,
    ) -> None:
        """Initialize the QA model, tokenizer, pipeline, and metrics.

        Args:
            model_name: Hugging Face model name or path for a QA model.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,
            max_seq_len=self.max_length,
            max_answer_len=self.max_answer_length,
        )
        self.squad_metric = evaluate.load("squad")

    def _calculate_metrics(self, answers: list[str], golden_answers: list[str]) -> dict[str, float]:
        """Compute SQuAD F1 and Exact Match using the metrics library.

        Args:
            answers: List of system-predicted answers.
            golden_answers: List of ground-truth answers.

        Returns:
            Dict with keys "f1" and "exact_match" representing aggregate scores.
        """
        assert len(answers) == len(golden_answers), "answers and golden_answers must align"

        predictions = [
            {"id": str(idx), "prediction_text": pred} for idx, pred in enumerate(answers)
        ]
        references = [
            {
                "id": str(idx),
                # Start indices are unused by F1/EM; populate with a dummy value.
                "answers": {"text": [gold], "answer_start": [0]},
            }
            for idx, gold in enumerate(golden_answers)
        ]
        metrics = self.squad_metric.compute(predictions=predictions, references=references)
        # metrics contains keys: "exact_match", "f1"
        return {
            "f1": float(metrics.get("f1", 0.0)),
            "exact_match": float(metrics.get("exact_match", 0.0)),
        }

    def get_answers(self, contexts: list[str], questions: list[str]) -> list[dict[str, any]]:
        """
        Get the answers for the given contexts and questions.

        Args:
            contexts: The contexts to answer the question.
            questions: The questions to answer.

        Returns:
            The answers for the given contexts and questions.
        """
        inputs = [
            {"context": context, "question": question}
            for context, question in zip(contexts, questions, strict=False)
        ]
        return self.qa_pipeline(inputs)

    def calculate_reduction_metrics(
        self,
        contexts: list[str],
        modified_contexts: list[str],
        questions: list[str],
        golden_answers: list[str],
        golden_answers_start_idx: list[int],
        golden_answers_end_idx: list[int],
    ) -> list[dict[str, float]]:
        """
        Calculate the reduction metrics for the answer.

        Args:
            contexts: The contexts to answer the question.
            modified_contexts: The modified contexts to answer the question.
            questions: The questions to answer.
            golden_answers: The golden answers to the question.

        Returns:
            The reduction metrics for the answer.
            - f1: The F1 score for the answer.
            - exact_match: The exact match for the answer.
            - span_difference: The span difference for the answer.
        """
        assert len(contexts) == len(modified_contexts) == len(questions) == len(golden_answers), (
            "All input lists must have the same length"
        )
        assert len(golden_answers_start_idx) == len(golden_answers), (
            "golden_answers_start_idx must align with golden_answers"
        )
        assert len(golden_answers_end_idx) == len(golden_answers), (
            "golden_answers_end_idx must align with golden_answers"
        )

        # Run QA pipeline only on modified contexts and compare to golden answers
        logger.debug("Running QA pipeline on modified contexts: %d items", len(modified_contexts))
        modified_results = self.get_answers(modified_contexts, questions)
        if isinstance(modified_results, dict):
            modified_results = [modified_results]

        per_item_metrics: list[dict[str, float]] = []
        for idx, mod in enumerate(modified_results):
            modified_answer = str(mod.get("answer", ""))

            # Per-item F1/EM against golden answer
            modified_metrics = self._calculate_metrics([modified_answer], [golden_answers[idx]])

            # Span difference relative to golden span adjusted by context length delta
            modified_start = int(mod.get("start", -1))
            modified_end = int(mod.get("end", -1))
            gold_start = int(golden_answers_start_idx[idx])
            gold_end = int(golden_answers_end_idx[idx])
            answers_match = modified_answer == golden_answers[idx]
            box = _format_box(
                [
                    "Modified answer details",
                    f"Modified answer: {modified_answer}",
                    f"Golden answer:   {golden_answers[idx]}",
                    f"Answer match (==): {'YES ✅' if answers_match else 'NO ❌'}",
                    f"Modified start:  {modified_start}",
                    f"Modified end:    {modified_end}",
                    f"Gold start:      {gold_start}",
                    f"Gold end:        {gold_end}",
                ]
            )
            logger.info("\n%s", box)

            original_context_length = len(contexts[idx])
            modified_context_length = len(modified_contexts[idx])
            context_length_difference = abs(original_context_length - modified_context_length)

            span_difference = abs(modified_start - (gold_start + context_length_difference)) + abs(
                modified_end - (gold_end + context_length_difference)
            )

            per_item_metrics.append(
                {
                    "f1": float(modified_metrics["f1"]),
                    "exact_match": float(modified_metrics["exact_match"]),
                    "span_difference": float(int(span_difference)) / modified_context_length,
                }
            )

        return per_item_metrics
