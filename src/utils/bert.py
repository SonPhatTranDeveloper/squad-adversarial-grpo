import logging

import evaluate
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

logger = logging.getLogger(__name__)


class BertQuestionAnswering:
    def __init__(self, model_name: str) -> None:
        """Initialize the QA model, tokenizer, pipeline, and metrics.

        Args:
            model_name: Hugging Face model name or path for a QA model.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline: Pipeline = pipeline(
            task="question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
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

    def calculate_reduction_metrics(
        self,
        contexts: list[str],
        modified_contexts: list[str],
        questions: list[str],
        golden_answers: list[str],
        golden_answers_start_idx: list[int],
        golden_answers_end_idx: list[int],
    ) -> dict[str, float]:
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

        # Run QA pipeline to get detailed predictions (answer text and spans)
        original_inputs = [
            {"context": context, "question": question}
            for context, question in zip(contexts, questions, strict=False)
        ]
        modified_inputs = [
            {"context": context, "question": question}
            for context, question in zip(modified_contexts, questions, strict=False)
        ]

        logger.debug("Running QA pipeline on original contexts: %d items", len(original_inputs))
        original_results = self.qa_pipeline(original_inputs)
        if isinstance(original_results, dict):
            original_results = [original_results]

        logger.debug("Running QA pipeline on modified contexts: %d items", len(modified_inputs))
        modified_results = self.qa_pipeline(modified_inputs)
        if isinstance(modified_results, dict):
            modified_results = [modified_results]

        original_answers = [r.get("answer", "") for r in original_results]
        modified_answers = [r.get("answer", "") for r in modified_results]

        # Compute F1/EM reduction directly
        original_metrics = self._calculate_metrics(original_answers, golden_answers)
        modified_metrics = self._calculate_metrics(modified_answers, golden_answers)

        # Compute average span difference reduction across the batch
        span_reductions: list[int] = []
        for idx, (orig, mod) in enumerate(zip(original_results, modified_results, strict=False)):
            original_start = int(orig.get("start", -1))
            original_end = int(orig.get("end", -1))
            modified_start = int(mod.get("start", -1))
            modified_end = int(mod.get("end", -1))
            gold_start = int(golden_answers_start_idx[idx])
            gold_end = int(golden_answers_end_idx[idx])

            original_diff = abs(original_start - gold_start) + abs(original_end - gold_end)
            modified_diff = abs(modified_start - gold_start) + abs(modified_end - gold_end)
            span_reductions.append(abs(original_diff - modified_diff))

        avg_span_reduction = (
            float(sum(span_reductions)) / float(len(span_reductions)) if span_reductions else 0.0
        )

        return {
            "f1": float(original_metrics["f1"] - modified_metrics["f1"]),
            "exact_match": float(original_metrics["exact_match"] - modified_metrics["exact_match"]),
            "span_difference": avg_span_reduction,
        }
