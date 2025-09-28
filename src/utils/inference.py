"""Inference utilities for a model trained with GRPO.

This module provides a high-level `GRPOInference` class that:
- Loads a base causal LM and (optionally) attaches LoRA adapters saved during GRPO training
- Constructs prompts consistent with the training format (system + user)
- Generates assistant responses and extracts the <answer> content (the adversarial sentence)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.utils.datasets import SYSTEM_PROMPT
from src.utils.string_utils import extract_answer

logger = logging.getLogger(__name__)


class GRPOInference:
    """
    Run inference with a GRPO-trained model (base + optional LoRA adapters).
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        device_map: str | dict[str, int] | None = "auto",
        dtype: any = torch.bfloat16,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> None:
        """Initialize model and tokenizer.

        Args:
            model_id: Hugging Face model id/path of the base causal LM.
            adapter_path: Optional path to LoRA adapter weights saved during training.
            device_map: Passed to `from_pretrained` for device placement.
            dtype: Optional dtype; if None, auto-selects bfloat16/float16 when available.
            max_new_tokens: Default generation max tokens.
            temperature: Default sampling temperature.
            top_p: Default top-p nucleus sampling.
            do_sample: Whether to sample during generation by default.
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )

        logger.info("Loading base model: %s", model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
        )

        if adapter_path and os.path.isdir(adapter_path):
            logger.info("Attaching LoRA adapters from: %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def _build_messages(self, context: str, question: str) -> list[dict[str, str]]:
        """Build chat-style messages consistent with training data.

        If `answer` is provided, we use the exact `create_user_prompt` used for training
        (which references the ground-truth answer for guidance). If not provided, we fall
        back to a version that omits the explicit answer but preserves requirements.

        Args:
            context: The context to be used for the adversarial sentence.
            question: The question to be used for the adversarial sentence.

        Returns:
            A list of chat-style messages.
        """
        user_prompt = (
            "You are asked to produce a single adversarial sentence to be added in front of the context\n"
            "so that a language model is likely to answer the question incorrectly.\n\n"
            "Requirements (must follow exactly):\n"
            "- The adversarial sentence must be in the same language, register, and style as the context and must be\n"
            "  coherent if read together with the context.\n"
            "- The adversarial sentence should be short-to-medium length (roughly 8–30 words) and grammatically well-formed.\n"
            "- The adversarial sentence should not contradict the context in any way.\n"
            "- Do not include explanation or comments in the <answer>, ONLY give the answer sentence inside the <answer> tags.\n\n"
            "Now here is the actual input you must work with:\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Now think carefully and produce the one adversarial sentence.\n"
            "Do not include explanation or comments in the <answer>, ONLY give the adversarial sentence inside the <answer> tags:"
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _messages_to_text(self, messages: list[dict[str, str]]) -> str:
        """
        Convert chat messages to a text prompt using tokenizer chat template if available.

        Args:
            messages: The list of chat messages to convert to a text prompt.

        Returns:
            A text prompt.
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_one(
        self,
        context: str,
        question: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> dict[str, Any]:
        """Generate one completion and extract the adversarial sentence.

        Args:
            context: The context to be used for the adversarial sentence.
            question: The question to be used for the adversarial sentence.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            do_sample: Whether to sample during generation.

        Returns:
            A dict with keys: {"prompt", "text", "answer"}.
        """
        messages = self._build_messages(context=context, question=question)
        prompt_text = self._messages_to_text(messages)

        # Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generation params
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "do_sample": self.do_sample if do_sample is None else do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # We generated continuation for the full prompt; the first item is the sequence
        full_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Extract only the assistant completion portion when chat template is used
        # If unavailable, fall back to extracting the <answer> block from the tail
        answer_text = extract_answer(full_text)

        return {
            "prompt": prompt_text,
            "text": full_text,
            "answer": answer_text,
        }
