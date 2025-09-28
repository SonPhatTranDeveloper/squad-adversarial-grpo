import argparse
import logging
import os

import pandas as pd
import torch

from src.utils.inference import GRPOInference
from src.utils.string_utils import format_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load a CSV with columns id,title,context,question,answer,"
            "answer_start_char_idx,answer_end_char_idx,answer_start_word_idx,answer_end_word_idx;"
            " run GRPO inference on (context, question), prepend generated sentence to question,"
            " and save to a new CSV."
        )
    )
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to output CSV file")

    # Optional model settings with sensible defaults mirroring example
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base HF model id/path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/qwen2.5-3b-grpo/checkpoint-2000",
        help="Optional LoRA adapter directory (if using GRPO adapters)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        help="Device map for model loading (e.g., 'auto', 'cuda')",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.set_defaults(do_sample=True)

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--show_progress", action="store_true", help="Show tqdm progress bar")

    return parser.parse_args()


def build_infer(args: argparse.Namespace) -> GRPOInference:
    """
    Build a GRPOInference object.

    Args:
        args: The arguments.

    Returns:
        A GRPOInference object.
    """
    return GRPOInference(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        device_map=args.device_map,
        dtype=torch.bfloat16,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )


def prepend_answer_to_question(
    df: pd.DataFrame, infer: GRPOInference, batch_size: int, show_progress: bool
) -> pd.DataFrame:
    """
    Prepend the adversarial sentence to the question.

    Args:
        df: The dataframe to augment.
        infer: The GRPOInference object.

    Returns:
        A dataframe with the augmented questions.
    """
    contexts = df["context"].astype(str).tolist()
    questions = df["question"].astype(str).tolist()

    results = infer.generate_many(
        contexts=contexts,
        questions=questions,
        batch_size=batch_size,
        show_progress=show_progress,
        max_new_tokens=infer.max_new_tokens,
        temperature=infer.temperature,
        top_p=infer.top_p,
        do_sample=infer.do_sample,
    )

    adv_sentences = [format_sentence(item.get("answer", "")) for item in results]
    new_contexts = [f"{a}{q}" if a else q for a, q in zip(adv_sentences, contexts, strict=False)]

    df_out = df.copy()
    df_out["context"] = new_contexts
    return df_out


def create_folder_if_not_exists(path: str) -> None:
    """
    Create a folder if it does not exist.

    Args:
        path: The path to the folder.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def main() -> None:
    """
    Main function.
    """
    args = parse_args()

    create_folder_if_not_exists(args.output_path)

    df = pd.read_csv(args.input_path)
    infer = build_infer(args)

    df_out = prepend_answer_to_question(df, infer, args.batch_size, args.show_progress)
    df_out.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
