import logging
import os

# Export path
import sys

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

sys.path.append("/workspace/squad-adversarial-grpo/")
from src.utils.datasets import load_csv_dataset
from src.utils.rewards import bert_reward, format_reward

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpo_training")

# Paths and config
DATASET_CSV = "/workspace/squad-adversarial-grpo/data/squad_golden.csv"
OUTPUT_DIR = "/workspace/squad-adversarial-grpo/outputs/qwen2.5-7b-grpo"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info("Output dir: %s", OUTPUT_DIR)


def main() -> None:
    """
    Main function for GRPO training.
    """
    # 1) Load dataset and adapt to conversation format expected by TRL GRPO
    raw_dataset: Dataset = load_csv_dataset(DATASET_CSV)
    raw_dataset = raw_dataset.shuffle(seed=42)
    train_dataset = raw_dataset.select(range(min(2000, len(raw_dataset))))
    logger.info("Train rows: %d", len(train_dataset))

    # Sanity check columns used by rewards
    required_cols = [
        "id",
        "context",
        "question",
        "answer",
        "answer_start_char_idx",
        "answer_end_char_idx",
        "prompt",
    ]
    missing = [c for c in required_cols if c not in train_dataset.column_names]
    if missing:
        raise ValueError(f"Missing columns for rewards: {missing}")

    # 2) Load base model and tokenizer
    # We apply LoRA for memory efficiency per HF guide
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    # Setup LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("Model with LoRA ready")

    # 3) Define GRPO config (adapted from HF guide)
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        remove_unused_columns=False,  # keep extra columns for reward kwargs
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        bf16=True,
        per_device_train_batch_size=1,
        # Preprocessing controls
        max_completion_length=256,  # shorter for quick runs
        num_generations=4,
        max_prompt_length=512,
        # Logging and saving
        report_to=["tensorboard"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
    )

    # 4) Instantiate GRPOTrainer with both rewards
    # format_reward expects the assistant completion content to match <think>...</think><answer>...</answer>
    # bert_reward uses the dataset columns to compute QA-based penalties
    reward_funcs = [format_reward, bert_reward]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 5) Train and save
    train_result = trainer.train()
    logger.info("Training complete: %s", str(train_result))

    trainer.save_model(OUTPUT_DIR)
    logger.info("Saved to %s", OUTPUT_DIR)
