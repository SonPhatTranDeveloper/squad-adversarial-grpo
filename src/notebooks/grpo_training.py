import logging
import os
from collections.abc import Callable

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from trl import GRPOConfig, GRPOTrainer

from src.utils.datasets import load_csv_dataset
from src.utils.rewards import (
    bert_reward,
    contain_explanation_reward,
    format_reward,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpo_training")

# Paths and config (override via environment variables if set)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/workspace/squad-adversarial-grpo/",
)
DATASET_CSV = os.environ.get(
    "DATASET_CSV",
    os.path.join(PROJECT_ROOT, "data", "squad_golden.csv"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "qwen2.5-3b-grpo"),
)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info("Output dir: %s", OUTPUT_DIR)


def load_train_dataset(dataset_csv: str, max_rows: int = 2000, seed: int = 42) -> Dataset:
    """Load, shuffle, and subsample the training dataset.

    Args:
        dataset_csv: Absolute path to the dataset CSV file.
        max_rows: Maximum number of rows to select for training.
        seed: Seed for dataset shuffling.

    Returns:
        A `datasets.Dataset` ready for GRPO training.
    """
    raw_dataset: Dataset = load_csv_dataset(dataset_csv)
    raw_dataset = raw_dataset.shuffle(seed=seed)
    train_dataset = raw_dataset.select(range(min(max_rows, len(raw_dataset))))
    logger.info("Train rows: %d", len(train_dataset))
    return train_dataset


def create_lora_model(model_id: str) -> PreTrainedModel:
    """Create a base causal LM and wrap it with LoRA adapters.

    Args:
        model_id: Hugging Face model identifier to load as the base model.

    Returns:
        A `transformers.PreTrainedModel` with LoRA adapters applied.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("Model with LoRA ready")
    return model


def create_grpo_config(output_dir: str) -> GRPOConfig:
    """Create GRPO training configuration.

    Args:
        output_dir: Directory where checkpoints and logs will be written.

    Returns:
        A configured `trl.GRPOConfig` instance.
    """
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        remove_unused_columns=False,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        bf16=True,
        per_device_train_batch_size=1,
        temperature=1.0,
        # Preprocessing controls
        max_completion_length=512,
        num_generations=16,
        max_prompt_length=4096,
        # Logging and saving
        report_to=["tensorboard"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
    )


def create_trainer(
    model: PreTrainedModel,
    train_dataset: Dataset,
    args: GRPOConfig,
) -> GRPOTrainer:
    """Construct a GRPOTrainer with reward functions.

    Args:
        model: The LoRA-wrapped pretrained model to train.
        train_dataset: The dataset to use for training.
        args: The GRPO configuration.

    Returns:
        An initialized `trl.GRPOTrainer` instance.
    """
    reward_funcs: list[Callable[..., list[float]]] = [
        format_reward,
        bert_reward,
        contain_explanation_reward,
        answer_length_reward,
    ]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
    )
    return trainer


def train_and_save(trainer: GRPOTrainer, output_dir: str) -> None:
    """Run training and save the final model to disk.

    Args:
        trainer: The configured GRPO trainer instance.
        output_dir: Output directory to save the trained model.

    Returns:
        None
    """
    train_result = trainer.train()
    logger.info("Training complete: %s", str(train_result))
    trainer.save_model(output_dir)
    logger.info("Saved to %s", output_dir)


def main() -> None:
    """Run the full GRPO training workflow end-to-end.

    Args:
        None

    Returns:
        None
    """
    train_dataset = load_train_dataset(DATASET_CSV, max_rows=2000, seed=42)
    model = create_lora_model(MODEL_ID)
    training_args = create_grpo_config(OUTPUT_DIR)
    trainer = create_trainer(model=model, train_dataset=train_dataset, args=training_args)
    train_and_save(trainer=trainer, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
