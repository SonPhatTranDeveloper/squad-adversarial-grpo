## GRPO Adversarial Prompter (SQuAD)

A small framework to generate adversarial prompts for SQuAD-style question answering and evaluate their impact using BERT QA metrics. It uses Hydra for configuration, Hugging Face Datasets/Transformers, and Evaluate for SQuAD metrics. Development uses Ruff for linting/formatting.

### Features
- Load SQuAD-like CSV into a Hugging Face `Dataset`
- Validate required columns and map rows into a conversation-style prompt
- Generate user prompts instructing an adversarial example generator
- Reward utilities including:
  - `format_reward`: structure-checking for `<think>` and `<answer>` tags
  - `bert_reward`: computes negative F1/EM and span difference via a QA pipeline
- Configuration via Hydra (`src/config/**`)
- Dev tooling with Ruff

### Installation
- Python 3.12+
- Recommended: `uv` for fast, reproducible installs

```bash
# Install runtime dependencies
uv pip install -e .

# (Optional) install dev tools (ruff)
uv pip install -e ".[dev]"
```

### Quickstart
The entrypoint uses Hydra to load configuration and dataset, then prints the mapped dataset.

```bash
python main.py
```

Hydra searches configs in `src/config`. Default config file is `main_config.yaml`.

### Configuration
- `src/config/main_config.yaml`
  - Uses defaults and selects dataset config `dataset: squad`
- `src/config/dataset/squad.yaml`
  - `path`: absolute path to the CSV file, e.g. `/Users/sonphat.tran/grpo_adv_prompter_squad/data/squad_golden.csv`
- `src/config/model/qwen.yaml`
  - Placeholder for future model settings (currently empty)

You can override config values from CLI, for example:

```bash
python main.py dataset.path=/absolute/path/to/your.csv
```

### Dataset Schema (CSV)
Input CSV must contain one JSON object per row loaded via `datasets.load_dataset("csv")` with split `train`. Required columns:
- `id`
- `context`
- `question`
- `answer`
- `answer_start_char_idx`
- `answer_end_char_idx`
- `answer_start_word_idx`
- `answer_end_word_idx`

During loading:
- Columns `answer_start_word_idx` and `answer_end_word_idx` are removed
- A new `prompt` column is created with a conversation structure:
  - `role: system` content defines the reasoning and answer format using `<think>` and `<answer>` tags
  - `role: user` content asks for one adversarial sentence to prepend to the context

### Prompting Format
System message (`SYSTEM_PROMPT`) instructs the assistant to think in `<think>` and answer in `<answer>`. The user prompt requests exactly one adversarial sentence (8–30 words) coherent with the context and designed to mislead a model on the question.

### Rewards
Utilities in `src/utils/rewards.py`:
- `format_reward(completions, **kwargs) -> list[float]`
  - Returns 1.0 if a completion matches the required tag structure, else 0.0
- `bert_reward(completions, **kwargs) -> list[float]`
  - Uses a lazily-initialized QA pipeline (`distilbert-base-uncased-distilled-squad`) to answer questions on modified contexts (adversarial sentence + original context)
  - Computes metrics per item and returns a weighted score: `-F1 - EM + 0.5 * span_difference`

Supporting modules:
- `src/utils/string.py`: `extract_answer`, `format_sentence`
- `src/utils/bert.py`: QA pipeline wrapper and SQuAD metric computation using `evaluate`

### Notebooks
- `src/notebooks/grpo_training.ipynb`: scratchpad for GRPO-style training experiments. Ensure the environment matches the package versions from `pyproject.toml`.

### Development
Ruff is configured in `pyproject.toml` with Google-style docstrings and type hints.

```bash
# Lint and autofix
ruff check --fix .

# Format
ruff format .
```

Conventions:
- Prefer absolute imports under the project root (e.g., `from src.utils.datasets import load_csv_dataset`)
- Functions should include type hints and Google-style docstrings with Args/Returns
- Keep try/except minimal; prefer a module-level logger over prints

### Project Structure
- `main.py`: Hydra entrypoint; loads dataset via `load_csv_dataset` and prints it
- `src/utils/datasets.py`: schema checks, conversation mapping, user prompt creation
- `src/utils/rewards.py`: reward functions (`format_reward`, `bert_reward`)
- `src/utils/bert.py`: BERT QA wrapper and SQuAD metric helpers
- `src/utils/string.py`: parsing and string helpers
- `src/config/**`: Hydra configs (dataset, model, main)
- `data/`: place your CSV data (see `src/config/dataset/squad.yaml`)
- `outputs/`: Hydra output directory

### Notes
- QA metrics rely on downloading models/metrics from Hugging Face on first use
- Ensure the dataset path in `squad.yaml` points to your CSV
- GPU will be auto-detected by `transformers.pipeline` when available

