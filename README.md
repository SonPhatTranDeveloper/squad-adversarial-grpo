
## Linting & Formatting

This project uses Ruff for linting and formatting.

- Install dev dependencies:

```bash
uv pip install -e ".[dev]"
```

- Run linter and auto-fix issues:

```bash
ruff check --fix .
```

- Format code:

```bash
ruff format .
```

Configuration is in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`.

