"""Entry point for the grpo-adv-prompter-squad package."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the CLI entry point.

    Args:
        None

    Returns:
        None
    """
    logger.info("Hello from grpo-adv-prompter-squad!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
