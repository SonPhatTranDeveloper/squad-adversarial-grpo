import hydra
from omegaconf import DictConfig

from src.utils.datasets import load_csv_dataset


@hydra.main(version_base=None, config_path="src/config", config_name="main_config.yaml")
def main(config: DictConfig) -> None:
    """Main function."""
    dataset = load_csv_dataset(config.dataset.path)
    print(dataset)


if __name__ == "__main__":
    main()
