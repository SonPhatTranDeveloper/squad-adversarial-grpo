from src.utils.datasets import load_csv_dataset
import hydra


@hydra.main(config_path="src/config", config_name="main_config.yaml")
def main() -> None:
    """Main function."""
    dataset = load_csv_dataset(config.dataset.path)


if __name__ == "__main__":
    main()
