import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample N random rows from a CSV using pandas.")
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to output CSV file")
    parser.add_argument("--num_rows", required=True, type=int, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    return parser.parse_args()


def simple_sample(df: pd.DataFrame, n: int, seed: int | None) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("--num_rows must be a positive integer")
    if n > len(df):
        raise ValueError(
            f"Requested --num_rows={n} exceeds dataset size {len(df)} for sampling without replacement"
        )
    return df.sample(n=n, replace=False, random_state=seed)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_path)
    sampled = simple_sample(df=df, n=args.num_rows, seed=args.seed)
    sampled.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
