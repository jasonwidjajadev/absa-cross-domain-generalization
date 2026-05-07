import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for sampling.
    parser = argparse.ArgumentParser(description="Randomly sample up to 3000 Yelp reviews.")
    parser.add_argument("--input-file", default="yelp_reviews.csv", help="Input CSV file.")
    parser.add_argument(
        "--output-file", default="sampled_3000_reviews.csv", help="Output sampled CSV file."
    )
    parser.add_argument(
        "--sample-size", type=int, default=3000, help="Maximum number of rows to sample."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    # Load input CSV, sample rows, and save output CSV.
    args = parse_args()
    df = pd.read_csv(args.input_file)
    print(f"Total rows: {len(df)}")

    sample_size = min(args.sample_size, len(df))
    sampled_df = df.sample(n=sample_size, random_state=args.seed)
    sampled_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")

    print(f"Saved {sample_size} sampled reviews to {args.output_file}")


if __name__ == "__main__":
    main()
