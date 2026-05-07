from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Keep only model-relevant columns and drop metadata/index fields.
COMMON_COLUMNS = ["sentence_id", "text", "target", "polarity", "label"]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    # Parse input/output dataset directory arguments.
    parser = argparse.ArgumentParser(
        description=(
            "Combine two dataset directories into one output directory by merging "
            "train.csv, val.csv, and test.csv."
        )
    )
    parser.add_argument(
        "--data-dir-a",
        type=Path,
        required=True,
        help="First dataset directory containing train.csv, val.csv, test.csv.",
    )
    parser.add_argument(
        "--data-dir-b",
        type=Path,
        required=True,
        help="Second dataset directory containing train.csv, val.csv, test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for merged train.csv, val.csv, test.csv.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    # Read all rows from one split CSV.
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def project_columns(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # Keep only shared model columns in each row.
    return [{col: (row.get(col) or "") for col in COMMON_COLUMNS} for row in rows]


def combine_and_write(split: str, dir_a: Path, dir_b: Path, output_dir: Path) -> None:
    # Merge one split from both dirs and write combined CSV.
    path_a = dir_a / f"{split}.csv"
    path_b = dir_b / f"{split}.csv"
    out_path = output_dir / f"{split}.csv"

    rows_a = project_columns(read_rows(path_a))
    rows_b = project_columns(read_rows(path_b))
    combined_rows = rows_a + rows_b

    output_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMMON_COLUMNS)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"[{split}] rows from dir A: {len(rows_a)}")
    print(f"[{split}] rows from dir B: {len(rows_b)}")
    print(f"[{split}] combined rows: {len(combined_rows)}")
    print(f"[{split}] saved: {out_path}")


def main() -> None:
    # Run merge for train/val/test splits.
    args = parse_args()
    dir_a = args.data_dir_a.expanduser().resolve()
    dir_b = args.data_dir_b.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for split in SPLITS:
        combine_and_write(split, dir_a, dir_b, output_dir)


if __name__ == "__main__":
    main()
