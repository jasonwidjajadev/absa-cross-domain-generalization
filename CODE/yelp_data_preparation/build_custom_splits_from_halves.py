from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DIR = ROOT / "data" / "custom"

OUTPUT_COLUMNS = [
    "source_year",
    "domain",
    "sentence_id",
    "text",
    "target",
    "polarity",
]


def clean(x: str | None) -> str:
    # Normalize a CSV cell into a stripped string.
    if x is None:
        return ""
    return x.strip()


def normalize_record(row: dict[str, str]) -> dict[str, str]:
    # Convert one input row to the fixed output schema.
    sentence_id = clean(row["sentence_id"])
    text = clean(row["text"])
    target = clean(row["target"])
    polarity = clean(row["polarity"]).lower()
    return {
        "source_year": "custom",
        "domain": "custom",
        "sentence_id": sentence_id,
        "text": text,
        "target": target,
        "polarity": polarity,
    }


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    # Write rows to CSV with fixed headers.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for half-file merge and split.
    p = argparse.ArgumentParser(
        description=(
            "Merge two half CSVs into one custom dataset, then split into "
            "train/val/test with ratio 80/10/10."
        )
    )
    p.add_argument(
        "--input-half1",
        type=Path,
        default=CUSTOM_DIR / "finalized_1-1500.csv",
        help="First half CSV path.",
    )
    p.add_argument(
        "--input-half2",
        type=Path,
        default=CUSTOM_DIR / "final_dataset.csv",
        help="Second half CSV path.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=CUSTOM_DIR,
        help="Output directory for merged_two_halves.csv and train/val/test CSV files.",
    )
    return p.parse_args()


def main() -> None:
    # Merge two halves, then create train/val/test splits.
    # Assumes each input row is already expanded.
    args = parse_args()
    input_files = [args.input_half1.resolve(), args.input_half2.resolve()]
    output_dir = args.output_dir.resolve()

    merged_rows: list[dict[str, str]] = []
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged_rows.append(normalize_record(row))

    if not merged_rows:
        raise ValueError("No rows produced after merging the two halves.")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = output_dir / "merged_two_halves.csv"
    write_csv(merged_csv, OUTPUT_COLUMNS, merged_rows)

    rows = list(merged_rows)
    rng = random.Random(42)
    rng.shuffle(rows)
    n = len(rows)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    write_csv(train_path, OUTPUT_COLUMNS, train_rows)
    write_csv(val_path, OUTPUT_COLUMNS, val_rows)
    write_csv(test_path, OUTPUT_COLUMNS, test_rows)

    print(f"Merged rows: {len(merged_rows)} -> {merged_csv}")
    print(f"Train rows (80%): {len(train_rows)} -> {train_path}")
    print(f"Val rows (10%): {len(val_rows)} -> {val_path}")
    print(f"Test rows (10%): {len(test_rows)} -> {test_path}")


if __name__ == "__main__":
    main()
