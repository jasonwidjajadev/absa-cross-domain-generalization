import argparse
import collections
import csv


def split_labels(value: str) -> list[str]:
    # Split semicolon-separated labels from one row.
    if value is None:
        return []
    text = value.strip()
    if text == "":
        return []
    return [x.strip() for x in text.split(";") if x.strip() != ""]


def load_polarities(path: str) -> list[str]:
    # Read the polarities column from a CSV file.
    values: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "polarities" not in reader.fieldnames:
            raise ValueError(f'{path} must contain a "polarities" column.')
        for row in reader:
            values.append(row["polarities"])
    return values


def build_aligned_pairs(a_rows: list[str], b_rows: list[str]) -> tuple[list[str], list[str], int]:
    # Align label pairs row by row and flatten for scoring.
    if len(a_rows) != len(b_rows):
        raise ValueError(
            f"Row count mismatch: file A has {len(a_rows)} rows, file B has {len(b_rows)} rows."
        )

    flat_a: list[str] = []
    flat_b: list[str] = []
    length_mismatch_rows = 0

    for idx in range(len(a_rows)):
        a_labels = split_labels(a_rows[idx])
        b_labels = split_labels(b_rows[idx])

        if len(a_labels) != len(b_labels):
            length_mismatch_rows += 1

        n = min(len(a_labels), len(b_labels))
        for j in range(n):
            flat_a.append(a_labels[j])
            flat_b.append(b_labels[j])

    return flat_a, flat_b, length_mismatch_rows


def compute_kappa(flat_a: list[str], flat_b: list[str]) -> tuple[float, float, float]:
    # Compute observed agreement, expected agreement, and Cohen's kappa.
    total = len(flat_a)
    if total == 0:
        raise ValueError("No aligned label pairs found.")

    matches = 0
    for i in range(total):
        if flat_a[i] == flat_b[i]:
            matches += 1
    observed = matches / total

    count_a = collections.Counter(flat_a)
    count_b = collections.Counter(flat_b)
    labels = sorted(set(count_a.keys()) | set(count_b.keys()))

    expected = 0.0
    for label in labels:
        expected += (count_a[label] / total) * (count_b[label] / total)

    if expected == 1.0:
        kappa = float("nan")
    else:
        kappa = (observed - expected) / (1.0 - expected)

    return observed, expected, kappa


def parse_args() -> argparse.Namespace:
    # Parse two annotator CSV paths from CLI.
    parser = argparse.ArgumentParser(description="Calculate Cohen's kappa from two CSV label files.")
    parser.add_argument("--file-a", required=True, help='CSV path for annotator A (must have "polarities").')
    parser.add_argument("--file-b", required=True, help='CSV path for annotator B (must have "polarities").')
    return parser.parse_args()


def main() -> None:
    # Run full kappa computation and print summary stats.
    args = parse_args()

    a_rows = load_polarities(args.file_a)
    b_rows = load_polarities(args.file_b)

    flat_a, flat_b, mismatch_rows = build_aligned_pairs(a_rows, b_rows)
    observed, expected, kappa = compute_kappa(flat_a, flat_b)

    print(f"aligned_pairs: {len(flat_a)}")
    print(f"rows_with_length_mismatch: {mismatch_rows}")
    print(f"observed_agreement: {observed:.6f}")
    print(f"expected_agreement: {expected:.6f}")
    print(f"cohens_kappa: {kappa:.6f}")


if __name__ == "__main__":
    main()
