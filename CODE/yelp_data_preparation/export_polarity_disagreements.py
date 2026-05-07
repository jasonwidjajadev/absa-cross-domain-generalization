import argparse
import collections
import csv
import json
from typing import Any, Dict, List, Tuple


def split_labels(value: str) -> List[str]:
    # Split semicolon-separated polarity labels.
    if not value:
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


def has_disagreement(pol_a: str, pol_b: str) -> bool:
    # Check whether two polarity strings disagree.
    a = split_labels(pol_a)
    b = split_labels(pol_b)
    if len(a) != len(b):
        return True
    return any(x != y for x, y in zip(a, b))


def strip_code_fences(text: str) -> str:
    # Remove markdown code fences before JSON parsing.
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def parse_polarities_from_response(response_text: str) -> str:
    # Parse "polarities" list from a JSON response payload.
    cleaned = strip_code_fences(response_text)
    if not cleaned:
        return ""
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return ""
    items = payload.get("polarities", [])
    if not isinstance(items, list):
        return ""
    values: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        polarity = str(item.get("polarity", "")).strip()
        if polarity:
            values.append(polarity)
    return ";".join(values)


def load_json_records(path: str) -> List[Dict[str, Any]]:
    # Load and validate a JSON list of records.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list.")
    return data


def build_merged_rows_from_two_sources(
    polarities_a_csv: str, polarities_b_json: str
) -> tuple[List[str], List[Dict[str, str]]]:
    # Align two annotation sources into one merged row list.
    json_records = load_json_records(polarities_b_json)
    by_sentence_id: Dict[int, str] = {}
    by_index: List[str] = []
    for rec in json_records:
        if not isinstance(rec, dict):
            by_index.append("")
            continue
        pol_b = parse_polarities_from_response(str(rec.get("response", "")))
        by_index.append(pol_b)
        sentence_id = rec.get("sentence_id")
        if isinstance(sentence_id, int):
            by_sentence_id[sentence_id] = pol_b

    with open(polarities_a_csv, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV has no header.")
        if "polarities" not in fieldnames:
            raise ValueError('CSV must contain a "polarities" column.')

        out_fieldnames = [
            "polarities_a" if name == "polarities" else name for name in fieldnames
        ]
        if "polarities_b" not in out_fieldnames:
            out_fieldnames.append("polarities_b")

        merged_rows: List[Dict[str, str]] = []
        for row_idx, row in enumerate(reader, start=1):
            old_pol = row.get("polarities", "")
            row_out = dict(row)
            row_out.pop("polarities", None)
            row_out["polarities_a"] = old_pol
            row_out["polarities_b"] = by_sentence_id.get(
                row_idx, by_index[row_idx - 1] if row_idx - 1 < len(by_index) else ""
            )
            merged_rows.append(row_out)
    return out_fieldnames, merged_rows


def build_aligned_pairs(pol_a: str, pol_b: str) -> Tuple[List[str], List[str], bool]:
    # Align two label lists and report length mismatch.
    a = split_labels(pol_a)
    b = split_labels(pol_b)
    mismatch = len(a) != len(b)
    n = min(len(a), len(b))
    return a[:n], b[:n], mismatch


def print_kappa_analysis(all_rows: List[Dict[str, str]]) -> None:
    # Print agreement stats and confusion matrix over aligned labels.
    # This gives a quick quality check before disagreement export.
    flat_a: List[str] = []
    flat_b: List[str] = []
    length_mismatch_rows = 0

    for row in all_rows:
        a_part, b_part, mismatch = build_aligned_pairs(
            row.get("polarities_a", ""), row.get("polarities_b", "")
        )
        if mismatch:
            length_mismatch_rows += 1
        flat_a.extend(a_part)
        flat_b.extend(b_part)

    total_pairs = len(flat_a)
    if total_pairs == 0:
        print("Kappa analysis: no aligned polarity pairs found.")
        return

    matches = sum(1 for x, y in zip(flat_a, flat_b) if x == y)
    observed = matches / total_pairs

    count_a = collections.Counter(flat_a)
    count_b = collections.Counter(flat_b)
    labels = sorted(set(count_a) | set(count_b))
    expected = sum((count_a[l] / total_pairs) * (count_b[l] / total_pairs) for l in labels)
    kappa = (observed - expected) / (1 - expected) if (1 - expected) != 0 else float("nan")

    confusion: Dict[str, Dict[str, int]] = {
        a_label: {b_label: 0 for b_label in labels} for a_label in labels
    }
    for a_val, b_val in zip(flat_a, flat_b):
        confusion[a_val][b_val] += 1

    print("\nKappa analysis (aligned item-level):")
    print(f"- aligned_pairs: {total_pairs}")
    print(f"- rows_with_length_mismatch: {length_mismatch_rows}")
    print(f"- observed_agreement: {observed:.6f}")
    print(f"- expected_agreement: {expected:.6f}")
    print(f"- cohens_kappa: {kappa:.6f}")
    print(f"- label_counts_a: {dict(sorted(count_a.items()))}")
    print(f"- label_counts_b: {dict(sorted(count_b.items()))}")
    print("- confusion_matrix (rows=polarities_a, cols=polarities_b):")
    for a_label in labels:
        row_counts = " ".join(f"{b_label}:{confusion[a_label][b_label]}" for b_label in labels)
        print(f"  {a_label} -> {row_counts}")


def main() -> None:
    # Build merged labels, print kappa stats, and export disagreements.
    parser = argparse.ArgumentParser(
        description=(
            "Build polarities_a/polarities_b from two sources, then export disagreement rows."
        )
    )
    parser.add_argument(
        "--polarities_a_csv",
        required=True,
        help='Source CSV containing "polarities" (will become polarities_a).',
    )
    parser.add_argument(
        "--polarities_b_json",
        required=True,
        help='JSON output from annotate_polarity_from_aspects_gpt.py (used to build polarities_b).',
    )
    parser.add_argument(
        "--output_csv",
        default="cleaned_disagreements.csv",
        help="Output CSV path for disagreement rows.",
    )
    args = parser.parse_args()

    fieldnames, all_rows = build_merged_rows_from_two_sources(
        args.polarities_a_csv, args.polarities_b_json
    )
    rows: List[Dict[str, str]] = []
    total = len(all_rows)
    for row in all_rows:
        pol_a = row.get("polarities_a", "")
        pol_b = row.get("polarities_b", "")
        if has_disagreement(pol_a, pol_b):
            rows.append(row)

    print_kappa_analysis(all_rows)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Done. Found {len(rows)} disagreement rows out of {total}. "
        f"Wrote {args.output_csv}"
    )


if __name__ == "__main__":
    main()
