"""
prepare_gold_batch.py

Erstellt eine handhabbare Label-Batch aus einer groesseren Gold-Datei.

Typischer Ablauf:
1) ungelabelte Zeilen auswaehlen
2) per Groesse/Seed samplen
3) reviewer-freundliche Spalten exportieren
"""

import argparse
import os
import pandas as pd


DEFAULT_INPUT = "backend/ml/data/processed/cmv_gold_batch_4000.csv"
DEFAULT_OUTPUT = "backend/ml/data/processed/cmv_gold_label_batch.csv"


def clean_str_series(series: pd.Series) -> pd.Series:
    """Normalisiert eine textaehnliche Spalte."""
    return series.fillna("").astype(str).str.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a manual gold-label batch from the current gold main file."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--size", type=int, default=1000, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--unique-sentence",
        action="store_true",
        help="Drop duplicate sentences before sampling",
    )
    parser.add_argument(
        "--prefill-from-weak",
        action="store_true",
        help="Prefill gold_label with weak_label if available (you still review manually)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI-Einstiegspunkt."""
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    required = ["sentence", "gold_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    # source_row behalten, damit spaeter sicher zurueckgemerged werden kann.
    if "source_row" not in df.columns:
        df["source_row"] = df.index
    if "weak_label" not in df.columns:
        df["weak_label"] = ""

    gold = clean_str_series(df["gold_label"])
    unlabeled = df[gold == ""].copy()

    if args.unique_sentence:
        unlabeled = unlabeled.drop_duplicates(subset=["sentence"], keep="first")

    if len(unlabeled) == 0:
        raise ValueError("No unlabeled rows found. gold_label is already filled everywhere.")

    n = min(args.size, len(unlabeled))
    batch = unlabeled.sample(n=n, random_state=args.seed).copy()

    # Optional: fuer schnelleres Labeln weak_label als Startwert vorbefuellen.
    if args.prefill_from_weak:
        weak = clean_str_series(batch["weak_label"])
        if (weak != "").any():
            batch["gold_label"] = weak
        else:
            print("[warn] --prefill-from-weak was set, but weak_label is empty/missing.")

    output_columns = [
        "source_row",
        "thread_id",
        "thread_title",
        "thread_permalink",
        "source_type",
        "comment_id",
        "parent_id",
        "level",
        "author",
        "sentence_index",
        "sentence",
        "weak_label",
        "gold_label",
        "label_notes",
    ]
    for col in output_columns:
        if col not in batch.columns:
            batch[col] = ""
    batch = batch[output_columns]
    if (clean_str_series(batch["weak_label"]) != "").any():
        batch = batch.sort_values(by=["weak_label", "thread_id", "comment_id", "sentence_index"])
    else:
        batch = batch.sort_values(by=["source_row"])

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    batch.to_csv(args.output, index=False, encoding="utf-8")

    print(f"[ok] Input rows: {len(df)}")
    print(f"[ok] Unlabeled pool: {len(unlabeled)}")
    print(f"[ok] Batch size: {len(batch)}")
    print(f"[ok] Output: {args.output}")
    weak_non_empty = clean_str_series(batch["weak_label"])
    if (weak_non_empty != "").any():
        print("[ok] Batch weak-label distribution:")
        print(weak_non_empty.value_counts().to_string())
    else:
        print("[ok] No weak_label suggestions in this batch (gold-only source).")


if __name__ == "__main__":
    main()
