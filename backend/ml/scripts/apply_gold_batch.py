"""
apply_gold_batch.py

Uebernimmt Labels aus einer manuell geprueften Batch-CSV und schreibt sie
ueber `source_row` in die Haupt-Gold-CSV zurueck.
"""

import argparse
import os
import pandas as pd


DEFAULT_MAIN = "backend/ml/data/processed/cmv_gold_batch_4000.csv"
DEFAULT_BATCH = "backend/ml/data/processed/cmv_gold_label_batch.csv"


def clean(series: pd.Series) -> pd.Series:
    """Normalisiert eine Textspalte (NaN entfernen, Whitespace trimmen)."""
    return series.fillna("").astype(str).str.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply manually labeled gold labels from a batch CSV back to the main CSV."
    )
    parser.add_argument("--main", default=DEFAULT_MAIN, help="Main dataset CSV path")
    parser.add_argument("--batch", default=DEFAULT_BATCH, help="Labeled batch CSV path")
    parser.add_argument(
        "--output",
        default="",
        help="Output path (default: overwrite --main)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI-Einstiegspunkt."""
    args = parse_args()
    output = args.output or args.main

    if not os.path.exists(args.main):
        raise FileNotFoundError(f"Main file not found: {args.main}")
    if not os.path.exists(args.batch):
        raise FileNotFoundError(f"Batch file not found: {args.batch}")

    main_df = pd.read_csv(args.main)
    batch_df = pd.read_csv(args.batch)

    required_batch = ["source_row", "gold_label"]
    missing = [c for c in required_batch if c not in batch_df.columns]
    if missing:
        raise ValueError(f"Batch missing required columns: {missing}")
    if "gold_label" not in main_df.columns:
        raise ValueError("Main CSV must contain column: gold_label")
    if "label_notes" not in main_df.columns:
        main_df["label_notes"] = ""

    # Bereinigte Label-Batch bauen, damit das Merge stabil ist.
    labeled = batch_df.copy()
    labeled["source_row"] = pd.to_numeric(labeled["source_row"], errors="coerce")
    labeled = labeled.dropna(subset=["source_row"])
    labeled["source_row"] = labeled["source_row"].astype(int)
    labeled["gold_label"] = clean(labeled["gold_label"])
    if "label_notes" in labeled.columns:
        labeled["label_notes"] = clean(labeled["label_notes"])
    else:
        labeled["label_notes"] = ""

    labeled = labeled[labeled["gold_label"] != ""].copy()
    labeled = labeled.drop_duplicates(subset=["source_row"], keep="last")

    max_idx = len(main_df) - 1
    labeled = labeled[(labeled["source_row"] >= 0) & (labeled["source_row"] <= max_idx)]

    # Labels zeilenweise anwenden, `source_row` ist der stabile Schluessel.
    updated = 0
    for _, row in labeled.iterrows():
        idx = int(row["source_row"])
        main_df.at[idx, "gold_label"] = row["gold_label"]
        if row["label_notes"] != "":
            main_df.at[idx, "label_notes"] = row["label_notes"]
        updated += 1

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    main_df.to_csv(output, index=False, encoding="utf-8")

    print(f"[ok] Main rows: {len(main_df)}")
    print(f"[ok] Batch rows with gold_label: {len(labeled)}")
    print(f"[ok] Updated rows: {updated}")
    print(f"[ok] Output: {output}")


if __name__ == "__main__":
    main()
