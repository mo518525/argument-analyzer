"""
prepare_balance_batch.py

Erstellt eine Zusatz-Batch, um Klassenhaeufigkeiten in manuell gelabelten Daten auszubalancieren.
"""

import argparse
import os
import pandas as pd


DEFAULT_GOLD = "backend/ml/data/processed/cmv_gold_batch_4000.csv"
DEFAULT_FULL = "backend/ml/data/processed/cmv_gold_batch_4000.csv"
DEFAULT_OUTPUT = "backend/ml/data/processed/cmv_gold_balance_batch.csv"
LABELS = ["claim", "premise", "objection", "conclusion", "other"]


def clean(series: pd.Series) -> pd.Series:
    """Normalisiert textaehnliche Werte."""
    return series.fillna("").astype(str).str.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt eine Klassen-Balance-Label-Batch aus einem CMV-Pool."
    )
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="Current gold batch CSV")
    parser.add_argument("--full", default=DEFAULT_FULL, help="Full pool CSV (gold or weak)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output batch CSV")
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=250,
        help="Target labeled rows per class in gold set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """CLI-Einstiegspunkt."""
    args = parse_args()

    if not os.path.exists(args.gold):
        raise FileNotFoundError(f"Gold file not found: {args.gold}")
    if not os.path.exists(args.full):
        raise FileNotFoundError(f"Full file not found: {args.full}")

    gold_df = pd.read_csv(args.gold)
    if "source_row" not in gold_df.columns:
        raise ValueError("Gold CSV must contain source_row")
    if "gold_label" not in gold_df.columns:
        raise ValueError("Gold CSV must contain gold_label")

    gold_df["source_row"] = pd.to_numeric(gold_df["source_row"], errors="coerce")
    gold_df = gold_df.dropna(subset=["source_row"]).copy()
    gold_df["source_row"] = gold_df["source_row"].astype(int)
    gold_df["gold_label"] = clean(gold_df["gold_label"])
    labeled_gold = gold_df[gold_df["gold_label"] != ""].copy()

    full_df = pd.read_csv(args.full)
    if "sentence" not in full_df.columns:
        raise ValueError("Full CSV must contain sentence")

    full_df = full_df.copy()
    if "source_row" not in full_df.columns:
        full_df["source_row"] = full_df.index
    if "weak_label" not in full_df.columns:
        full_df["weak_label"] = ""
    full_df["weak_label"] = clean(full_df["weak_label"])
    has_weak_suggestions = (full_df["weak_label"] != "").any()
    if "gold_label" in full_df.columns:
        full_df["gold_label"] = clean(full_df["gold_label"])

    used_source_rows = set(gold_df["source_row"].tolist())
    # `chunks` sammelt die gesampelten Teilmengen pro benoetigter Klasse.
    chunks: list[pd.DataFrame] = []

    print("[info] Current labeled distribution:")
    current_counts = labeled_gold["gold_label"].value_counts().to_dict()
    for label in LABELS:
        print(f"  {label}: {current_counts.get(label, 0)}")

    print(f"[info] Target per class: {args.target_per_class}")
    print("[info] Erzeuge Top-up-Batch aus bisher ungenutzten Zeilen...")

    if has_weak_suggestions:
        # Bevorzugter Weg: pro Klasse ueber weak-label Vorschlaege samplen.
        for label in LABELS:
            current = current_counts.get(label, 0)
            need = max(0, args.target_per_class - current)
            if need == 0:
                continue

            pool = full_df[
                (full_df["weak_label"] == label)
                & (~full_df["source_row"].isin(used_source_rows))
            ].copy()

            if len(pool) == 0:
                print(f"  {label}: need {need}, selected 0 (no pool)")
                continue

            n = min(need, len(pool))
            sample = pool.sample(n=n, random_state=args.seed).copy()
            sample["gold_label"] = ""
            sample["label_notes"] = ""
            chunks.append(sample)
            used_source_rows.update(sample["source_row"].tolist())
            print(f"  {label}: need {need}, selected {n}")
    else:
        # Gold-only fallback: no weak suggestions available.
        # We still create a top-up batch by sampling currently unlabeled rows.
        total_need = sum(max(0, args.target_per_class - current_counts.get(label, 0)) for label in LABELS)
        pool = full_df[(~full_df["source_row"].isin(used_source_rows))].copy()
        if "gold_label" in pool.columns:
            pool = pool[pool["gold_label"] == ""].copy()

        if len(pool) == 0 or total_need <= 0:
            print("[info] No weak suggestions found and no unlabeled pool available for top-up.")
        else:
            n = min(total_need, len(pool))
            sample = pool.sample(n=n, random_state=args.seed).copy()
            sample["gold_label"] = ""
            sample["label_notes"] = ""
            sample["weak_label"] = ""
            chunks.append(sample)
            print(f"[info] weak_label unavailable -> sampled {n} unlabeled rows (gold-only fallback).")

    if not chunks:
        print("[ok] No additional rows needed. Already at/above target for all classes.")
        return

    out_df = pd.concat(chunks, ignore_index=True)

    out_cols = [
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
    for col in out_cols:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[out_cols]
    out_df = out_df.sort_values(by=["weak_label", "source_row"]).reset_index(drop=True)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8")

    print(f"[ok] Output: {args.output}")
    print(f"[ok] Rows in batch: {len(out_df)}")
    print("[ok] Batch weak-label distribution:")
    print(out_df["weak_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
