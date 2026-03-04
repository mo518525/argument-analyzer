"""
Label lint checks for CMV role data.

Checks:
1) Conflicts:
   - same source_row with multiple labels
   - same sentence (normalized) with multiple labels
2) Likely "other" leaks:
   - rows labeled other that look argumentative
3) Risky claim/premise swaps:
   - claim rows with premise-like markers
   - premise rows with claim-like markers

Usage:
  python backend/ml/scripts/label_lint.py
  python backend/ml/scripts/label_lint.py --input backend/ml/data/processed/cmv_gold_batch_4000.csv
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = ML_DIR / "data" / "processed" / "cmv_gold_batch_4000.csv"
DEFAULT_OUTPUT_DIR = ML_DIR / "data" / "reports" / "lint"


ARG_MARKERS = [
    "because",
    "therefore",
    "thus",
    "hence",
    "for example",
    "for instance",
    "this means",
    "which means",
    "implies",
    "evidence",
    "study",
    "data",
    "if ",
    "then ",
    "however",
    "but ",
]

META_HINTS = [
    "automod",
    "moderator",
    "submission",
    "delta",
    "rule",
    "removed",
    "thanks for",
    "edit:",
    "cmv:",
    "title:",
]

PREMISE_LIKE_MARKERS = [
    "because",
    "for example",
    "for instance",
    "evidence",
    "study",
    "data",
    "due to",
    "as shown",
    "which means",
    "therefore",
]

CLAIM_LIKE_MARKERS = [
    "i think",
    "i believe",
    "we should",
    "should ",
    "must ",
    "need to",
    "ought",
    "is wrong",
    "is right",
    "should be",
]


@dataclass
class LintResult:
    source_row_conflicts: pd.DataFrame
    sentence_conflicts: pd.DataFrame
    other_leaks: pd.DataFrame
    claim_premise_risks: pd.DataFrame
    unlabeled: pd.DataFrame


def _norm_text(text: str) -> str:
    """Kleinbuchstaben + Leerzeichen normalisieren."""
    x = (text or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _has_any(text: str, markers: list[str]) -> bool:
    """Gibt True zurueck, wenn mindestens ein Marker im Text vorkommt."""
    t = (text or "").lower()
    return any(m in t for m in markers)


def _is_meta_like(text: str) -> bool:
    """Heuristik fuer nicht-argumentativen bzw. Meta-Inhalt."""
    t = (text or "").strip().lower()
    if not t:
        return True
    if len(t) < 12:
        return True
    return _has_any(t, META_HINTS)


def _safe_col(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    """Liefert Spalte wenn vorhanden, sonst eine sichere Default-Serie."""
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def run_lint(df: pd.DataFrame) -> LintResult:
    """
    Fuehrt alle Lint-Checks aus und gibt gruppierte Ergebnis-Tabellen zurueck.

    Die Tabellen koennen als getrennte CSVs fuer das Review exportiert werden.
    """
    work = df.copy()
    work["sentence"] = _safe_col(work, "sentence").astype(str).str.strip()
    work["gold_label"] = _safe_col(work, "gold_label").astype(str).str.strip().str.lower()
    work["source_row"] = _safe_col(work, "source_row").astype(str).str.strip()
    work["thread_id"] = _safe_col(work, "thread_id").astype(str).str.strip()
    work["sentence_norm"] = work["sentence"].map(_norm_text)

    unlabeled = work[work["gold_label"] == ""].copy()
    labeled = work[work["gold_label"] != ""].copy()

    # 1) Konflikte: gleiches Item/Satz mit mehreren Gold-Labels.
    sr_counts = labeled.groupby("source_row")["gold_label"].nunique()
    bad_sr = sr_counts[sr_counts > 1].index
    source_row_conflicts = (
        labeled[labeled["source_row"].isin(bad_sr)]
        .sort_values(["source_row", "gold_label"])
        .drop(columns=["sentence_norm"], errors="ignore")
    )

    sent_counts = labeled.groupby("sentence_norm")["gold_label"].nunique()
    bad_sent = sent_counts[sent_counts > 1].index
    sentence_conflicts = (
        labeled[labeled["sentence_norm"].isin(bad_sent)]
        .sort_values(["sentence_norm", "gold_label"])
        .drop(columns=["sentence_norm"], errors="ignore")
    )

    # 2) Moegliche "other"-Lecks: wirkt argumentativ, ist aber als other gelabelt.
    other = labeled[labeled["gold_label"] == "other"].copy()
    other["looks_argumentative"] = other["sentence"].map(
        lambda s: (not _is_meta_like(s)) and _has_any(s.lower(), ARG_MARKERS)
    )
    other_leaks = other[other["looks_argumentative"]].drop(columns=["looks_argumentative"])

    # 3) Riskante claim/premise Vertauschungen ueber Marker-Mismatch.
    cp = labeled[labeled["gold_label"].isin(["claim", "premise"])].copy()
    cp["has_premise_markers"] = cp["sentence"].map(
        lambda s: _has_any(s.lower(), PREMISE_LIKE_MARKERS)
    )
    cp["has_claim_markers"] = cp["sentence"].map(
        lambda s: _has_any(s.lower(), CLAIM_LIKE_MARKERS)
    )
    risky_claim = cp[(cp["gold_label"] == "claim") & (cp["has_premise_markers"])]
    risky_premise = cp[(cp["gold_label"] == "premise") & (cp["has_claim_markers"])]
    claim_premise_risks = pd.concat([risky_claim, risky_premise], ignore_index=True)
    claim_premise_risks = claim_premise_risks.drop(
        columns=["has_premise_markers", "has_claim_markers"]
    )

    return LintResult(
        source_row_conflicts=source_row_conflicts,
        sentence_conflicts=sentence_conflicts,
        other_leaks=other_leaks,
        claim_premise_risks=claim_premise_risks,
        unlabeled=unlabeled,
    )


def write_reports(result: LintResult, output_dir: Path) -> None:
    """Schreibt jede Lint-Tabelle als CSV-Datei."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.source_row_conflicts.to_csv(output_dir / "source_row_conflicts.csv", index=False)
    result.sentence_conflicts.to_csv(output_dir / "sentence_conflicts.csv", index=False)
    result.other_leaks.to_csv(output_dir / "other_leaks.csv", index=False)
    result.claim_premise_risks.to_csv(output_dir / "claim_premise_risks.csv", index=False)
    result.unlabeled.to_csv(output_dir / "unlabeled_rows.csv", index=False)


def print_summary(result: LintResult, total_rows: int, labeled_rows: int) -> None:
    """Gibt eine kompakte Terminal-Zusammenfassung der Lint-Funde aus."""
    print("=== Label Lint Summary ===")
    print(f"total_rows: {total_rows}")
    print(f"labeled_rows: {labeled_rows}")
    print(f"unlabeled_rows: {len(result.unlabeled)}")
    print(f"source_row_conflict_rows: {len(result.source_row_conflicts)}")
    print(f"sentence_conflict_rows: {len(result.sentence_conflicts)}")
    print(f"other_leak_rows: {len(result.other_leaks)}")
    print(f"claim_premise_risk_rows: {len(result.claim_premise_risks)}")


def main() -> None:
    """CLI-Einstiegspunkt."""
    parser = argparse.ArgumentParser(description="Lint CMV gold labels for consistency and risk.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to labeled CSV (default: cmv_gold_batch_4000.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where lint report CSVs are written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    total_rows = len(df)
    labeled_rows = int((df.get("gold_label", "") != "").sum()) if "gold_label" in df.columns else 0

    result = run_lint(df)
    output_dir = Path(args.output_dir)
    write_reports(result, output_dir)
    print_summary(result, total_rows, labeled_rows)
    print(f"reports_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
