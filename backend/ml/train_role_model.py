"""
train_role_model.py
===================

Dieses Skript trainiert das Rollenmodell von Anfang bis Ende.

Für Einsteiger:
- Du startest genau dieses Skript.
- Es baut zuerst die Trainingsdaten neu auf.
- Dann trennt es sauber in Train/Test.
- Danach trainiert es das beste Modell per GridSearch.
- Am Ende speichert es Modell + Metadaten für das Backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    CSV_PATH,
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    ROLE_TRAIN_OUTPUT_COLUMNS,
    TEST_FIXED_PATH,
    TEXT_COLUMN_CANDIDATES,
    TRAIN_LIGHT_OVERSAMPLE,
)
from data_ops import (
    find_column,
    make_clean_training_frame,
    prepare_fixed_test_split,
    read_training_csv,
    rebuild_role_train_from_gold,
    resolve_label_series,
    validate_clean_training_frame,
    validate_role_train_dataframe,
)
from modeling import fit_best_text_model


def _light_oversample_train(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Führt ein leichtes Oversampling auf dem Train-Teil aus.

    Ziel:
    - kleinere Klassen etwas anheben
    - aber nicht aggressiv auf die größte Klasse aufblasen

    Strategie:
    - Zielgröße = Median der Klassenhäufigkeiten
    - nur Klassen unterhalb des Medians werden ergänzt
    """
    counts = train_df["__label__"].value_counts()
    if len(counts) == 0:
        return train_df, {"applied": False, "target_per_class": 0, "added_total": 0, "added_by_class": {}}

    target_per_class = int(counts.median())
    target_per_class = max(1, target_per_class)

    parts: list[pd.DataFrame] = []
    added_by_class: dict[str, int] = {}

    # Jede Klasse separat bearbeiten.
    for label, group in train_df.groupby("__label__", sort=True):
        current_n = len(group)
        if current_n < target_per_class:
            add_n = target_per_class - current_n

            # replace=True:
            # Falls zu wenige Originalzeilen existieren, dürfen Zeilen dupliziert werden.
            extra = group.sample(n=add_n, replace=True, random_state=RANDOM_STATE)
            group = pd.concat([group, extra], ignore_index=True)
            added_by_class[str(label)] = int(add_n)
        parts.append(group)

    # Alles zusammenführen und mischen.
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out, {
        "applied": True,
        "target_per_class": int(target_per_class),
        "added_total": int(sum(added_by_class.values())),
        "added_by_class": added_by_class,
    }


def _norm_set(df: pd.DataFrame, col: str) -> set[str]:
    """
    Liest eine Spalte als normalisierte Menge nicht-leerer Strings.

    Wird für Überlappungsprüfungen genutzt.
    """
    if col not in df.columns:
        return set()
    return set(df[col].fillna("").astype(str).str.strip().tolist()) - {""}


def _assert_disjoint_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    """
    Harte Sicherheitsprüfung gegen Leakage zwischen Train und Test.

    Erzwingt:
    - keine gemeinsamen source_row IDs
    - keine gemeinsamen thread_id Werte
    """
    overlap_source = len(_norm_set(train_df, "source_row") & _norm_set(test_df, "source_row"))
    overlap_sentence = len(_norm_set(train_df, "__text__") & _norm_set(test_df, "__text__"))
    overlap_thread = len(_norm_set(train_df, "thread_id") & _norm_set(test_df, "thread_id"))

    # Gleicher Satztext kann in verschiedenen Threads vorkommen.
    # Darum wird overlap_sentence nur gemeldet, aber nicht als Hard-Fehler gewertet.
    if overlap_source > 0 or overlap_thread > 0:
        raise ValueError(
            "Train/Test leakage detected. "
            f"overlap_source_row={overlap_source}, "
            f"overlap_sentence={overlap_sentence}, "
            f"overlap_thread_id={overlap_thread}"
        )

    return {
        "overlap_source_row": int(overlap_source),
        "overlap_sentence": int(overlap_sentence),
        "overlap_thread_id": int(overlap_thread),
    }


def _persist_disjoint_train_csv(train_df: pd.DataFrame, output_path: str | Path) -> int:
    """
    Speichert den finalen Train-Teil zurück in role_train.csv.

    Warum?
    - So enthält role_train.csv selbst nur echte Train-Zeilen.
    - Das verhindert Verwechslungen bei späteren Durchläufen.
    """
    out = pd.DataFrame(
        {
            "sentence": train_df["__text__"].fillna("").astype(str),
            "gold_label": train_df["__label__"].fillna("").astype(str),
            "weak_label": "",
            "source_row": train_df["source_row"].fillna("").astype(str) if "source_row" in train_df.columns else "",
            "thread_id": train_df["thread_id"].fillna("").astype(str) if "thread_id" in train_df.columns else "",
            "comment_id": "",
            "sentence_index": "",
            "label_notes": "",
        }
    )

    # Feste Spaltenreihenfolge für stabile CSV-Struktur.
    out = out[ROLE_TRAIN_OUTPUT_COLUMNS].copy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return int(len(out))


def main() -> None:
    """
    Startet die komplette Trainingspipeline.

    Schrittfolge:
    1) role_train.csv aus Golddaten neu bauen
    2) CSV laden und Spalten bestimmen
    3) Daten intern bereinigen
    4) festen Testsplit erstellen/verwenden
    5) optionales Oversampling im Train-Teil
    6) GridSearch-Training
    7) Evaluation auf Testset
    8) Modell + Metadaten speichern
    """
    # ---------------------------------------------------------------------
    # Schritt 1: Trainingsdatei aus Goldlabels aufbauen
    # ---------------------------------------------------------------------
    print("role_train.csv wird aus Gold-Dateien neu aufgebaut ...")
    rebuild_role_train_from_gold(CSV_PATH)
    print(f"CSV: {CSV_PATH}")

    # ---------------------------------------------------------------------
    # Schritt 2: CSV laden und Text-/Labelspalten erkennen
    # ---------------------------------------------------------------------
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    df = read_training_csv(csv_path)
    validate_role_train_dataframe(df, context="loaded role_train.csv")

    text_col = find_column(df, TEXT_COLUMN_CANDIDATES)
    if text_col is None:
        raise ValueError(
            "Text-Spalte nicht gefunden.\n"
            f"Gefundene Spalten: {list(df.columns)}\n"
            f"Erwartet eine von: {TEXT_COLUMN_CANDIDATES}"
        )

    # Wichtige Entscheidung:
    # Für echtes Training verwenden wir nur gold_label (kein weak fallback).
    label_series, label_col = resolve_label_series(df, allow_weak_fallback=False)
    print(f"Text-Spalte:  {text_col}")
    print(f"Label-Spalte: {label_col}")

    # ---------------------------------------------------------------------
    # Schritt 3: Internes, bereinigtes Trainingsformat erstellen
    # ---------------------------------------------------------------------
    cleaned_df = make_clean_training_frame(df, text_col, label_series)
    validate_clean_training_frame(cleaned_df)
    if len(cleaned_df) < 20:
        print("Warnung: Datensatz ist sehr klein; Metriken können instabil sein.")

    # ---------------------------------------------------------------------
    # Schritt 4: Fixes Testset nutzen (oder neu erstellen)
    # ---------------------------------------------------------------------
    train_df, test_df, split_info = prepare_fixed_test_split(cleaned_df, TEST_FIXED_PATH)
    print(f"Fixed-Test-Datei: {split_info['path']}")
    print(
        f"Split: train={split_info['train_rows']} | test={split_info['test_rows']} "
        f"(mode={split_info['match_mode']})"
    )

    if split_info.get("test_missing_in_train", 0):
        print(
            f"Warnung: {split_info['test_missing_in_train']} Testzeile(n) "
            "existieren nicht mehr in role_train, bleiben aber zur Vergleichbarkeit im Test."
        )

    # Harte Leakage-Prüfung.
    leakage_stats = _assert_disjoint_train_test(train_df, test_df)
    print(
        "Leakage-Check: "
        f"source_row={leakage_stats['overlap_source_row']}, "
        f"sentence={leakage_stats['overlap_sentence']}, "
        f"thread_id={leakage_stats['overlap_thread_id']}"
    )

    # Train-CSV absichtlich disjunkt speichern.
    persisted_rows = _persist_disjoint_train_csv(train_df, CSV_PATH)
    print(f"role_train.csv disjunkt gespeichert: {persisted_rows} Zeilen")

    # ---------------------------------------------------------------------
    # Schritt 5: Arrays vorbereiten + optional Oversampling
    # ---------------------------------------------------------------------
    oversample_stats = {"applied": False, "target_per_class": 0, "added_total": 0, "added_by_class": {}}
    if TRAIN_LIGHT_OVERSAMPLE:
        train_df, oversample_stats = _light_oversample_train(train_df)
        print(
            "Leichtes Oversampling (nur Train): "
            f"added={oversample_stats['added_total']} "
            f"target_per_class={oversample_stats['target_per_class']}"
        )

    X_train = train_df["__text__"].values
    y_train = train_df["__label__"].values
    X_test = test_df["__text__"].values
    y_test = test_df["__label__"].values

    # ---------------------------------------------------------------------
    # Schritt 6: Bestes Modell via GridSearch ermitteln
    # ---------------------------------------------------------------------
    best_model, best_params_serializable, best_cv_f1 = fit_best_text_model(
        X_train, y_train, stage_name="Single-Stage (all labels)"
    )

    # ---------------------------------------------------------------------
    # Schritt 7: Evaluation auf dem festen Testset
    # ---------------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n====================")
    print("Evaluation (Testset)")
    print("====================")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------------------
    # Schritt 8: Modell + Metadaten schreiben
    # ---------------------------------------------------------------------
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    meta = {
        "csv_path": str(CSV_PATH),
        "fixed_test_path": str(TEST_FIXED_PATH),
        "text_column": text_col,
        "label_column": label_col,
        "classes": sorted(list(set(cleaned_df["__label__"].values))),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy_test": float(acc),
        "best_params": best_params_serializable,
        "best_cv_f1_macro": float(best_cv_f1),
        "leakage_check": leakage_stats,
        "train_oversampling": oversample_stats,
        "notes": "Single-stage role model via GridSearchCV over TF-IDF + (LogReg or LinearSVC).",
    }

    with Path(META_PATH).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print("Gespeichert")
    print("====================")
    print(f"Model: {MODEL_PATH}")
    print(f"Meta:  {META_PATH}")
    print("\nFertig.")


if __name__ == "__main__":
    main()
