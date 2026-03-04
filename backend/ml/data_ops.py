"""
data_ops.py

Zentraler Ort für datenbezogene ML-Logik:
- Schema-Validierung
- CSV-Lesen / Vorbereitung
- Gold -> Train-Neuaufbau
- fester Test-Split

So bleibt das Projekt strukturierter und die Dateianzahl kleiner.

Einsteiger-Orientierung:
- Diese Datei ist der "Datenmotor" des Trainings.
- `train_role_model.py` ruft die Hilfsfunktionen in Reihenfolge auf.
- Wenn Train/Test-Zeilen merkwürdig wirken, zuerst hier prüfen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ALLOWED_LABELS,
    GOLD_PATHS,
    GOLD_REQUIRED_COLUMNS,
    LABEL_MAP_TO_CORE,
    RANDOM_STATE,
    ROLE_TEST_FIXED_REQUIRED_COLUMNS,
    ROLE_TRAIN_OUTPUT_COLUMNS,
    ROLE_TRAIN_REQUIRED_COLUMNS,
    TEST_FIXED_ROWS,
    TEST_SIZE_FALLBACK,
    WEAK_LABEL_COLUMN_CANDIDATES,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Schema-Validierung
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def assert_required_columns(df: pd.DataFrame, required_cols: list[str], context: str) -> None:
    """
    Bricht früh ab, wenn erwartete Spalten fehlen.

    Frühes Abbrechen liefert klarere Fehler als spätere Abstürze.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _assert_nonempty_string_column(df: pd.DataFrame, col: str, context: str) -> None:
    vals = df[col].fillna("").astype(str).str.strip()
    empty = int((vals == "").sum())
    if empty > 0:
        raise ValueError(
            f"{context}: column '{col}' contains {empty} empty value(s). "
            "Please fix data before training."
        )


def _assert_numeric_column(df: pd.DataFrame, col: str, context: str) -> None:
    vals = pd.to_numeric(df[col], errors="coerce")
    bad = int(vals.isna().sum())
    if bad > 0:
        raise ValueError(
            f"{context}: column '{col}' contains {bad} non-numeric value(s). "
            "Expected numeric IDs."
        )


def validate_gold_dataframe(df: pd.DataFrame, source_name: str) -> None:
    """
    Validiert die Struktur einer rohen Gold-Datei.

    Wichtig:
    - Leere `gold_label` ist erlaubt (noch nicht fertig gelabelt).
    - Nicht-leere Labels müssen Kernlabels oder bekannte Mapping-Labels sein.
    """
    context = f"gold file '{source_name}'"
    assert_required_columns(df, GOLD_REQUIRED_COLUMNS, context)

    _assert_nonempty_string_column(df, "sentence", context)
    _assert_nonempty_string_column(df, "thread_id", context)
    _assert_numeric_column(df, "source_row", context)

    raw = df["gold_label"].fillna("").astype(str).str.strip().str.lower()
    non_empty = raw[raw != ""]
    allowed = set(ALLOWED_LABELS) | set(LABEL_MAP_TO_CORE.keys())
    unknown = sorted(set(non_empty.unique()) - allowed)
    if unknown:
        preview = unknown[:10]
        raise ValueError(
            f"{context}: found unsupported label(s): {preview}. "
            f"Allowed core labels: {sorted(ALLOWED_LABELS)} + mapped labels: {sorted(LABEL_MAP_TO_CORE.keys())}"
        )


def validate_role_train_dataframe(df: pd.DataFrame, context: str = "role_train.csv") -> None:
    """
    Validiert die neu aufgebaute role_train.csv für das Modelltraining.

    Diese Prüfung ist strenger als bei Roh-Gold, weil Training saubere Daten braucht.
    """
    assert_required_columns(df, ROLE_TRAIN_REQUIRED_COLUMNS, context)
    _assert_nonempty_string_column(df, "sentence", context)
    _assert_nonempty_string_column(df, "gold_label", context)
    _assert_nonempty_string_column(df, "thread_id", context)
    _assert_numeric_column(df, "source_row", context)

    labels = df["gold_label"].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(labels.unique()) - set(ALLOWED_LABELS))
    if unknown:
        preview = unknown[:10]
        raise ValueError(
            f"{context}: contains unsupported core label(s): {preview}. "
            f"Expected only: {sorted(ALLOWED_LABELS)}"
        )


def validate_clean_training_frame(df: pd.DataFrame, context: str = "cleaned training frame") -> None:
    """Validiert das interne bereinigte DataFrame direkt vor dem Split."""
    assert_required_columns(df, ["__text__", "__label__", "source_row", "thread_id"], context)
    _assert_nonempty_string_column(df, "__text__", context)
    _assert_nonempty_string_column(df, "__label__", context)
    _assert_nonempty_string_column(df, "thread_id", context)
    _assert_numeric_column(df, "source_row", context)


def validate_role_test_fixed_dataframe(df: pd.DataFrame, context: str = "role_test_fixed.csv") -> None:
    """Validiert das Schema des festen Testsets (falls vorhanden)."""
    assert_required_columns(df, ROLE_TEST_FIXED_REQUIRED_COLUMNS, context)
    _assert_nonempty_string_column(df, "sentence", context)
    _assert_nonempty_string_column(df, "gold_label", context)
    _assert_nonempty_string_column(df, "thread_id", context)
    _assert_numeric_column(df, "source_row", context)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Daten-Pipeline / Vorbereitung
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Findet einen passenden Spaltennamen ohne Groß-/Kleinschreibung zu beachten.

    Beispiel:
    - Wenn Datei `Sentence` hat und wir `sentence` suchen, funktioniert es trotzdem.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def resolve_label_series(
    df: pd.DataFrame,
    allow_weak_fallback: bool = False,
) -> tuple[pd.Series, str]:
    """
    Bestimmt, welche Label-Spalte für das Training verwendet wird.

    Standardverhalten ist strikt (empfohlen):
    - use only gold_label (manual labels)
    - do NOT silently fall back to weak labels

    Optionaler Kompatibilitätsmodus (nur für Legacy-Fälle):
    - set allow_weak_fallback=True to use older weak/legacy columns
    """
    gold_col = find_column(df, ["gold_label"])
    weak_col = find_column(df, WEAK_LABEL_COLUMN_CANDIDATES)

    if gold_col:
        gold = df[gold_col].astype(str).str.strip()
        if allow_weak_fallback and weak_col:
            weak = df[weak_col].astype(str).str.strip()
            merged = gold.where(gold != "", weak)
            return merged, f"{gold_col} (fallback {weak_col})"
        return gold, gold_col

    if allow_weak_fallback and weak_col:
        return df[weak_col].astype(str).str.strip(), weak_col

    raise ValueError(
        "No gold_label column found. Training is configured for gold-only labels."
    )


def read_training_csv(path: str | Path) -> pd.DataFrame:
    """
    Robuster CSV-Reader mit Fallback-Parser.

    Warum es einen Fallback gibt:
    - Manche Satztexte enthalten Kommas.
    - Bei kaputtem Quoting kann normales CSV-Parsing scheitern.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("Less than 2 columns found.")
        return df
    except Exception:
        # Fallback-Parser: jede Zeile am LETZTEN Komma teilen.
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().lstrip("\ufeff")
            if not header:
                raise ValueError("CSV header is missing or file is empty.")

            for line_no, line in enumerate(f, start=2):
                raw = line.strip()
                if not raw:
                    continue
                if "," not in raw:
                    raise ValueError(f"Invalid line without comma at line {line_no}: {raw}")

                text_value, label_value = raw.rsplit(",", 1)
                rows.append({"text": text_value.strip(), "label": label_value.strip()})

        return pd.DataFrame(rows, columns=["text", "label"])


def map_gold_label_to_core(label: str) -> str:
    """
    Mappt optionale feinere Labels auf die 5 Kernklassen.

    Design-Entscheidung:
    - Unbekannte nicht-leere Labels werden zu `other`.
    - Linting/Validierung soll solche Fälle trotzdem separat sichtbar machen.
    """
    lab = (label or "").strip().lower()
    if not lab:
        return ""
    if lab in ALLOWED_LABELS:
        return lab
    return LABEL_MAP_TO_CORE.get(lab, "other")


def rebuild_role_train_from_gold(
    output_path: str | Path,
    gold_paths: Iterable[str | Path] = GOLD_PATHS,
) -> tuple[int, dict[str, int]]:
    """
    Baut role_train.csv aus verfügbaren Gold-Dateien neu.

    Diese Funktion wird am Trainingsanfang aufgerufen, damit role_train.csv
    immer den neuesten manuellen Label-Stand hat.
    """
    frames: list[pd.DataFrame] = []
    used_sources: list[str] = []
    mapped_counts: dict[str, int] = {}

    for gold_path in gold_paths:
        gold_path = Path(gold_path)
        if not gold_path.exists():
            continue

        df = pd.read_csv(gold_path)
        validate_gold_dataframe(df, str(gold_path))

        # Optionale Spalten beibehalten, damit ältere Tools/Skripte weiterlaufen.
        if "label_notes" not in df.columns:
            df["label_notes"] = ""
        if "weak_label" not in df.columns:
            df["weak_label"] = ""

        df["sentence"] = df["sentence"].fillna("").astype(str).str.strip()
        df["gold_label_raw"] = df["gold_label"].fillna("").astype(str).str.strip().str.lower()
        df["gold_label"] = df["gold_label_raw"].map(map_gold_label_to_core)

        # Mitzählen, wie viele feine Labels auf Kernlabels gemappt wurden.
        mapped_mask = (df["gold_label_raw"] != "") & (df["gold_label_raw"] != df["gold_label"])
        if mapped_mask.any():
            for raw_lab, count in df.loc[mapped_mask, "gold_label_raw"].value_counts().items():
                mapped_counts[raw_lab] = mapped_counts.get(raw_lab, 0) + int(count)

        # Nur Zeilen mit nicht-leerem Satz und gültigem Kernlabel behalten.
        df = df[(df["sentence"] != "") & (df["gold_label"].isin(ALLOWED_LABELS))].copy()
        if len(df) == 0:
            continue

        df = df.drop(columns=["gold_label_raw"], errors="ignore")
        frames.append(df)
        used_sources.append(str(gold_path))

    if not frames:
        raise FileNotFoundError(
            "No labeled gold data found. "
            f"Expected at least one of: {[str(p) for p in gold_paths]}"
        )

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Pro `source_row` bevorzugt genau eine Zeile, um Duplikate zu reduzieren.
    if "source_row" in merged.columns:
        merged["__source_row_num__"] = pd.to_numeric(merged["source_row"], errors="coerce")
        with_sr = merged[merged["__source_row_num__"].notna()].copy()
        without_sr = merged[merged["__source_row_num__"].isna()].copy()
        with_sr = with_sr.sort_values(by=["__source_row_num__"]).drop_duplicates(
            subset=["__source_row_num__"],
            keep="first",
        )
        merged = pd.concat([with_sr, without_sr], ignore_index=True, sort=False)
        merged = merged.drop(columns=["__source_row_num__"])

    # Zusätzlich exakte Satz+Label-Duplikate entfernen.
    merged = merged.drop_duplicates(subset=["sentence", "gold_label"], keep="first")

    # Stabiles Ausgabe-Schema erzwingen, damit Training reproduzierbar bleibt.
    out_cols = ROLE_TRAIN_OUTPUT_COLUMNS
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged[out_cols]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_role_train_dataframe(merged, context="rebuilt role_train dataframe")
    merged.to_csv(output_path, index=False, encoding="utf-8")

    dist = merged["gold_label"].value_counts().to_dict()
    print("[ok] role_train.csv rebuilt from:")
    for src in used_sources:
        print(f"  - {src}")

    if mapped_counts:
        mapped_txt = ", ".join(
            f"{k}:{v}" for k, v in sorted(mapped_counts.items(), key=lambda x: (-x[1], x[0]))
        )
        print(f"[ok] mapped labels -> core: {mapped_txt}")
    else:
        print("[ok] mapped labels -> core: none")

    print(f"[ok] role_train rows: {len(merged)}")
    print("[ok] role_train class distribution:")
    for label in sorted(ALLOWED_LABELS):
        print(f"  {label}: {dist.get(label, 0)}")

    return len(merged), dist


def make_clean_training_frame(df: pd.DataFrame, text_col: str, label_series: pd.Series) -> pd.DataFrame:
    """
    Baut ein bereinigtes DataFrame mit normalisierten internen Spalten:
    - __text__
    - __label__
    plus optional `source_row`/`thread_id` für anti-Leakage-Splitting.
    """
    cleaned_df = pd.DataFrame({
        "__text__": df[text_col],
        "__label__": label_series,
    })
    if "source_row" in df.columns:
        cleaned_df["source_row"] = df["source_row"]
    if "thread_id" in df.columns:
        cleaned_df["thread_id"] = df["thread_id"]

    cleaned_df = cleaned_df.dropna(subset=["__text__", "__label__"])
    cleaned_df["__text__"] = cleaned_df["__text__"].astype(str).str.strip()
    cleaned_df["__label__"] = cleaned_df["__label__"].astype(str).str.strip()
    cleaned_df = cleaned_df[(cleaned_df["__text__"] != "") & (cleaned_df["__label__"] != "")]
    return cleaned_df


# ---------------------------------------------------------------------------
# Fixed split utilities
# ---------------------------------------------------------------------------
def _normalize_string_series(series: pd.Series) -> pd.Series:
    """Normalisiert textähnliche Serien auf getrimmte Strings."""
    return series.fillna("").astype(str).str.strip()


def _split_by_thread_target_rows(
    working: pd.DataFrame,
    desired_test_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Erstellt einen Split, bei dem ganze `thread_id` Gruppen im Test landen.

    Warum wichtig:
    - Sätze aus demselben Thread sind oft ähnlich.
    - Wenn gleicher Thread in Train und Test ist, werden Metriken zu optimistisch.
    """
    if "thread_id" not in working.columns:
        raise ValueError("thread_id missing for thread-based split.")
    if len(working) < 2:
        raise ValueError("Not enough rows for split.")

    ids = _normalize_string_series(working["thread_id"])

    # Fehlt `thread_id`, bekommt die Zeile eine eigene künstliche Gruppe.
    missing_mask = ids == ""
    if missing_mask.any():
        ids = ids.copy()
        ids.loc[missing_mask] = [f"__row__{i}" for i in working.index[missing_mask]]

    group_sizes = ids.value_counts().to_dict()
    group_ids = list(group_sizes.keys())
    rng = random.Random(RANDOM_STATE)
    rng.shuffle(group_ids)

    selected: list[str] = []
    current = 0
    for gid in group_ids:
        if current < desired_test_rows:
            selected.append(gid)
            current += int(group_sizes[gid])
        else:
            break

    # Feintuning: letzte Gruppe entfernen, falls Zielgröße dann näher ist.
    if len(selected) > 1:
        last_gid = selected[-1]
        without_last = current - int(group_sizes[last_gid])
        if abs(without_last - desired_test_rows) <= abs(current - desired_test_rows):
            selected.pop()

    if not selected:
        selected = [group_ids[0]]

    test_mask = ids.isin(set(selected))
    test_df = working[test_mask].copy()
    train_df = working[~test_mask].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError("Thread split produced empty train or test.")

    return train_df, test_df, {
        "split_mode": "thread_id",
        "thread_groups_test": int(len(selected)),
    }


def prepare_fixed_test_split(
    df: pd.DataFrame,
    test_csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Baut oder verwendet einen stabilen festen Test-Split.

    Erster Lauf:
    - feste Testdatei einmal erstellen
    Nächste Läufe:
    - dieselben Testzeilen wiederverwenden und nur Train-Zeilen aktualisieren
    """
    working = df.copy()
    working["__text__"] = _normalize_string_series(working["__text__"])
    working["__label__"] = _normalize_string_series(working["__label__"])
    if "source_row" in working.columns:
        working["source_row"] = _normalize_string_series(working["source_row"])
    if "thread_id" in working.columns:
        working["thread_id"] = _normalize_string_series(working["thread_id"])

    has_thread_info = "thread_id" in working.columns and (working["thread_id"] != "").any()
    test_csv_path = Path(test_csv_path)
    test_csv_path.parent.mkdir(parents=True, exist_ok=True)

    desired_test_rows = min(TEST_FIXED_ROWS, max(1, len(working) - 1))

    # Bestehendes fixes Testset wenn möglich weiterverwenden.
    if test_csv_path.exists():
        fixed_df = pd.read_csv(test_csv_path)
        try:
            validate_role_test_fixed_dataframe(fixed_df)
        except Exception as exc:
            print(f"[warn] role_test_fixed.csv schema invalid ({exc}). Recreate.")
            test_csv_path.unlink(missing_ok=True)
            return prepare_fixed_test_split(working, test_csv_path)
        fixed_text_col = find_column(fixed_df, ["sentence", "text", "__text__"])
        fixed_label_col = find_column(fixed_df, ["gold_label", "label", "__label__"])
        if fixed_text_col is None or fixed_label_col is None:
            print("[warn] role_test_fixed.csv invalid. Recreate.")
            test_csv_path.unlink(missing_ok=True)
            return prepare_fixed_test_split(working, test_csv_path)
        if has_thread_info and "thread_id" not in fixed_df.columns:
            print("[warn] role_test_fixed.csv has no thread_id. Recreate for thread split.")
            test_csv_path.unlink(missing_ok=True)
            return prepare_fixed_test_split(working, test_csv_path)

        fixed_labels = _normalize_string_series(fixed_df[fixed_label_col]).map(map_gold_label_to_core)
        test_df = pd.DataFrame({
            "__text__": _normalize_string_series(fixed_df[fixed_text_col]),
            "__label__": fixed_labels,
        })
        if "source_row" in fixed_df.columns:
            test_df["source_row"] = _normalize_string_series(fixed_df["source_row"])
        if "thread_id" in fixed_df.columns:
            test_df["thread_id"] = _normalize_string_series(fixed_df["thread_id"])

        test_df = test_df[(test_df["__text__"] != "") & (test_df["__label__"] != "")]
        if len(test_df) == 0:
            print("[warn] role_test_fixed.csv empty after cleanup. Recreate.")
            test_csv_path.unlink(missing_ok=True)
            return prepare_fixed_test_split(working, test_csv_path)

        # Testgröße stabil halten, sobald fixed file existiert.
        desired_test_rows = len(test_df)

        # Train-Maske mit stärkstem Matching-Signal zuerst bauen.
        train_mask = pd.Series(True, index=working.index)
        match_mode = "none"

        if has_thread_info and "thread_id" in test_df.columns:
            fixed_threads = set(test_df["thread_id"].replace("", pd.NA).dropna().tolist())
            if fixed_threads:
                train_mask = ~working["thread_id"].isin(fixed_threads)
                match_mode = "thread_id"

        # Fallback 2: Match über source_row.
        if match_mode == "none" and "source_row" in working.columns and "source_row" in test_df.columns:
            fixed_ids = set(test_df["source_row"].replace("", pd.NA).dropna().tolist())
            if fixed_ids:
                train_mask = ~working["source_row"].isin(fixed_ids)
                match_mode = "source_row"

        # Fallback 3: Match über Satztext.
        if match_mode == "none":
            fixed_sentences = set(test_df["__text__"].tolist())
            if fixed_sentences:
                train_mask = ~working["__text__"].isin(fixed_sentences)
                match_mode = "sentence"

        train_df = working[train_mask].copy()
        if len(train_df) == 0:
            raise ValueError("Fixed test set covers all rows. Training not possible.")

        missing_in_train = 0
        if (
            match_mode == "thread_id"
            and "thread_id" in working.columns
            and "thread_id" in test_df.columns
        ):
            fixed_threads = set(test_df["thread_id"].replace("", pd.NA).dropna().tolist())
            working_threads = set(working["thread_id"].replace("", pd.NA).dropna().tolist())
            missing_in_train = len(fixed_threads - working_threads)
        elif "source_row" in working.columns and "source_row" in test_df.columns:
            fixed_ids = set(test_df["source_row"].replace("", pd.NA).dropna().tolist())
            working_ids = set(working["source_row"].replace("", pd.NA).dropna().tolist())
            missing_in_train = len(fixed_ids - working_ids)

        return train_df, test_df, {
            "created_new": False,
            "match_mode": match_mode,
            "test_rows": int(len(test_df)),
            "train_rows": int(len(train_df)),
            "test_missing_in_train": int(missing_in_train),
            "path": str(test_csv_path),
        }

    # Wenn noch kein fixed file existiert, jetzt neu erstellen.
    split_mode = "new_split_row_level"
    thread_group_count = 0
    can_thread_split = has_thread_info

    if can_thread_split:
        try:
            train_df, test_df, thread_info = _split_by_thread_target_rows(
                working,
                desired_test_rows=desired_test_rows,
            )
            split_mode = "new_split_thread_id"
            thread_group_count = int(thread_info.get("thread_groups_test", 0))
        except Exception as exc:
            print(f"[warn] Thread split failed ({exc}). Falling back to row-level split.")
            stratify = working["__label__"] if working["__label__"].nunique() > 1 else None
            train_df, test_df = train_test_split(
                working,
                test_size=desired_test_rows if len(working) > desired_test_rows else TEST_SIZE_FALLBACK,
                random_state=RANDOM_STATE,
                stratify=stratify,
            )
    else:
        stratify = working["__label__"] if working["__label__"].nunique() > 1 else None
        train_df, test_df = train_test_split(
            working,
            test_size=desired_test_rows if len(working) > desired_test_rows else TEST_SIZE_FALLBACK,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

    fixed_out = test_df[["__text__", "__label__"]].copy()
    fixed_out["__label__"] = _normalize_string_series(fixed_out["__label__"]).map(map_gold_label_to_core)
    fixed_out = fixed_out.rename(
        columns={"__text__": "sentence", "__label__": "gold_label"}
    )
    insert_pos = 0
    if "source_row" in test_df.columns:
        fixed_out.insert(insert_pos, "source_row", test_df["source_row"])
        insert_pos += 1
    if "thread_id" in test_df.columns:
        fixed_out.insert(insert_pos, "thread_id", test_df["thread_id"])

    validate_role_test_fixed_dataframe(fixed_out, context="new role_test_fixed dataframe")
    fixed_out.to_csv(test_csv_path, index=False, encoding="utf-8")
    print(f"[ok] New fixed test set created: {test_csv_path}")
    print(f"[ok] Fixed test rows: {len(test_df)}")

    return train_df, test_df, {
        "created_new": True,
        "match_mode": split_mode,
        "test_rows": int(len(test_df)),
        "train_rows": int(len(train_df)),
        "test_thread_groups": int(thread_group_count),
        "test_missing_in_train": 0,
        "path": str(test_csv_path),
    }
