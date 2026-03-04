"""
config.py

Zentrale Konfiguration für die Trainings-Pipeline des Rollen-Klassifikators.
Wenn du Verhalten anpassen willst, starte in dieser Datei.

Einsteiger-Tipp:
- Diese Datei ist die "Single Source of Truth" für Pfade und Konstanten.
- Werte nicht in mehreren Dateien doppelt hart codieren.
"""

from pathlib import Path


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
# `ML_DIR` zeigt auf `backend/ml`.
ML_DIR = Path(__file__).resolve().parent
# `BACKEND_DIR` zeigt auf `backend`.
BACKEND_DIR = ML_DIR.parent

# Kern-Dateien für Training und Evaluation.
CSV_PATH = ML_DIR / "data" / "role_train.csv"
TEST_FIXED_PATH = ML_DIR / "data" / "role_test_fixed.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "role_model.joblib"
META_PATH = MODEL_DIR / "role_model_meta.json"

# Report-Ordner für Prüfberichte.
REPORTS_DIR = ML_DIR / "data" / "reports"
LINT_REPORTS_DIR = REPORTS_DIR / "lint"

GOLD_PATHS = [
    ML_DIR / "data" / "processed" / "cmv_gold_master.csv",
]


# -------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------
# Finale Kernklassen, die das Modell lernt.
ALLOWED_LABELS = {"claim", "premise", "objection", "conclusion", "other"}

# Optionale feinere Labels werden auf 5 Kernklassen gemappt.
# So kann man reichere manuelle Labels importieren, aber weiter ein 5-Klassen-Modell trainieren.
LABEL_MAP_TO_CORE = {
    "concession": "objection",
    "question": "objection",
    "rhetorical_question": "objection",
    "rebuttal": "objection",
    "dismissal": "other",
    "challenge": "objection",
    "counterexample": "objection",
    "counterargument": "objection",
    "counterevidence": "objection",
    "counterproposal": "objection",
    "counteraccusation": "objection",
    "counterposition": "objection",
    "anticipatory_objection": "objection",
    "reductio": "objection",
    "paradox": "objection",
    "analogy": "premise",
    "anecdote": "premise",
    "distinction": "premise",
    "definition": "premise",
    "qualification": "premise",
    "evidence": "premise",
    "example": "premise",
    "answer": "premise",
    "speculation": "claim",
    "hypothesis": "claim",
    "conditional_claim": "claim",
    "position_statement": "claim",
    "argument": "claim",
    "recommendation": "claim",
    "proposal": "claim",
    "request_for_change": "claim",
    "counterclaim": "claim",
    "clarification": "other",
    "clarification_request": "other",
    "recommendation_request": "other",
    "quote": "other",
    "reported_speech": "other",
    "meta": "other",
    "restatement": "other",
    "aphorism": "other",
}


# -------------------------------------------------------------------
# Spaltennamen und Schema
# -------------------------------------------------------------------
# Kandidaten-Spaltennamen für Dateien mit leicht unterschiedlichen Headern.
TEXT_COLUMN_CANDIDATES = ["text", "sentence", "content", "span"]
WEAK_LABEL_COLUMN_CANDIDATES = ["weak_label", "label", "role", "y", "target", "class"]

GOLD_REQUIRED_COLUMNS = ["sentence", "gold_label", "source_row", "thread_id"]
ROLE_TRAIN_REQUIRED_COLUMNS = ["sentence", "gold_label", "source_row", "thread_id"]
ROLE_TEST_FIXED_REQUIRED_COLUMNS = ["source_row", "thread_id", "sentence", "gold_label"]

ROLE_TRAIN_OUTPUT_COLUMNS = [
    "sentence",
    "gold_label",
    "weak_label",
    "source_row",
    "thread_id",
    "comment_id",
    "sentence_index",
    "label_notes",
]


# -------------------------------------------------------------------
# Split settings
# -------------------------------------------------------------------
# Zielgröße des festen Testsets.
TEST_FIXED_ROWS = 2000
TEST_SIZE_FALLBACK = 0.2
RANDOM_STATE = 42
CV_N_SPLITS = 5

# Optionales leichtes Oversampling nur im TRAIN-Teil
# (Testset wird nie oversampled).
TRAIN_LIGHT_OVERSAMPLE = True


# -------------------------------------------------------------------
# Modell-Sucheinstellungen (GridSearch-Raum)
# -------------------------------------------------------------------
WORD_NGRAM_OPTIONS = [(1, 2)]
WORD_MIN_DF_OPTIONS = [1, 2]
WORD_MAX_DF_OPTIONS = [0.9, 1.0]

CHAR_NGRAM_OPTIONS = [(3, 5), (3, 6)]
CHAR_MIN_DF_OPTIONS = [2, 3]
CHAR_MAX_DF_OPTIONS = [1.0]

LOGREG_C_OPTIONS = [0.5, 1.0, 3.0]
SVC_C_OPTIONS = [0.5, 1.0, 2.0, 3.0]
CLASS_WEIGHT_OPTIONS = ["balanced"]

UNION_LOGREG_C_OPTIONS = [0.8, 1.0]
UNION_WORD_MAX_DF = 0.9
UNION_WORD_MIN_DF = 1
UNION_CHAR_MAX_DF = 1.0
UNION_CHAR_MIN_DF_OPTIONS = [2, 3]
