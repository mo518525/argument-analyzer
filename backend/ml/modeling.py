"""
modeling.py

Dieses Modul enthält die Modellsuche.
Es baut Kandidaten-Pipelines und gibt das beste Modell nach Macro-F1 im CV zurück.

Einsteiger-Zusammenfassung:
- Wir testen mehrere Kombinationen aus Text-Features und Klassifikatoren.
- GridSearchCV wählt die beste Variante auf Validierungs-Folds.
"""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import (
    CHAR_MAX_DF_OPTIONS,
    CHAR_MIN_DF_OPTIONS,
    CHAR_NGRAM_OPTIONS,
    CLASS_WEIGHT_OPTIONS,
    CV_N_SPLITS,
    LOGREG_C_OPTIONS,
    RANDOM_STATE,
    SVC_C_OPTIONS,
    UNION_CHAR_MAX_DF,
    UNION_CHAR_MIN_DF_OPTIONS,
    UNION_LOGREG_C_OPTIONS,
    UNION_WORD_MAX_DF,
    UNION_WORD_MIN_DF,
    WORD_MAX_DF_OPTIONS,
    WORD_MIN_DF_OPTIONS,
    WORD_NGRAM_OPTIONS,
)


def build_text_model_search():
    """
    Baut einen Suchraum mit:
    - Wort-TF-IDF-Modellen
    - Zeichen-TF-IDF-Modellen
    - kombinierten Wort+Zeichen-Modellen (FeatureUnion)
    """
    # Kombi-Featureblock #1:
    # - Wort-N-Gramme für semantische Hinweise
    # - Zeichen-N-Gramme für robuste Muster (z. B. Tippfehler/Varianten)
    # - char min_df=2 => Zeichenmuster müssen mindestens 2x vorkommen
    union_word_char_min2 = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            analyzer="word",
            ngram_range=WORD_NGRAM_OPTIONS[0],
            min_df=UNION_WORD_MIN_DF,
            max_df=UNION_WORD_MAX_DF,
        )),
        ("char", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM_OPTIONS[0],
            min_df=UNION_CHAR_MIN_DF_OPTIONS[0],
            max_df=UNION_CHAR_MAX_DF,
        )),
    ])

    # Kombi-Featureblock #2:
    # Gleich wie oben, aber strengeres char min_df=3.
    # Das kann Rauschen reduzieren.
    union_word_char_min3 = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            analyzer="word",
            ngram_range=WORD_NGRAM_OPTIONS[0],
            min_df=UNION_WORD_MIN_DF,
            max_df=UNION_WORD_MAX_DF,
        )),
        ("char", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM_OPTIONS[0],
            min_df=UNION_CHAR_MIN_DF_OPTIONS[1],
            max_df=UNION_CHAR_MAX_DF,
        )),
    ])

    # Basis-Pipeline:
    # GridSearch ersetzt "features" und "clf" je nach Kandidat im Suchraum.
    pipe = Pipeline(
        steps=[
            ("features", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
            )),
            ("clf", LinearSVC()),
        ]
    )

    # Suchraum:
    # Jedes Dict steht für eine Modell-Familie.
    param_grid = [
        # Nur Wort-Features + Logistic Regression
        {
            "features__analyzer": ["word"],
            "features__ngram_range": WORD_NGRAM_OPTIONS,
            "features__min_df": WORD_MIN_DF_OPTIONS,
            "features__max_df": WORD_MAX_DF_OPTIONS,
            "clf": [LogisticRegression(max_iter=3000)],
            "clf__C": LOGREG_C_OPTIONS,
            "clf__class_weight": CLASS_WEIGHT_OPTIONS,
        },
        # Nur Wort-Features + LinearSVC
        {
            "features__analyzer": ["word"],
            "features__ngram_range": WORD_NGRAM_OPTIONS,
            "features__min_df": WORD_MIN_DF_OPTIONS,
            "features__max_df": WORD_MAX_DF_OPTIONS,
            "clf": [LinearSVC()],
            "clf__C": SVC_C_OPTIONS,
            "clf__class_weight": CLASS_WEIGHT_OPTIONS,
        },
        # Nur Zeichen-Features + LinearSVC
        {
            "features__analyzer": ["char_wb"],
            "features__ngram_range": CHAR_NGRAM_OPTIONS,
            "features__min_df": CHAR_MIN_DF_OPTIONS,
            "features__max_df": CHAR_MAX_DF_OPTIONS,
            "clf": [LinearSVC()],
            "clf__C": SVC_C_OPTIONS,
            "clf__class_weight": CLASS_WEIGHT_OPTIONS,
        },
        # Kombinierte Wort+Zeichen-Features + Logistic Regression
        {
            "features": [union_word_char_min2, union_word_char_min3],
            "clf": [LogisticRegression(max_iter=4000)],
            "clf__C": UNION_LOGREG_C_OPTIONS,
            "clf__class_weight": CLASS_WEIGHT_OPTIONS,
        },
    ]

    # StratifiedKFold behält die Klassenverteilung pro Fold ähnlich.
    cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return pipe, param_grid, cv


def fit_best_text_model(X_train, y_train, stage_name: str):
    """
    Startet GridSearch und gibt zurück:
    - bestes trainiertes Modell
    - beste Parameter als Strings
    - beste CV-Macro-F1
    """
    # Suchraum + CV-Setup laden.
    pipe, param_grid, cv = build_text_model_search()

    # GridSearch testet alle Konfigurationen und merkt sich die beste.
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    print(f" Starte GridSearch {stage_name} (kann etwas dauern)...")
    search.fit(X_train, y_train)

    # Beste gefundene Variante extrahieren.
    best_model = search.best_estimator_
    best_params_serializable = {k: str(v) for k, v in search.best_params_.items()}
    return best_model, best_params_serializable, float(search.best_score_)
