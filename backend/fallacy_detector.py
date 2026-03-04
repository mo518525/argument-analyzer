"""
fallacy_detector.py

Eigenständiges Modul für Fehlschluss-Erkennung.

Warum separate Datei?
- Rollen-Klassifikation und Fehlschluss-Erkennung sind zwei verschiedene Aufgaben.
- Die Trennung macht das Backend leichter verständlich und wartbar.
"""

from __future__ import annotations


def detect_fallacies(sentence: str):
    """
    Erkennt mögliche Fehlschlüsse in einem Satz über Keyword-Heuristiken.

    Wie das Scoring funktioniert:
    - Jeder Fehlschluss-Typ hat Trigger-Phrasen.
    - Bei Treffern wird ein Basis-Score gesetzt.
    - Weitere Treffer erhöhen den Score leicht.
    - Der Score ist gedeckelt, damit er stabil bleibt.
    """

    s = sentence.lower()
    fallacies = []

    def add_if_triggered(
        label: str,
        triggers: list[str],
        base: float,
        step: float,
        max_score: float = 0.95,
    ):
        """Fügt einen Fehlschluss-Eintrag hinzu, sobald ein Trigger gefunden wurde."""
        hits = sum(1 for t in triggers if t in s)
        if hits > 0:
            score = base + (hits - 1) * step
            score = min(score, max_score)
            fallacies.append({"label": label, "score": round(score, 2), "hits": hits})

    add_if_triggered(
        label="appeal_to_common_sense",
        triggers=["everyone knows", "obviously", "clearly", "of course"],
        base=0.55,
        step=0.07,
    )

    add_if_triggered(
        label="false_dilemma",
        triggers=["either", " or ", "only", "no other", "two options"],
        base=0.55,
        step=0.05,
    )

    add_if_triggered(
        label="ad_hominem",
        triggers=[
            "you are stupid",
            "you're stupid",
            "idiot",
            "moron",
            "liar",
            "ignorant",
            "trash",
            "clown",
            "shut up",
        ],
        base=0.65,
        step=0.08,
    )

    add_if_triggered(
        label="strawman",
        triggers=[
            "so you are saying",
            "so you're saying",
            "you claim that",
            "you want to",
            "your position is that",
            "so basically",
        ],
        base=0.58,
        step=0.06,
    )

    add_if_triggered(
        label="slippery_slope",
        triggers=[
            "if we allow",
            "if we accept",
            "this will lead to",
            "then soon",
            "eventually",
            "next thing you know",
            "will inevitably",
        ],
        base=0.58,
        step=0.06,
    )

    add_if_triggered(
        label="hasty_generalization",
        triggers=["always", "never", "all of them", "everyone", "no one", "they are all"],
        base=0.55,
        step=0.05,
    )

    add_if_triggered(
        label="appeal_to_authority",
        triggers=[
            "experts say",
            "scientists say",
            "a professor said",
            "according to the expert",
            "authority says",
        ],
        base=0.55,
        step=0.06,
    )

    add_if_triggered(
        label="appeal_to_emotion",
        triggers=[
            "think of the children",
            "it's disgusting",
            "it's terrifying",
            "how dare you",
            "you should be ashamed",
            "fear",
        ],
        base=0.55,
        step=0.06,
    )

    add_if_triggered(
        label="false_cause",
        triggers=["since then", "must be because", "caused by", "because of that"],
        base=0.50,
        step=0.06,
    )

    # Pro Fehlschluss-Typ bleibt nur der stärkste Treffer (keine Duplikate).
    unique = {}
    for f in fallacies:
        if f["label"] not in unique or f["score"] > unique[f["label"]]["score"]:
            unique[f["label"]] = f

    # Für die UI: stärkste Treffer zuerst, Hilfsfeld "hits" entfernen.
    out = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
    for f in out:
        f.pop("hits", None)
    return out
