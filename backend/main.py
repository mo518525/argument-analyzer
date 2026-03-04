"""
main.py
=======

Dieses Modul ist der zentrale Backend-Einstiegspunkt.

Was hier passiert:
1) FastAPI-App wird aufgebaut.
2) Das trainierte Rollenmodell wird beim Start einmal geladen.
3) Der Endpunkt `/analyze` nimmt Text an und gibt Analyse-Ergebnisse zurÃ¼ck.

Warum diese Datei wichtig ist:
- Sie verbindet alle Teile des Projekts (Rollenmodell + Fehlschluss-Erkennung + API).
- Wenn du verstehen willst, wie der gesamte Ablauf funktioniert, starte hier.
"""

from pathlib import Path
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

try:
    # Normaler Import, wenn `backend` als Paket lÃ¤uft.
    from .fallacy_detector import detect_fallacies
except ImportError:
    # Fallback, wenn man main.py direkt startet.
    from fallacy_detector import detect_fallacies


# ============================================================================
# 1) App-Konfiguration
# ============================================================================
# Wir erlauben hier Requests vom Frontend (localhost:3000), damit Browser-Calls
# auf den Backend-Endpunkt funktionieren.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Globales Modellobjekt:
# - Wird beim Start geladen.
# - Danach fÃ¼r alle Requests wiederverwendet.
# - So vermeiden wir teures Neu-Laden bei jedem API-Call.
ROLE_MODEL = None


@app.on_event("startup")
def load_role_model():
    """
    LÃ¤dt das trainierte Rollenmodell beim Serverstart.

    Wichtig:
    - Wenn keine Modelldatei existiert, lÃ¤uft die API trotzdem.
    - In diesem Fall nutzt `/analyze` automatisch Heuristiken als Fallback.
    """
    global ROLE_MODEL

    # Pfad relativ zu dieser Datei bestimmen.
    base_dir = Path(__file__).resolve().parent
    model_path = (base_dir / "models" / "role_model.joblib").resolve()

    if not model_path.exists():
        print("Warnung: role_model.joblib nicht gefunden:", model_path)
        ROLE_MODEL = None
        return

    ROLE_MODEL = joblib.load(model_path)
    print("OK: role_model geladen:", model_path)


# ============================================================================
# 2) Request/Response-Hilfen
# ============================================================================
class AnalyzeRequest(BaseModel):
    """
    Eingabe-Schema fÃ¼r den Endpunkt `/analyze`.

    Erwartet:
    {
      "text": "Beliebiger lÃ¤ngerer Text ..."
    }
    """

    text: str


def split_sentences(text: str) -> list[str]:
    """
    Teilt einen Text robust in SÃ¤tze.

    Warum "robust"?
    - Nutzer fÃ¼gen oft ZeilenumbrÃ¼che, Listen oder unregelmÃ¤ÃŸige AbstÃ¤nde ein.
    - Deshalb splitten wir zuerst nach Zeilen, dann nach Satzzeichen.

    Ablauf:
    1) Leere Zeilen entfernen.
    2) Pro Zeile nach `.`, `!`, `?` splitten.
    3) Leere Teile verwerfen.
    """
    # Schritt 1: Zeilen sauber vorbereiten.
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    sentences: list[str] = []

    # Schritt 2+3: Jede Zeile weiter in SÃ¤tze zerlegen.
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def _softmax(scores: np.ndarray) -> np.ndarray:
    """
    Wandelt rohe Modell-Scores in wahrscheinlichkeit-Ã¤hnliche Werte um.

    Das ist nÃ¼tzlich, wenn ein Modell nur `decision_function` liefert und
    keine echten `predict_proba`-Werte.
    """
    # Numerisch stabile Variante: vorher um den Maximalwert verschieben.
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores)

    # Sicherheitsfall: falls etwas numerisch schiefgeht, gleichverteilen.
    if denom <= 0:
        return np.ones_like(scores) / max(1, len(scores))
    return exp_scores / denom


def _prob_dict_from_model(sentence: str):
    """
    Liefert pro Klasse eine Konfidenz (wenn mÃ¶glich).

    Reihenfolge:
    1) Wenn vorhanden: `predict_proba`
    2) Sonst: `decision_function` + Softmax
    3) Sonst: None
    """
    global ROLE_MODEL
    classes = [str(c) for c in getattr(ROLE_MODEL, "classes_", [])]
    if not classes:
        return None

    # Best Case: Modell liefert direkt Klassenwahrscheinlichkeiten.
    if hasattr(ROLE_MODEL, "predict_proba"):
        probs = np.asarray(ROLE_MODEL.predict_proba([sentence])[0], dtype=float)
        if len(probs) != len(classes):
            return None
        return {c: float(p) for c, p in zip(classes, probs)}

    # Fallback fÃ¼r Modelle ohne predict_proba (z. B. LinearSVC).
    if hasattr(ROLE_MODEL, "decision_function"):
        raw = np.asarray(ROLE_MODEL.decision_function([sentence]))
        raw = raw[0] if raw.ndim == 2 else raw
        raw = np.asarray(raw, dtype=float)

        # Spezialfall: BinÃ¤re Klassifikation mit nur einem Score.
        if len(classes) == 2 and raw.size == 1:
            pos = 1.0 / (1.0 + np.exp(-raw.item()))
            probs = np.array([1.0 - pos, pos], dtype=float)
            return {c: float(p) for c, p in zip(classes, probs)}

        if raw.size != len(classes):
            return None

        probs = _softmax(raw)
        return {c: float(p) for c, p in zip(classes, probs)}

    return None


# ============================================================================
# 3) Rollen-Klassifikation
# ============================================================================
def classify_role_ml(sentence: str):
    """
    Klassifiziert die Rolle eines Satzes mit dem ML-Modell.

    RÃ¼ckgabe enthÃ¤lt:
    - `label` / `score` (vorhergesagte Klasse + Sicherheit)
    - `second_label` / `second_score` (zweitbeste Klasse)
    - `margin` (Abstand Top1 zu Top2)
    """
    global ROLE_MODEL

    stripped = sentence.strip()

    # Leere Eingabe -> neutrales Default-Ergebnis.
    if not stripped:
        return {
            "label": "other",
            "score": 0.50,
            "raw_label": "other",
            "raw_score": 0.50,
            "second_label": "",
            "second_score": 0.0,
            "margin": 1.0,
            "adjusted": False,
            "adjust_reason": "",
        }

    # Wenn kein Modell geladen ist, nutzen wir Heuristiken als Ersatz.
    if ROLE_MODEL is None:
        base = classify_role(sentence)
        return {
            "label": base["label"],
            "score": base["score"],
            "raw_label": base["label"],
            "raw_score": base["score"],
            "second_label": "",
            "second_score": 0.0,
            "margin": 1.0,
            "adjusted": False,
            "adjust_reason": "model_not_loaded",
        }

    predicted_label = str(ROLE_MODEL.predict([stripped])[0])
    prob_dict = _prob_dict_from_model(stripped)

    # Falls Konfidenzen verfÃ¼gbar sind, berechnen wir Top1/Top2 sauber.
    if prob_dict:
        ranked = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = ranked[0]
        if len(ranked) > 1:
            second_label, second_score = ranked[1]
        else:
            second_label, second_score = "", 0.0

        predicted_score = prob_dict.get(predicted_label, top_score)
        return {
            "label": predicted_label,
            "score": round(float(predicted_score), 2),
            "raw_label": predicted_label,
            "raw_score": round(float(predicted_score), 2),
            "second_label": second_label,
            "second_score": round(float(second_score), 2),
            "margin": round(float(top_score - second_score), 4),
            "adjusted": False,
            "adjust_reason": "",
        }

    # Ohne Konfidenzinfos liefern wir einen stabilen Default-Score.
    return {
        "label": predicted_label,
        "score": 0.80,
        "raw_label": predicted_label,
        "raw_score": 0.80,
        "second_label": "",
        "second_score": 0.0,
        "margin": 1.0,
        "adjusted": False,
        "adjust_reason": "",
    }


def classify_role(sentence: str):
    """
    Heuristische Rollen-Klassifikation (Fallback ohne ML-Modell).

    Idee:
    - Jede Rolle hat TriggerwÃ¶rter.
    - Trefferzahl wird in einen Score umgerechnet.
    - Bei Mehrdeutigkeit gilt eine feste PrioritÃ¤t.
    """
    s = sentence.strip().lower()

    def score_from_hits(hits: int, base: float, step: float, max_score: float = 0.95) -> float:
        """
        Rechnet Trefferzahl in Score um.

        Beispiel:
        - 1 Treffer -> base
        - 2 Treffer -> base + step
        """
        if hits <= 0:
            return 0.0
        return min(base + (hits - 1) * step, max_score)

    # Triggerlisten fÃ¼r jede Rollenklasse.
    conclusion_triggers = ["therefore", "thus", "hence", "it follows", "consequently"]
    premise_triggers = [
        "because",
        "since",
        "due to",
        "as a result",
        "given that",
        "for the reason",
        "experts say",
        "scientists say",
        "according to",
    ]
    objection_triggers = ["however", "but", "although", "yet", "nevertheless", "on the other hand"]
    claim_triggers = ["should", "must", "ought", "i think", "i believe", "we need to", "it is necessary"]

    def count_hits(triggers: list[str]) -> int:
        return sum(1 for trigger in triggers if trigger in s)

    # Treffer zÃ¤hlen.
    hits_conclusion = count_hits(conclusion_triggers)
    hits_premise = count_hits(premise_triggers)
    hits_objection = count_hits(objection_triggers)
    hits_claim = count_hits(claim_triggers)

    # Treffer in Scores umwandeln.
    score_conclusion = score_from_hits(hits_conclusion, base=0.70, step=0.08)
    score_premise = score_from_hits(hits_premise, base=0.65, step=0.08)
    score_objection = score_from_hits(hits_objection, base=0.60, step=0.07)
    score_claim = score_from_hits(hits_claim, base=0.58, step=0.07)

    # PrioritÃ¤t fÃ¼r KonfliktfÃ¤lle:
    # objection > conclusion > premise > claim > other
    if hits_objection > 0:
        return {"label": "objection", "score": round(score_objection, 2)}
    if hits_conclusion > 0:
        return {"label": "conclusion", "score": round(score_conclusion, 2)}
    if hits_premise > 0:
        return {"label": "premise", "score": round(score_premise, 2)}
    if hits_claim > 0:
        return {"label": "claim", "score": round(score_claim, 2)}

    return {"label": "other", "score": 0.50}


# ============================================================================
# 4) Health-Endpunkte
# ============================================================================
@app.get("/")
def root():
    """Einfacher Browser-Check: LÃ¤uft der Server grundsÃ¤tzlich?"""
    return {"status": "Backend lÃ¤uft"}


@app.get("/health")
def health():
    """
    Technischer Gesundheitscheck:
    - zeigt, ob das Rollenmodell geladen wurde.
    """
    return {"model_loaded": ROLE_MODEL is not None}


# ============================================================================
# 5) Haupt-Endpunkt
# ============================================================================
@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    """
    FÃ¼hrt die komplette Analyse-Pipeline aus.

    Pipeline:
    1) Text in SÃ¤tze teilen
    2) Pro Satz: Rollenklasse + FehlschlÃ¼sse
    """
    sentences = split_sentences(data.text)

    # FÃ¼r jeden Satz ein Ergebnisobjekt bauen.
    items = [
        {
            "id": i + 1,
            "text": sentence,
            "role": classify_role_ml(sentence),
            "fallacies": detect_fallacies(sentence),
        }
        for i, sentence in enumerate(sentences)
    ]
    return {"sentences": items}


