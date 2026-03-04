import bz2
import csv
import json
import os
import re
from collections import Counter


# =========================
# CMV -> CSV for labeling
# =========================
# This script prepares sentence-level data for human annotation:
# 1) fixed label definitions are written to a guide file
# 2) weak_label is a suggestion, gold_label is left empty for manual correction
# 3) each row includes context ids (thread_id, comment_id, parent_id, level)
# =========================

IN_PATH = "backend/ml/data/raw/cmv/threads.jsonl.bz2"
OUT_DIR = "backend/ml/data/processed"
OUT_CSV = os.path.join(OUT_DIR, "cmv_role_weak_5000.csv")
OUT_GUIDE = os.path.join(OUT_DIR, "cmv_role_labeling_guide.md")

THREAD_LIMIT = 5000
MAX_SENTENCES_TOTAL = 50000
MIN_SENTENCE_LEN = 10

LABEL_GUIDELINES = {
    "claim": "Main stance or assertion that can be argued for/against.",
    "premise": "Reason, evidence, or support for a claim.",
    "objection": "Disagreement, counterpoint, or challenge to another statement.",
    "conclusion": "Wrap-up, final takeaway, or inferred result from earlier points.",
    "other": "Question, greeting, anecdote, hedging, or text with no clear argument role.",
}


def write_labeling_guide(path: str) -> None:
    """Erzeugt eine kurze Markdown-Anleitung fuer menschliche Annotatoren."""
    lines = [
        "# CMV Role Labeling Guide",
        "",
        "Use exactly one label per sentence from this set:",
        "",
    ]
    for label, definition in LABEL_GUIDELINES.items():
        lines.append(f"- `{label}`: {definition}")

    lines += [
        "",
        "Decision order for ambiguous cases:",
        "1. If it directly attacks/challenges another point -> `objection`.",
        "2. If it gives explicit support/evidence/reason -> `premise`.",
        "3. If it states the main stance -> `claim`.",
        "4. If it summarizes/infers final point -> `conclusion`.",
        "5. Otherwise -> `other`.",
        "",
        "Annotation workflow:",
        "- Keep `weak_label` as model suggestion.",
        "- Fill `gold_label` manually.",
        "- Optional notes go into `label_notes`.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def safe_text(value: object) -> str:
    """Gibt getrimmten String zurueck, sonst leeren String."""
    if isinstance(value, str):
        return value.strip()
    return ""


def clean_text(text: str) -> str:
    """Entfernt Markdown/Noise und normalisiert Leerzeichen."""
    if not text:
        return ""

    # Remove quote blocks, code blocks, urls, then normalize spaces.
    text = re.sub(r"(?m)^\s*>.*$", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Satz-Splitter fuer den Schritt vor weak labeling."""
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_LEN]


def weak_label(sentence: str) -> str:
    """
    Leichte weak-label Vorschlagslogik.

    Das ist nur eine Start-Hilfe fuer manuelles Labeln.
    """
    s = sentence.lower().strip()
    if len(s) < 20:
        return "other"

    objection_markers = [
        "i disagree",
        "you are wrong",
        "you're wrong",
        "however",
        "but ",
        "although",
        "yet ",
        "nevertheless",
        "on the other hand",
        "that said",
    ]
    if any(marker in s for marker in objection_markers):
        return "objection"

    conclusion_markers = [
        "therefore",
        "thus",
        "hence",
        "consequently",
        "overall",
        "in conclusion",
        "to conclude",
    ]
    if any(marker in s for marker in conclusion_markers):
        return "conclusion"

    premise_markers = [
        "because",
        "since",
        "due to",
        "given that",
        "for example",
        "for instance",
        "evidence",
        "data shows",
        "studies show",
        "research shows",
    ]
    if any(marker in s for marker in premise_markers):
        return "premise"

    claim_markers = [
        "i think",
        "i believe",
        "i argue",
        "in my opinion",
        "should",
        "must",
        "we need",
        "it is",
        "it's",
    ]
    if any(marker in s for marker in claim_markers):
        return "claim"

    return "other"


def pick_comment_list(obj: dict) -> list[dict]:
    """Unterstuetzt mehrere moegliche JSON-Strukturen fuer Threads."""
    for key in ("comments", "posts", "replies"):
        value = obj.get(key)
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    return []


def iter_sources(obj: dict) -> list[dict]:
    """Sammelt OP-Titel/Body und Kommentare als Label-Quellen."""
    sources: list[dict] = []
    thread_id = safe_text(obj.get("id") or obj.get("name"))
    thread_title = safe_text(obj.get("title") or obj.get("op_title"))
    thread_permalink = safe_text(obj.get("permalink"))
    thread_author = safe_text(obj.get("author"))

    if thread_title:
        sources.append(
            {
                "thread_id": thread_id,
                "thread_title": thread_title,
                "thread_permalink": thread_permalink,
                "source_type": "op_title",
                "comment_id": "",
                "parent_id": "",
                "level": "0",
                "author": thread_author,
                "text": thread_title,
            }
        )

    op_body = safe_text(obj.get("selftext") or obj.get("op_text") or obj.get("text"))
    if op_body:
        sources.append(
            {
                "thread_id": thread_id,
                "thread_title": thread_title,
                "thread_permalink": thread_permalink,
                "source_type": "op_body",
                "comment_id": "",
                "parent_id": "",
                "level": "0",
                "author": thread_author,
                "text": op_body,
            }
        )

    for comment in pick_comment_list(obj):
        body = safe_text(comment.get("body") or comment.get("text"))
        if not body:
            continue

        sources.append(
            {
                "thread_id": thread_id,
                "thread_title": thread_title,
                "thread_permalink": thread_permalink,
                "source_type": "comment",
                "comment_id": safe_text(comment.get("id") or comment.get("name")),
                "parent_id": safe_text(comment.get("parent_id")),
                "level": str(comment.get("level", "")),
                "author": safe_text(comment.get("author")),
                "text": body,
            }
        )

    return sources


def main() -> None:
    """Parst den komprimierten CMV-Dump und schreibt eine Satz-CSV."""
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
    write_labeling_guide(OUT_GUIDE)

    threads_read = 0
    sentences_written = 0
    label_counts: Counter[str] = Counter()

    with bz2.open(IN_PATH, "rt", encoding="utf-8") as f_in, open(
        OUT_CSV, "w", newline="", encoding="utf-8"
    ) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
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
        )

        for line in f_in:
            if threads_read >= THREAD_LIMIT:
                break

            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            for source in iter_sources(obj):
                sentences = split_sentences(source["text"])
                for sentence_index, sentence in enumerate(sentences, start=1):
                    suggested = weak_label(sentence)
                    writer.writerow(
                        [
                            source["thread_id"],
                            source["thread_title"],
                            source["thread_permalink"],
                            source["source_type"],
                            source["comment_id"],
                            source["parent_id"],
                            source["level"],
                            source["author"],
                            sentence_index,
                            sentence,
                            suggested,
                            "",
                            "",
                        ]
                    )
                    sentences_written += 1
                    label_counts[suggested] += 1

                    if sentences_written >= MAX_SENTENCES_TOTAL:
                        print(f"[stop] Reached MAX_SENTENCES_TOTAL={MAX_SENTENCES_TOTAL}")
                        print(f"[ok] CSV: {OUT_CSV}")
                        print(f"[ok] Guide: {OUT_GUIDE}")
                        print("[ok] Weak label distribution:")
                        for label in LABEL_GUIDELINES:
                            print(f"  {label}: {label_counts.get(label, 0)}")
                        return

            threads_read += 1
            if threads_read % 250 == 0:
                print(
                    f"[prog] threads={threads_read} sentences={sentences_written}"
                )

    print(f"[ok] Threads read: {threads_read}")
    print(f"[ok] Sentences written: {sentences_written}")
    print(f"[ok] CSV: {OUT_CSV}")
    print(f"[ok] Guide: {OUT_GUIDE}")
    print("[ok] Weak label distribution:")
    for label in LABEL_GUIDELINES:
        print(f"  {label}: {label_counts.get(label, 0)}")


if __name__ == "__main__":
    main()
