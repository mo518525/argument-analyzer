"""
download_cmv.py

Laedt den CMV-Thread-Dump in den lokalen Speicher:
`backend/ml/data/raw/cmv/threads.jsonl.bz2`
"""

import os

import requests


# Zielordner fuer rohe Datensatz-Dateien.
OUT_DIR = "backend/ml/data/raw/cmv"
os.makedirs(OUT_DIR, exist_ok=True)

# Oeffentliche Datensatz-URL.
URL = "https://zenodo.org/records/3778298/files/threads.jsonl.bz2?download=1"
OUT_PATH = os.path.join(OUT_DIR, "threads.jsonl.bz2")


def download_stream(url: str, out_path: str) -> None:
    """
    Stream-Download einer grossen Datei in Blöcken.

    Warum Streaming?
    - Verhindert, dass die ganze Datei in den RAM geladen wird.
    """
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    """CLI-Einstiegspunkt."""
    if os.path.exists(OUT_PATH):
        print(f"[skip] Datei existiert schon: {OUT_PATH}")
        return

    print("[down] Lade CMV threads.jsonl.bz2 runter ...")
    download_stream(URL, OUT_PATH)
    print(f"[ok] Fertig: {OUT_PATH}")


if __name__ == "__main__":
    main()
