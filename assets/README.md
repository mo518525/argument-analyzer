# Assets-Ordner

Dieser Ordner ist für Bilder gedacht, die ins GitHub-Repository sollen.

## Zweck
- Screenshots für die Doku (`README.md`)
- Demo-Bilder für Präsentation

## Aktuell enthalten
- `frontend-input.png`
- `frontend-output.png`
- `model-metrics.png`

## Regeln
1. Keine sensiblen Daten in Bildern (API-Keys, private Daten, Tokens).
2. Dateinamen klar und stabil halten, z. B. `feature-name.png`.
3. Bilder für Doku möglichst komprimieren, damit das Repo klein bleibt.

## Nutzung in README.md
In Markdown mit relativem Pfad:

```md
![Frontend Input](assets/frontend-input.png)
```

## Nutzung im Frontend (optional)
Wenn ein Bild direkt in der App angezeigt werden soll:
1. Datei nach `frontend/public/assets/` kopieren
2. Dann über `/assets/<datei>.png` referenzieren

Beispiel:
- `frontend/public/assets/logo.png`
- URL in der App: `/assets/logo.png`
