"use client";

import { useState } from "react";

/**
 * Typen für die API-Antwort.
 * In dieser vereinfachten Version gibt es nur noch Satzdaten.
 * Zusätzliche Verknüpfungslogik wurde bewusst entfernt.
 */
type Fallacy = {
  label: string;
  score: number;
};

type RoleInfo = {
  label: string;
  score: number;
};

type SentenceItem = {
  id: number;
  text: string;
  role?: RoleInfo;
  fallacies?: Fallacy[];
};

type AnalyzeResponse = {
  sentences?: SentenceItem[];
};

/**
 * Rollen-Badge-Farben.
 */
const roleBadgeClass = (label: string) => {
  switch ((label ?? "other").toLowerCase()) {
    case "premise":
      return "bg-blue-300/20 text-blue-200 border border-blue-400/30";
    case "conclusion":
      return "bg-green-300/20 text-green-200 border border-green-400/30";
    case "claim":
      return "bg-purple-300/20 text-purple-200 border border-purple-400/30";
    case "objection":
      return "bg-orange-300/20 text-orange-200 border border-orange-400/30";
    default:
      return "bg-gray-300/10 text-gray-200 border border-gray-400/20";
  }
};

/**
 * Ampelfarbe für den Rollen-Score.
 */
const confidenceBarClass = (score: number) => {
  if (score >= 0.75) return "bg-green-400";
  if (score >= 0.55) return "bg-yellow-400";
  return "bg-red-400";
};

/**
 * Farbe für Fehlschluss-Badges.
 */
const fallacyBadgeClass = (score: number) => {
  if (score >= 0.75) return "bg-red-400/20 text-red-200 border border-red-400/40";
  if (score >= 0.55) return "bg-yellow-400/15 text-yellow-200 border border-yellow-400/30";
  return "bg-gray-300/10 text-gray-200 border border-gray-400/20";
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  /**
   * Schickt den Text an das Backend.
   */
  const analyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const msg = await response.text();
        throw new Error(`Backend-Fehler ${response.status}: ${msg}`);
      }

      const data = (await response.json()) as AnalyzeResponse;
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unbekannter Fehler");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8 max-w-3xl mx-auto bg-black text-white">
      <h1 className="text-3xl font-bold mb-6">Philosophical Argument Analyzer</h1>

      <textarea
        className="w-full h-48 p-4 border border-gray-400 rounded-md bg-white text-black"
        placeholder="Füge hier deinen Text ein ..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        className="mt-4 px-6 py-2 bg-white text-black rounded-md disabled:opacity-60"
        onClick={analyze}
        disabled={loading || text.trim().length === 0}
      >
        {loading ? "Analysiere ..." : "Analysieren"}
      </button>

      {error && (
        <div className="mt-6 p-4 border border-red-400 rounded-md bg-gray-800">
          <h2 className="font-semibold mb-2 text-red-300">Fehler:</h2>
          <p className="whitespace-pre-wrap text-red-200">{error}</p>
        </div>
      )}

      {result && (
        <div className="mt-6 p-4 border border-gray-700 rounded-md bg-gray-800">
          <h2 className="font-semibold mb-3">Ergebnis (Satz für Satz):</h2>

          <div className="space-y-3">
            {(result.sentences ?? []).map((sentence) => {
              const roleLabel = sentence.role?.label ?? "other";
              const roleScore = Number(sentence.role?.score ?? 0);

              // Relevantere Fallacies zuerst anzeigen.
              const filteredFallacies = [...(sentence.fallacies ?? [])]
                .filter((f) => Number(f.score ?? 0) >= 0.55)
                .sort((a, b) => Number(b.score ?? 0) - Number(a.score ?? 0));

              return (
                <div
                  key={sentence.id}
                  className="p-3 rounded-md border border-gray-700 bg-gray-900"
                >
                  <div className="flex items-center gap-3 mb-2">
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold tracking-wide ${roleBadgeClass(
                        roleLabel
                      )}`}
                    >
                      {roleLabel.toUpperCase()}
                    </span>

                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-300">Score: {roleScore.toFixed(2)}</span>
                      <div className="w-28 h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${confidenceBarClass(roleScore)}`}
                          style={{ width: `${Math.round(roleScore * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <p className="whitespace-pre-wrap">{sentence.text}</p>

                  <div className="mt-2 text-xs text-gray-400">
                    Fehlschlüsse gefunden: {sentence.fallacies?.length ?? 0}
                  </div>

                  {filteredFallacies.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs text-gray-300 mb-2">Fehlschlüsse:</div>
                      <div className="flex flex-wrap gap-2">
                        {filteredFallacies.map((fallacy, idx) => (
                          <span
                            key={idx}
                            className={`px-3 py-1 rounded-full text-xs font-semibold ${fallacyBadgeClass(
                              Number(fallacy.score ?? 0)
                            )}`}
                            title={`Score: ${fallacy.score}`}
                          >
                            {String(fallacy.label).replaceAll("_", " ").toUpperCase()} •{" "}
                            {Number(fallacy.score ?? 0).toFixed(2)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-gray-300">
              Debug: Rohes JSON anzeigen
            </summary>
            <pre className="mt-2 text-xs whitespace-pre-wrap">
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </main>
  );
}
