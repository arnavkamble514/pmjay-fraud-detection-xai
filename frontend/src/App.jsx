import { useEffect, useState } from "react";
import { api } from "./api.js";
import Queue from "./components/Queue.jsx";
import Overview from "./components/Overview.jsx";
import ScoreForm from "./components/ScoreForm.jsx";
import About from "./components/About.jsx";

const TABS = [
  { id: "queue", label: "Review queue" },
  { id: "overview", label: "Overview" },
  { id: "score", label: "Score a claim" },
  { id: "about", label: "About the model" },
];

export default function App() {
  const [tab, setTab] = useState(() => {
    try { return localStorage.getItem("cg-tab") || "queue"; } catch { return "queue"; }
  });
  const [claims, setClaims] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    try { localStorage.setItem("cg-tab", tab); } catch { /* storage unavailable */ }
  }, [tab]);

  const loadClaims = () =>
    api.claims().then(setClaims).catch((e) => setError(e.message));

  useEffect(() => { loadClaims(); }, []);

  const markDecision = (claimId, decision) =>
    setClaims((cs) => cs && cs.map((c) => (c.claim_id === claimId ? { ...c, decision } : c)));

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-rule bg-sheet">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="6" fill="#1b2a38" />
              <circle cx="16" cy="16" r="9" fill="none" stroke="#b23a33" strokeWidth="3" />
            </svg>
            <span className="text-lg font-bold tracking-tight">ClaimGuard</span>
          </div>
          <nav className="-mb-3 flex gap-1 overflow-x-auto" aria-label="Sections">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`whitespace-nowrap border-b-2 px-3 pb-3 pt-1 text-[15px] font-medium transition-colors ${
                  tab === t.id
                    ? "border-pen text-ink"
                    : "border-transparent text-ink-soft hover:text-ink"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <p className="ml-auto text-sm text-ink-soft">
            Demonstration on synthetic PM-JAY-style claims
          </p>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 lg:overflow-hidden py-4 sm:px-6">
        {error ? (
          <div className="mx-auto mt-16 max-w-lg rounded-lg border border-seal/30 bg-sheet p-6">
            <p className="font-semibold text-seal">The claims could not be loaded.</p>
            <p className="mt-2 text-ink-soft">{error}</p>
            <button onClick={() => { setError(null); loadClaims(); }}
              className="mt-4 rounded-md bg-pen px-4 py-2 font-medium text-white">
              Try again
            </button>
          </div>
        ) : tab === "queue" ? (
          <Queue claims={claims} onDecision={markDecision} />
        ) : tab === "overview" ? (
          <Overview />
        ) : tab === "score" ? (
          <ScoreForm />
        ) : (
          <About />
        )}
      </main>
    </div>
  );
}
