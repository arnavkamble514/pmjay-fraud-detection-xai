import { useEffect, useMemo, useState } from "react";
import { BAND, DECISION, inr } from "../api.js";
import ClaimDetail from "./ClaimDetail.jsx";

const PAGE = 60;
const BAND_FILTERS = [
  { id: "flagged", label: "Needs review" },
  { id: "high", label: "High" },
  { id: "medium", label: "Medium" },
  { id: "low", label: "Low" },
  { id: "all", label: "All" },
];

export default function Queue({ claims, onDecision }) {
  const [bandFilter, setBandFilter] = useState("flagged");
  const [hideDone, setHideDone] = useState(false);
  const [query, setQuery] = useState("");
  const [showTruth, setShowTruth] = useState(false);
  const [limit, setLimit] = useState(PAGE);
  const [selected, setSelected] = useState(null);

  const counts = useMemo(() => {
    const c = { all: 0, high: 0, medium: 0, low: 0, flagged: 0 };
    (claims || []).forEach((x) => {
      c.all++; c[x.band]++;
      if (x.band !== "low") c.flagged++;
    });
    return c;
  }, [claims]);

  const rows = useMemo(() => {
    if (!claims) return [];
    const q = query.trim().toLowerCase();
    return claims
      .filter((c) =>
        bandFilter === "all" ? true
          : bandFilter === "flagged" ? c.band !== "low"
          : c.band === bandFilter)
      .filter((c) => !hideDone || !c.decision)
      .filter((c) => !q || [c.claim_id, c.hospital, c.provider_id, c.procedure, c.state]
        .some((v) => String(v).toLowerCase().includes(q)))
      .sort((a, b) => b.score - a.score);
  }, [claims, bandFilter, hideDone, query]);

  useEffect(() => { setLimit(PAGE); }, [bandFilter, hideDone, query]);
  useEffect(() => {
    if (!selected && rows.length) setSelected(rows[0].claim_id);
  }, [rows, selected]);

  if (!claims) {
    return <p className="mt-16 text-center text-ink-soft">Loading claims and scores. The first start can take about a minute.</p>;
  }

  const reviewed = claims.filter((c) => c.decision).length;

  return (
    <div className="grid gap-4 lg:h-full lg:grid-cols-[minmax(380px,520px)_1fr]">
      <section className="flex max-h-[70vh] min-h-0 flex-col rounded-lg border border-rule bg-sheet lg:max-h-none" aria-label="Claims ranked by risk">
        <div className="space-y-3 border-b border-rule p-3">
          <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by risk">
            {BAND_FILTERS.map((f) => (
              <button key={f.id} onClick={() => setBandFilter(f.id)} aria-pressed={bandFilter === f.id}
                className={`rounded-md px-2.5 py-1 text-sm font-medium ${
                  bandFilter === f.id ? "bg-ink text-white" : "text-ink-soft hover:bg-paper hover:text-ink"}`}>
                {f.label} <span className="opacity-70">{counts[f.id]}</span>
              </button>
            ))}
          </div>
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)}
            placeholder="Search claim ID, hospital, procedure or state"
            aria-label="Search claims"
            className="w-full rounded-md border border-rule bg-paper/50 px-3 py-1.5 text-sm placeholder:text-ink-soft/70" />
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-ink-soft">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={hideDone} onChange={(e) => setHideDone(e.target.checked)} />
              Hide reviewed ({reviewed})
            </label>
            <label className="flex items-center gap-2" title="The data is synthetic, so the true answer is known">
              <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
              Show true labels
            </label>
          </div>
        </div>

        <ol className="min-h-0 flex-1 overflow-y-auto">
          {rows.slice(0, limit).map((c) => (
            <QueueRow key={c.claim_id} c={c} active={c.claim_id === selected}
              showTruth={showTruth} onClick={() => setSelected(c.claim_id)} />
          ))}
          {rows.length === 0 && (
            <li className="p-6 text-center text-ink-soft">No claims match. Clear the search or pick another filter.</li>
          )}
          {rows.length > limit && (
            <li className="p-3 text-center">
              <button onClick={() => setLimit((l) => l + PAGE)}
                className="rounded-md px-3 py-1.5 text-sm font-medium text-pen hover:bg-paper">
                Show {Math.min(PAGE, rows.length - limit)} more of {rows.length - limit}
              </button>
            </li>
          )}
        </ol>
      </section>

      <section className="min-h-0 rounded-lg border border-rule bg-sheet lg:overflow-y-auto" aria-label="Claim detail">
        {selected
          ? <ClaimDetail key={selected} claimId={selected} showTruth={showTruth} onDecision={onDecision} />
          : <p className="p-8 text-ink-soft">Pick a claim from the queue.</p>}
      </section>
    </div>
  );
}

function QueueRow({ c, active, showTruth, onClick }) {
  const b = BAND[c.band];
  const ratio = c.claimed_amount / c.package_rate;
  const missed = showTruth && c.is_fraud && c.band === "low";
  const falseAlarm = showTruth && !c.is_fraud && c.band !== "low";
  return (
    <li>
      <button onClick={onClick} aria-current={active ? "true" : undefined}
        className={`grid w-full grid-cols-[3.2rem_1fr_auto] items-center gap-3 border-b border-rule px-3 py-2.5 text-left ${
          active ? "bg-pen/[0.07]" : "hover:bg-paper/60"}`}>
        <div className="text-right">
          <div className={`text-lg font-semibold leading-none ${b.text}`}>{Math.round(c.score * 100)}</div>
          <div className="mt-1.5 h-1 rounded-full bg-rule">
            <div className={`h-1 rounded-full ${b.bg}`} style={{ width: `${Math.max(4, c.score * 100)}%` }} />
          </div>
        </div>
        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="font-semibold">{c.claim_id}</span>
            <span className="truncate text-sm text-ink-soft">{c.procedure}</span>
          </div>
          <div className="truncate text-sm text-ink-soft">
            {c.hospital.replace(" Multispeciality Hospital", "")}, {c.state}. {inr(c.claimed_amount)} ({ratio.toFixed(2)}× package)
          </div>
          {showTruth && (
            <div className={`text-xs ${c.is_fraud ? "text-seal" : "text-ink-soft"}`}>
              {c.label}{missed ? ", missed" : ""}{falseAlarm ? ", false alarm" : ""}
            </div>
          )}
        </div>
        <div className="text-right text-xs">
          {c.decision
            ? <span className={`font-semibold ${DECISION[c.decision].color}`}>{DECISION[c.decision].label}</span>
            : <span className={b.text}>{b.label}</span>}
        </div>
      </button>
    </li>
  );
}
