import { useEffect, useState } from "react";
import { api, BAND, DECISION, inr } from "../api.js";

export default function ClaimDetail({ claimId, showTruth, onDecision }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState(null);
  const [zoom, setZoom] = useState(false);
  const [note, setNote] = useState("");
  const [reviewer, setReviewer] = useState(() => {
    try { return localStorage.getItem("cg-reviewer") || ""; } catch { return ""; }
  });
  const [saving, setSaving] = useState(false);
  const [justStamped, setJustStamped] = useState(false);

  useEffect(() => {
    api.claim(claimId).then(setD).catch((e) => setError(e.message));
  }, [claimId]);

  if (error) return <p className="p-8 text-seal">{error}</p>;
  if (!d) return <p className="p-8 text-ink-soft">Loading {claimId}…</p>;

  const b = BAND[d.band];
  const last = d.history.length ? d.history[d.history.length - 1] : null;
  const maxImpact = Math.max(...d.reasons.map((r) => r.impact), 0.001);

  const decide = async (decision) => {
    setSaving(true);
    try {
      const name = reviewer.trim() || "Reviewer";
      try { localStorage.setItem("cg-reviewer", reviewer.trim()); } catch { /* ignore */ }
      const history = await api.review(d.claim_id, { decision, note, reviewer: name });
      setD({ ...d, history });
      setNote("");
      setJustStamped(true);
      onDecision(d.claim_id, decision);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <article className="grid gap-6 p-4 sm:p-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      {/* Left: the claim and the decision */}
      <div className="space-y-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">{d.claim_id}</h1>
          <p className="text-ink-soft">
            {d.procedure} ({d.diagnosis_code}) at {d.hospital}, {d.state}
          </p>
        </header>

        <div className="flex items-end gap-5">
          <div>
            <div className={`text-6xl font-semibold leading-none tracking-tight ${b.text}`}>
              {Math.round(d.score * 100)}
            </div>
            <div className="mt-1 text-sm text-ink-soft">risk score out of 100</div>
          </div>
          <div className={`rounded-md px-3 py-2 ${b.soft}`}>
            <div className={`font-semibold ${b.text}`}>{b.label}</div>
            <div className="text-sm">{d.action}</div>
          </div>
        </div>

        {showTruth && (
          <p className={`rounded-md border px-3 py-2 text-sm ${d.is_fraud ? "border-seal/30 text-seal" : "border-rule text-ink-soft"}`}>
            True label (synthetic data): <strong>{d.label}</strong>
            {d.is_fraud && d.band === "low" && ". The model missed this one."}
            {!d.is_fraud && d.band !== "low" && ". This is a false alarm."}
          </p>
        )}

        <section aria-labelledby="why">
          <h2 id="why" className="mb-2 font-semibold">Why this score</h2>
          <ul className="space-y-2.5">
            {d.reasons.map((r) => (
              <li key={r.feature} className="grid grid-cols-[1.25rem_1fr] gap-2">
                <span aria-hidden="true" className={r.direction === "raises" ? "text-seal" : "text-leaf"}>
                  {r.direction === "raises" ? "▲" : "▼"}
                </span>
                <div>
                  <div className="text-[15px]">
                    <span className="sr-only">{r.direction === "raises" ? "Raises risk: " : "Lowers risk: "}</span>
                    {r.detail}
                  </div>
                  <div className="mt-1 h-1 rounded-full bg-paper">
                    <div className={`h-1 rounded-full ${r.direction === "raises" ? "bg-seal/70" : "bg-leaf/70"}`}
                      style={{ width: `${(r.impact / maxImpact) * 100}%` }} />
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="decide" className="rounded-lg border border-rule p-4">
          <h2 id="decide" className="font-semibold">Your decision</h2>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)}
              placeholder="Your name" aria-label="Reviewer name"
              className="rounded-md border border-rule px-3 py-1.5 text-sm" />
            <input value={note} onChange={(e) => setNote(e.target.value)}
              placeholder="Note (optional)" aria-label="Note"
              className="rounded-md border border-rule px-3 py-1.5 text-sm" />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button disabled={saving} onClick={() => decide("cleared")}
              className="rounded-md border border-leaf px-3 py-2 text-sm font-semibold text-leaf hover:bg-leaf/10 disabled:opacity-50">
              Clear claim
            </button>
            <button disabled={saving} onClick={() => decide("fraud_confirmed")}
              className="rounded-md bg-seal px-3 py-2 text-sm font-semibold text-white hover:bg-seal/90 disabled:opacity-50">
              Confirm fraud and hold payment
            </button>
            <button disabled={saving} onClick={() => decide("escalated")}
              className="rounded-md border border-pen px-3 py-2 text-sm font-semibold text-pen hover:bg-pen/10 disabled:opacity-50">
              Escalate to senior auditor
            </button>
          </div>
          {d.history.length > 0 && (
            <ol className="mt-4 space-y-1.5 border-t border-rule pt-3 text-sm">
              {[...d.history].reverse().map((h, i) => (
                <li key={i}>
                  <span className={`font-semibold ${DECISION[h.decision].color}`}>{DECISION[h.decision].label}</span>
                  {" by "}{h.reviewer}, {new Date(h.at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })}
                  {h.note && <span className="text-ink-soft">. “{h.note}”</span>}
                </li>
              ))}
            </ol>
          )}
        </section>

        <section aria-labelledby="facts">
          <h2 id="facts" className="mb-2 font-semibold">Claim against the record</h2>
          <dl className="divide-y divide-rule border-y border-rule text-[15px]">
            <Fact label="Claimed amount" value={`${inr(d.claimed_amount)} against a package rate of ${inr(d.package_rate)} (${d.amount_ratio}×)`} />
            <Fact label="Length of stay" value={`${d.length_of_stay} days; typical is ${d.typical_los} (${d.los_ratio}×)`} />
            <Fact label="Amount read from bill" value={
              d.ocr_amount_found === 0 || d.ocr_amount_mismatch_pct == null
                ? "No amount could be read"
                : d.ocr_amount_mismatch_pct < 0.01
                  ? "Matches the claim"
                  : `Differs from the claim by ${(d.ocr_amount_mismatch_pct * 100).toFixed(0)}%`} />
            <Fact label="Noise check" value={d.noise_block_min_z == null ? "Not run" : `Quietest patch scores ${d.noise_block_min_z}`} />
            <Fact label="Tamper CNN" value={d.tamper_score == null ? "Not run" : `${d.tamper_score} (adds no measurable signal on these bills)`} />
            <Fact label="Patient age" value={d.patient_age} />
          </dl>
        </section>
      </div>

      {/* Right: the bill itself, where the stamp lands */}
      <figure className="self-start xl:sticky xl:top-0">
        <button onClick={() => setZoom(true)} className="relative block w-full overflow-hidden rounded-md border border-rule bg-paper"
          aria-label="Enlarge bill image">
          <img src={d.image_url} alt={`Submitted bill for ${d.claim_id}`} className="block w-full" />
          {last && (
            <div key={d.history.length} className={`stamp ${DECISION[last.decision].color} ${justStamped ? "landing" : ""}`}>
              {DECISION[last.decision].stamp}
              <small>{last.reviewer}, {new Date(last.at).toLocaleDateString("en-IN")}</small>
            </div>
          )}
        </button>
        <figcaption className="mt-2 text-sm text-ink-soft">Bill as submitted. Click to enlarge.</figcaption>
      </figure>

      {zoom && (
        <div role="dialog" aria-modal="true" aria-label="Bill image" onClick={() => setZoom(false)}
          onKeyDown={(e) => e.key === "Escape" && setZoom(false)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink/80 p-4">
          <img src={d.image_url} alt={`Submitted bill for ${d.claim_id}`} className="max-h-full max-w-full rounded bg-white" />
          <button autoFocus onClick={() => setZoom(false)}
            className="absolute right-4 top-4 rounded-md bg-sheet px-3 py-1.5 text-sm font-semibold">Close</button>
        </div>
      )}
    </article>
  );
}

function Fact({ label, value }) {
  return (
    <div className="grid grid-cols-[9.5rem_1fr] gap-3 py-2">
      <dt className="text-ink-soft">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
