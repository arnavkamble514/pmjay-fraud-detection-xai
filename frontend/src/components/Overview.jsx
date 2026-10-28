import { useEffect, useMemo, useState } from "react";
import { api, BAND } from "../api.js";

const GROUPS = ["Upcoding", "Phantom billing", "Impossible length of stay",
  "Bill edited: amount", "Bill edited: other field"];

function groupOf(label) {
  if (label === "Genuine") return "Genuine";
  if (label.startsWith("Bill edited: amount")) return "Bill edited: amount";
  if (label.startsWith("Bill edited:")) return "Bill edited: other field";
  return label;
}

export default function Overview() {
  const [s, setS] = useState(null);
  const [error, setError] = useState(null);
  const [t, setT] = useState(0.25);

  useEffect(() => { api.summary().then(setS).catch((e) => setError(e.message)); }, []);

  const stats = useMemo(() => {
    if (!s) return null;
    let tp = 0, fp = 0, fn = 0;
    const byGroup = {};
    for (const x of s.scores) {
      const flagged = x.score >= t;
      const g = groupOf(x.label);
      byGroup[g] ??= { n: 0, caught: 0 };
      byGroup[g].n++;
      if (flagged) byGroup[g].caught++;
      if (flagged && x.is_fraud) tp++;
      else if (flagged) fp++;
      else if (x.is_fraud) fn++;
    }
    return {
      flagged: tp + fp, precision: tp + fp ? tp / (tp + fp) : 0,
      recall: tp + fn ? tp / (tp + fn) : 0, byGroup,
    };
  }, [s, t]);

  if (error) return <p className="mt-16 text-center text-seal">{error}</p>;
  if (!s) return <p className="mt-16 text-center text-ink-soft">Loading…</p>;

  const pct = (v) => `${Math.round(v * 100)}%`;
  const genuine = stats.byGroup.Genuine || { n: 0, caught: 0 };

  return (
    <div className="pb-8 lg:h-full lg:overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6">
        <section className="rounded-lg border border-rule bg-sheet p-5 sm:p-6">
          <h1 className="text-2xl font-semibold tracking-tight">Where to draw the review line</h1>
          <p className="mt-1 max-w-2xl text-ink-soft">
            Every claim above the line goes to a person. Move it to see how many claims reviewers must check
            and how much of the fraud they would catch. Scores come from models that never saw the claim they scored.
          </p>

          <div className="mt-6 grid items-end gap-6 md:grid-cols-[1fr_auto]">
            <label className="block">
              <span className="text-sm text-ink-soft">Review everything scoring at least</span>
              <div className="flex items-center gap-4">
                <span className="w-16 text-4xl font-semibold tracking-tight">{Math.round(t * 100)}</span>
                <input type="range" min="0.05" max="0.95" step="0.05" value={t}
                  onChange={(e) => setT(Number(e.target.value))} className="w-full" aria-label="Review threshold" />
              </div>
            </label>
            <dl className="grid grid-cols-3 gap-6 text-right">
              <Stat label="Claims to review" value={stats.flagged.toLocaleString("en-IN")} sub={`of ${s.n_claims.toLocaleString("en-IN")}`} />
              <Stat label="Fraud caught" value={pct(stats.recall)} sub="recall" />
              <Stat label="Flags that are fraud" value={pct(stats.precision)} sub="precision" />
            </dl>
          </div>

          <Histogram scores={s.scores} t={t} />
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          <div className="rounded-lg border border-rule bg-sheet p-5">
            <h2 className="font-semibold">Caught at this line, by kind of fraud</h2>
            <ul className="mt-3 space-y-3">
              {GROUPS.filter((g) => stats.byGroup[g]).map((g) => {
                const v = stats.byGroup[g];
                return (
                  <li key={g}>
                    <div className="flex justify-between text-[15px]">
                      <span>{g}</span>
                      <span className="text-ink-soft">{v.caught} of {v.n}</span>
                    </div>
                    <div className="mt-1 h-2 rounded-full bg-paper">
                      <div className="h-2 rounded-full bg-seal/80" style={{ width: pct(v.caught / v.n) }} />
                    </div>
                  </li>
                );
              })}
              <li>
                <div className="flex justify-between text-[15px]">
                  <span>Genuine claims flagged (false alarms)</span>
                  <span className="text-ink-soft">{genuine.caught} of {genuine.n}</span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-paper">
                  <div className="h-2 rounded-full bg-ink-soft/60" style={{ width: pct(genuine.n ? genuine.caught / genuine.n : 0) }} />
                </div>
              </li>
            </ul>
          </div>

          <div className="rounded-lg border border-rule bg-sheet p-5">
            <h2 className="font-semibold">What happens to each claim</h2>
            <p className="mt-1 text-sm text-ink-soft">The system routes every claim by its band. Nothing is rejected without a person.</p>
            <ul className="mt-3 divide-y divide-rule">
              {[...s.bands].reverse().map((band, i, arr) => {
                const lower = i === arr.length - 1 ? 0 : arr[i + 1].upper;
                const b = BAND[band.band];
                return (
                  <li key={band.band} className="grid grid-cols-[6.5rem_1fr_auto] gap-3 py-2.5">
                    <span className={`font-semibold ${b.text}`}>{b.label}</span>
                    <span className="text-[15px]">{band.action}
                      <span className="block text-sm text-ink-soft">score {Math.round(lower * 100)}–{Math.min(100, Math.round(band.upper * 100))}</span>
                    </span>
                    <span className="text-right font-semibold">{band.count.toLocaleString("en-IN")}</span>
                  </li>
                );
              })}
            </ul>
            <p className="mt-3 text-sm text-ink-soft">
              Ranking quality on all {s.n_claims.toLocaleString("en-IN")} claims: ROC AUC {s.roc_auc}, PR-AUC {s.pr_auc}.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div>
      <dt className="text-sm text-ink-soft">{label}</dt>
      <dd className="text-3xl font-semibold tracking-tight">{value}</dd>
      <dd className="text-xs text-ink-soft">{sub}</dd>
    </div>
  );
}

/* Fraud above the axis, genuine below it, so the overlap the review line
   has to cut through is visible at a glance. Each side has its own
   square-root scale, so the large genuine pile does not flatten the fraud bars. */
function Histogram({ scores, t }) {
  const BINS = 20, W = 900, H = 220, mid = 118, pad = 4;
  const fraud = Array(BINS).fill(0), genuine = Array(BINS).fill(0);
  for (const x of scores) {
    const i = Math.min(BINS - 1, Math.floor(x.score * BINS));
    (x.is_fraud ? fraud : genuine)[i]++;
  }
  const maxF = Math.sqrt(Math.max(...fraud, 1)), maxG = Math.sqrt(Math.max(...genuine, 1));
  const bw = W / BINS;
  const hf = (n) => (Math.sqrt(n) / maxF) * (mid - 26);
  const hg = (n) => (Math.sqrt(n) / maxG) * (H - mid - 22);
  const x = t * W;
  return (
    <figure className="mt-6">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label="Distribution of risk scores: fraudulent claims above the axis, genuine claims below">
        {fraud.map((n, i) => n > 0 && (
          <rect key={`f${i}`} x={i * bw + pad} width={bw - pad * 2} y={mid - hf(n)} height={hf(n)}
            fill={i / BINS >= t - 1e-9 ? "#b23a33" : "#b23a3366"} rx="2" />
        ))}
        {genuine.map((n, i) => n > 0 && (
          <rect key={`g${i}`} x={i * bw + pad} width={bw - pad * 2} y={mid + 2} height={hg(n)}
            fill={i / BINS >= t - 1e-9 ? "#586674" : "#58667455"} rx="2" />
        ))}
        <line x1="0" x2={W} y1={mid + 1} y2={mid + 1} stroke="#d3d9d6" />
        <line x1={x} x2={x} y1="4" y2={H - 14} stroke="#2450a6" strokeWidth="2.5" />
        <text x={Math.min(x + 6, W - 120)} y="16" fill="#2450a6" fontSize="14" fontWeight="600">review line</text>
        <text x="4" y="16" fill="#b23a33" fontSize="14">fraudulent claims</text>
        <text x="4" y={H - 2} fill="#586674" fontSize="14">genuine claims</text>
        <text x={W - 4} y={H - 2} fill="#586674" fontSize="14" textAnchor="end">score 0 to 100</text>
      </svg>
      <figcaption className="mt-2 text-sm text-ink-soft">
        Bars show how many claims fall in each score range. Fraud and genuine are drawn on separate scales.
      </figcaption>
    </figure>
  );
}
