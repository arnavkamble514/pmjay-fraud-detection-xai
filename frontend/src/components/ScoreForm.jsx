import { useEffect, useState } from "react";
import { api, BAND, inr } from "../api.js";

export default function ScoreForm() {
  const [opts, setOpts] = useState(null);
  const [form, setForm] = useState({
    procedure: "", state: "", provider_id: "", patient_age: 45, claimed_amount: "", length_of_stay: "",
  });
  const [bill, setBill] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.options().then((o) => {
      setOpts(o);
      const p = o.procedures.find((x) => x.procedure === "Laparoscopic cholecystectomy") || o.procedures[0];
      setForm((f) => ({ ...f, procedure: p.procedure, state: o.states[0], provider_id: o.providers[0],
        claimed_amount: Math.round(p.package_rate * 1.86), length_of_stay: p.typical_los * 3 }));
    }).catch((e) => setError(e.message));
  }, []);

  if (!opts) return <p className="mt-16 text-center text-ink-soft">{error || "Loading…"}</p>;

  const proc = opts.procedures.find((p) => p.procedure === form.procedure);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setError(null); setResult(null);
    const fd = new FormData();
    Object.entries({ ...form, diagnosis_code: proc.diagnosis_code }).forEach(([k, v]) => fd.append(k, v));
    if (bill) fd.append("bill", bill);
    try { setResult(await api.score(fd)); } catch (err) { setError(err.message); } finally { setBusy(false); }
  };

  const b = result && BAND[result.risk_band];

  return (
    <div className="pb-8 lg:h-full lg:overflow-y-auto">
      <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-[1fr_1fr]">
        <form onSubmit={submit} className="space-y-4 rounded-lg border border-rule bg-sheet p-5 sm:p-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Score a new claim</h1>
            <p className="mt-1 text-ink-soft">Enter the claim as submitted. Add the bill image for the full check.</p>
          </div>
          <Field label="Procedure">
            <select value={form.procedure} onChange={set("procedure")} className={inputCls}>
              {opts.procedures.map((p) => <option key={p.procedure}>{p.procedure}</option>)}
            </select>
            {proc && <span className="mt-1 block text-sm text-ink-soft">
              Package rate {inr(proc.package_rate)}, typical stay {proc.typical_los} days, code {proc.diagnosis_code}
            </span>}
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Claimed amount (₹)">
              <input type="number" min="0" required value={form.claimed_amount} onChange={set("claimed_amount")} className={inputCls} />
            </Field>
            <Field label="Length of stay (days)">
              <input type="number" min="1" required value={form.length_of_stay} onChange={set("length_of_stay")} className={inputCls} />
            </Field>
            <Field label="Patient age">
              <input type="number" min="0" max="120" required value={form.patient_age} onChange={set("patient_age")} className={inputCls} />
            </Field>
            <Field label="State">
              <select value={form.state} onChange={set("state")} className={inputCls}>
                {opts.states.map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Hospital (provider ID)">
              <select value={form.provider_id} onChange={set("provider_id")} className={inputCls}>
                {opts.providers.map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Bill image (optional)">
            <input type="file" accept="image/png,image/jpeg" onChange={(e) => setBill(e.target.files[0] || null)}
              className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-paper file:px-3 file:py-1.5 file:font-medium" />
          </Field>
          <button disabled={busy} className="rounded-md bg-pen px-4 py-2 font-semibold text-white hover:bg-pen/90 disabled:opacity-60">
            {busy ? (bill ? "Reading the bill…" : "Scoring…") : "Score claim"}
          </button>
        </form>

        <section className="rounded-lg border border-rule bg-sheet p-5 sm:p-6" aria-live="polite">
          {error && <p className="text-seal">{error}</p>}
          {!result && !error && (
            <p className="text-ink-soft">The score, the reasons and what happens next will appear here.</p>
          )}
          {result && (
            <div className="space-y-5">
              <div className="flex items-end gap-5">
                <div className={`text-6xl font-semibold leading-none tracking-tight ${b.text}`}>{Math.round(result.score * 100)}</div>
                <div className={`rounded-md px-3 py-2 ${b.soft}`}>
                  <div className={`font-semibold ${b.text}`}>{b.label}</div>
                  <div className="text-sm">{result.action}</div>
                </div>
              </div>
              <ul className="space-y-2">
                {result.top_reasons.map((r) => (
                  <li key={r.feature} className="grid grid-cols-[1.25rem_1fr] gap-2 text-[15px]">
                    <span aria-hidden="true" className={r.direction === "raises risk" ? "text-seal" : "text-leaf"}>
                      {r.direction === "raises risk" ? "▲" : "▼"}
                    </span>
                    <span>{r.detail}</span>
                  </li>
                ))}
              </ul>
              <p className="border-t border-rule pt-3 text-sm text-ink-soft">
                Scored with the {result.model_used}.
                {!result.image_checked && " Without a bill, edited documents can't be detected."}
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

const inputCls = "mt-1 w-full rounded-md border border-rule bg-sheet px-3 py-1.5";

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="text-sm font-medium">{label}</span>
      {children}
    </label>
  );
}
