const ABLATION = [
  { step: "Structured claim data only", auc: 0.822, gain: null },
  { step: "Add OCR on the bill", auc: 0.887, gain: "+0.065" },
  { step: "Add the tamper CNN", auc: 0.885, gain: "−0.002" },
  { step: "Add noise forensics", auc: 0.916, gain: "+0.031" },
];

export default function About() {
  return (
    <div className="pb-8 lg:h-full lg:overflow-y-auto">
      <article className="mx-auto max-w-3xl space-y-6 rounded-lg border border-rule bg-sheet p-5 text-[16px] leading-relaxed sm:p-8">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">How ClaimGuard scores a claim</h1>
          <p className="mt-2 text-ink-soft">
            Three kinds of evidence go into one model. The claim record shows whether the amount and stay fit the
            procedure. OCR reads the bill and checks it against the claim. Noise forensics looks for a patch of the
            bill that lacks the scan noise around it, as a pasted edit would.
          </p>
        </header>

        <section>
          <h2 className="font-semibold">What each kind of evidence adds</h2>
          <p className="text-ink-soft">Five-fold cross-validated ROC AUC on 3,000 synthetic claims, adding one source at a time.</p>
          <table className="mt-3 w-full text-left text-[15px]">
            <thead>
              <tr className="border-b border-rule text-ink-soft">
                <th className="py-2 font-medium">Evidence used</th>
                <th className="py-2 text-right font-medium">AUC</th>
                <th className="py-2 text-right font-medium">Change</th>
              </tr>
            </thead>
            <tbody>
              {ABLATION.map((r) => (
                <tr key={r.step} className="border-b border-rule">
                  <td className="py-2">{r.step}</td>
                  <td className="py-2 text-right font-semibold">{r.auc.toFixed(3)}</td>
                  <td className="py-2 text-right text-ink-soft">{r.gain ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3">
            On bills where a field other than the amount was edited, noise forensics raised the share caught from
            10% to 55%. The tamper CNN, trained on edited photographs, added nothing on scanned bills.
          </p>
        </section>

        <section>
          <h2 className="font-semibold">What this does not show</h2>
          <ul className="mt-2 list-disc space-y-1.5 pl-5">
            <li>Every claim and bill here is synthetic. No real PM-JAY data was used, so these numbers say nothing yet about real claims.</li>
            <li>The forged bills and the forensics check were built in the same project. Careful forgeries that imitate scan noise are still mostly missed.</li>
            <li>Real fraud is rarer than the 19% in this data, so a real queue would hold a smaller share of true fraud.</li>
            <li>The score ranks claims for review. It is not a finding of fraud, and nothing is rejected without a person.</li>
          </ul>
        </section>

        <p className="border-t border-rule pt-4 text-sm text-ink-soft">
          Academic prototype. Not affiliated with the National Health Authority or the Government of India.
        </p>
      </article>
    </div>
  );
}
