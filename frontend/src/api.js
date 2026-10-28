async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Can't reach the ClaimGuard server. Start it with: python api.py (in the stage2 folder).");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  claims: () => request("/api/claims"),
  claim: (id) => request(`/api/claims/${encodeURIComponent(id)}`),
  review: (id, body) =>
    request(`/api/claims/${encodeURIComponent(id)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  summary: () => request("/api/summary"),
  options: () => request("/api/options"),
  score: (formData) => request("/api/score", { method: "POST", body: formData }),
};

export const inr = (n) =>
  n == null ? "–" : "₹" + Math.round(n).toLocaleString("en-IN");

export const BAND = {
  high: { label: "High risk", text: "text-seal", bg: "bg-seal", soft: "bg-seal/10" },
  medium: { label: "Medium risk", text: "text-ochre", bg: "bg-ochre", soft: "bg-ochre/10" },
  low: { label: "Low risk", text: "text-leaf", bg: "bg-leaf", soft: "bg-leaf/10" },
};

export const DECISION = {
  cleared: { label: "Cleared", stamp: "Cleared", note: "Approved for payment", color: "text-leaf" },
  fraud_confirmed: { label: "Fraud confirmed", stamp: "Payment held", note: "Fraud confirmed", color: "text-seal" },
  escalated: { label: "Escalated", stamp: "Escalated", note: "Sent to senior auditor", color: "text-pen" },
};
