// In dev, Vite proxies /api to Flask. Set VITE_API_URL to call another host directly.
const API = (import.meta.env && import.meta.env.VITE_API_URL) || "";

// The hosted single-file build sets this flag: there is no Python server behind it.
export const STANDALONE = typeof window !== "undefined" && window.__SER_STANDALONE__ === true;

export async function getStatus() {
  const r = await fetch(`${API}/api/status`);
  if (!r.ok) throw new Error(`Status check failed (${r.status})`);
  return r.json();
}

export async function predictEmotion(wavBlob) {
  const fd = new FormData();
  fd.append("audio", wavBlob, "clip.wav");
  let r;
  try {
    r = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
  } catch {
    throw new Error("Can’t reach the server. Make sure python app.py is running.");
  }
  let data = {};
  try {
    data = await r.json();
  } catch {
    /* non-JSON error page */
  }
  if (!r.ok) throw new Error(data.error || `Prediction failed (${r.status}).`);
  return data;
}
