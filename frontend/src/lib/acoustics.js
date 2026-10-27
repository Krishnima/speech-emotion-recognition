// Real acoustic measurements computed in the browser, plus a simple rule-based
// "demo" emotion guess used when the trained Python model isn't connected.
import { EMOTIONS } from "./emotions.js";

const FRAME = 1024;
const HOP = 512;

function fftMagnitude(input) {
  const n = input.length;
  const re = Float32Array.from(input);
  const im = new Float32Array(n);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const t = re[i]; re[i] = re[j]; re[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  const mag = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) mag[i] = Math.hypot(re[i], im[i]);
  return mag;
}

// Autocorrelation pitch estimate for one frame. Returns Hz or 0 when unvoiced.
function pitchOf(frame, sr) {
  const n = frame.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += frame[i];
  mean /= n;
  const x = new Float32Array(n);
  let energy = 0;
  for (let i = 0; i < n; i++) { x[i] = frame[i] - mean; energy += x[i] * x[i]; }
  if (energy < 1e-6) return 0;

  const minLag = Math.floor(sr / 400);
  const maxLag = Math.min(Math.floor(sr / 70), n - 2);
  const r = new Float32Array(maxLag + 2);
  let best = 0;
  for (let lag = minLag; lag <= maxLag + 1; lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += x[i] * x[i + lag];
    r[lag] = s / energy;
    if (lag <= maxLag && r[lag] > best) best = r[lag];
  }
  if (best < 0.35) return 0;
  // first strong peak (avoids octave-down errors)
  for (let lag = minLag + 1; lag <= maxLag; lag++) {
    if (r[lag] >= 0.85 * best && r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1]) {
      const a = r[lag - 1], b = r[lag], c = r[lag + 1];
      const denom = a - 2 * b + c;
      const shift = denom ? (0.5 * (a - c)) / denom : 0;
      return sr / (lag + shift);
    }
  }
  return 0;
}

const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};
const std = (arr) => {
  if (arr.length < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
};

export function measureAcoustics(samples, sr = 22050) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
  if (peak < 1e-4) throw new Error("silent");

  const frames = Math.max(1, Math.floor((samples.length - FRAME) / HOP) + 1);
  const rms = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    let s = 0;
    for (let i = 0; i < FRAME; i++) {
      const v = (samples[f * HOP + i] || 0) / peak;
      s += v * v;
    }
    rms[f] = Math.sqrt(s / FRAME);
  }
  const maxR = Math.max(...rms);
  const thr = Math.max(maxR * 0.12, 0.01);

  const hann = new Float32Array(FRAME).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FRAME - 1)));
  const pitches = [], centroids = [], activeDb = [];
  let active = 0, first = -1, last = -1;

  for (let f = 0; f < frames; f++) {
    if (rms[f] < thr) continue;
    active++;
    if (first < 0) first = f;
    last = f;
    activeDb.push(20 * Math.log10(rms[f] + 1e-9));
    const frame = samples.subarray(f * HOP, f * HOP + FRAME);
    if (frame.length < FRAME) continue;
    const p = pitchOf(frame, sr);
    if (p) pitches.push(p);
    const win = new Float32Array(FRAME);
    for (let i = 0; i < FRAME; i++) win[i] = frame[i] * hann[i];
    const mag = fftMagnitude(win);
    let num = 0, den = 0;
    for (let k = 1; k < mag.length; k++) { num += k * mag[k]; den += mag[k]; }
    if (den > 0) centroids.push(((num / den) * sr) / FRAME);
  }

  // syllable-like energy peaks -> speaking pace
  const env = Array.from(rms, (v, i) => (v + (rms[i - 1] ?? v) + (rms[i + 1] ?? v)) / 3);
  let peaks = 0;
  for (let i = 2; i < env.length - 2; i++) {
    if (env[i] > thr * 1.5 && env[i] >= env[i - 1] && env[i] >= env[i + 1] &&
        env[i] > env[i - 2] && env[i] > env[i + 2]) peaks++;
  }
  const speechSec = first >= 0 ? ((last - first + 1) * HOP) / sr : 0;

  const pitchHz = pitches.length >= 3 ? median(pitches) : null;
  const pitchVarSt = pitchHz ? std(pitches.map((p) => 12 * Math.log2(p / pitchHz))) : null;

  return {
    pitchHz,
    pitchVarSt,
    loudnessDb: activeDb.length ? activeDb.reduce((a, b) => a + b, 0) / activeDb.length : -60,
    brightnessHz: centroids.length ? centroids.reduce((a, b) => a + b, 0) / centroids.length : 0,
    pace: speechSec > 0.3 ? peaks / speechSec : 0,
    voicedPct: frames ? (active / frames) * 100 : 0,
  };
}

// z-scored features: [loudness, brightness, pitch variation, pace, pitch height]
const PROTOTYPES = {
  neutral:   [0.0, 0.0, -0.5, 0.0, 0.0],
  calm:      [-0.7, -0.5, -0.7, -0.6, -0.2],
  happy:     [0.6, 0.6, 0.9, 0.4, 0.6],
  sad:       [-1.0, -0.9, -0.6, -0.9, -0.3],
  angry:     [1.2, 1.1, 0.6, 0.5, 0.3],
  fearful:   [0.3, 0.3, 0.5, 0.9, 1.0],
  disgust:   [-0.2, -0.2, 0.1, -0.6, -0.5],
  surprised: [0.7, 0.5, 1.4, 0.2, 1.1],
};
const clamp = (v) => Math.max(-2.5, Math.min(2.5, v));

export function demoPredict(ac, duration) {
  const z = [
    clamp((ac.loudnessDb + 17) / 4),
    clamp((ac.brightnessHz - 1800) / 600),
    ac.pitchVarSt == null ? 0 : clamp((ac.pitchVarSt - 2.5) / 1.2),
    ac.pace ? clamp((ac.pace - 4) / 1.2) : 0,
    ac.pitchHz ? clamp(Math.log2(ac.pitchHz / 165) / 0.5) : 0,
  ];
  const scores = EMOTIONS.map((e) => -PROTOTYPES[e].reduce((s, p, i) => s + (z[i] - p) ** 2, 0) / 1.5);
  const max = Math.max(...scores);
  const exp = scores.map((s) => Math.exp(s - max));
  const total = exp.reduce((a, b) => a + b, 0);
  const probs = exp.map((v) => v / total);
  const order = EMOTIONS.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
  return {
    emotion: EMOTIONS[order[0]],
    confidence: probs[order[0]],
    probabilities: Object.fromEntries(order.map((i) => [EMOTIONS[i], probs[i]])),
    duration: Math.round(duration * 100) / 100,
    windows: 1,
    mode: "demo",
  };
}
