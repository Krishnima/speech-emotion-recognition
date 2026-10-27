// Browser versions of the notebook's audio steps: STFT, the 5 features
// (ZCR, Chroma, MFCC, RMS, Mel) and the 4 augmentations (noise, stretch, shift, pitch).
// Defaults mirror librosa: n_fft 2048, hop 512, 128 mel bands, 20 MFCCs.

export const N_FFT = 2048;
export const HOP = 512;

// ---------- FFT ----------
export function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const nc = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nc;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

const winCache = {};
export function hann(n) {
  if (!winCache[n]) winCache[n] = Float32Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
  return winCache[n];
}

// Complex STFT with librosa-style centering (zero padding of n_fft/2 on both sides).
export function stft(y, nFft = N_FFT, hop = HOP) {
  const pad = nFft >> 1;
  const padded = new Float32Array(y.length + 2 * pad);
  padded.set(y, pad);
  const frames = 1 + Math.floor((padded.length - nFft) / hop);
  const bins = (nFft >> 1) + 1;
  const w = hann(nFft);
  const re = [], im = [];
  for (let f = 0; f < frames; f++) {
    const r = new Float32Array(nFft), i = new Float32Array(nFft);
    for (let k = 0; k < nFft; k++) r[k] = padded[f * hop + k] * w[k];
    fft(r, i);
    re.push(r.slice(0, bins));
    im.push(i.slice(0, bins));
  }
  return { re, im, frames, bins, nFft, hop };
}

export function magnitude(S) {
  return S.re.map((r, f) => {
    const m = new Float32Array(S.bins);
    for (let k = 0; k < S.bins; k++) m[k] = Math.hypot(r[k], S.im[f][k]);
    return m;
  });
}

export function istft(re, im, nFft, hop, length) {
  const frames = re.length;
  const pad = nFft >> 1;
  const total = nFft + hop * (frames - 1);
  const out = new Float32Array(total);
  const norm = new Float32Array(total);
  const w = hann(nFft);
  const bins = (nFft >> 1) + 1;
  for (let f = 0; f < frames; f++) {
    const r = new Float32Array(nFft), i = new Float32Array(nFft);
    for (let k = 0; k < bins; k++) { r[k] = re[f][k]; i[k] = im[f][k]; }
    for (let k = 1; k < nFft - bins + 1; k++) { r[nFft - k] = re[f][k]; i[nFft - k] = -im[f][k]; }
    fft(r, i, true);
    for (let k = 0; k < nFft; k++) {
      out[f * hop + k] += r[k] * w[k];
      norm[f * hop + k] += w[k] * w[k];
    }
  }
  for (let k = 0; k < total; k++) if (norm[k] > 1e-8) out[k] /= norm[k];
  const res = new Float32Array(length);
  res.set(out.subarray(pad, pad + length));
  return res;
}

// ---------- augmentations (same as the notebook) ----------
function gauss() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function addNoise(y, level = 0.035) {
  let peak = 0;
  for (const v of y) peak = Math.max(peak, Math.abs(v));
  const amp = level * peak;
  return y.map((v) => v + amp * gauss());
}

export function shift(y, samples) {
  const n = y.length, out = new Float32Array(n);
  const s = ((Math.round(samples) % n) + n) % n;
  for (let i = 0; i < n; i++) out[(i + s) % n] = y[i];
  return out;
}

// Phase vocoder, like librosa.effects.time_stretch: rate < 1 slows down, pitch unchanged.
export function timeStretch(y, rate) {
  if (Math.abs(rate - 1) < 1e-6) return Float32Array.from(y);
  const S = stft(y);
  const { bins, nFft, hop } = S;
  const zero = new Float32Array(bins);
  const re = [...S.re, zero], im = [...S.im, zero];
  const adv = Float32Array.from({ length: bins }, (_, k) => (Math.PI * hop * k) / (bins - 1));
  const phase = Float32Array.from({ length: bins }, (_, k) => Math.atan2(im[0][k], re[0][k]));
  const outRe = [], outIm = [];
  for (let t = 0; t < S.frames; t += rate) {
    const i = Math.floor(t), a = t - i;
    const r0 = re[i], i0 = im[i], r1 = re[i + 1], i1 = im[i + 1];
    const fr = new Float32Array(bins), fi = new Float32Array(bins);
    for (let k = 0; k < bins; k++) {
      const mag = (1 - a) * Math.hypot(r0[k], i0[k]) + a * Math.hypot(r1[k], i1[k]);
      fr[k] = mag * Math.cos(phase[k]);
      fi[k] = mag * Math.sin(phase[k]);
      let d = Math.atan2(i1[k], r1[k]) - Math.atan2(i0[k], r0[k]) - adv[k];
      d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      phase[k] += adv[k] + d;
    }
    outRe.push(fr);
    outIm.push(fi);
  }
  return istft(outRe, outIm, nFft, hop, Math.round(y.length / rate));
}

function resampleLinear(y, outLength) {
  const out = new Float32Array(outLength);
  const ratio = (y.length - 1) / Math.max(1, outLength - 1);
  for (let i = 0; i < outLength; i++) {
    const x = i * ratio, j = Math.floor(x), a = x - j;
    out[i] = (1 - a) * y[j] + a * (y[j + 1] ?? y[j]);
  }
  return out;
}

// Like librosa.effects.pitch_shift: stretch, then resample back to the original length.
export function pitchShift(y, nSteps) {
  if (Math.abs(nSteps) < 1e-6) return Float32Array.from(y);
  const rate = Math.pow(2, -nSteps / 12);
  return resampleLinear(timeStretch(y, rate), y.length);
}

// ---------- features ----------
const hzToMel = (f) => (f < 1000 ? f / (200 / 3) : 15 + Math.log(f / 1000) / (Math.log(6.4) / 27));
const melToHz = (m) => (m < 15 ? m * (200 / 3) : 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)));

const melCache = {};
function melFilters(sr, nFft = N_FFT, nMels = 128) {
  const key = `${sr}-${nFft}-${nMels}`;
  if (melCache[key]) return melCache[key];
  const bins = (nFft >> 1) + 1;
  const fftFreqs = Float32Array.from({ length: bins }, (_, k) => (k * sr) / nFft);
  const mMax = hzToMel(sr / 2);
  const pts = Array.from({ length: nMels + 2 }, (_, i) => melToHz((i * mMax) / (nMels + 1)));
  const filters = [];
  for (let m = 0; m < nMels; m++) {
    const [lo, c, hi] = [pts[m], pts[m + 1], pts[m + 2]];
    const enorm = 2 / (hi - lo); // slaney normalisation
    const w = [];
    for (let k = 0; k < bins; k++) {
      const f = fftFreqs[k];
      const v = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c)));
      if (v > 0) w.push([k, v * enorm]);
    }
    filters.push(w);
  }
  return (melCache[key] = filters);
}

function frameStat(y, fn) {
  // centered frames of 2048 with hop 512, like librosa's rms / zero_crossing_rate
  const pad = N_FFT >> 1;
  const p = new Float32Array(y.length + 2 * pad);
  p.set(y, pad);
  const frames = 1 + Math.floor((p.length - N_FFT) / HOP);
  return Float32Array.from({ length: frames }, (_, f) => fn(p.subarray(f * HOP, f * HOP + N_FFT)));
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const colMeans = (rows, n) => Float32Array.from({ length: n }, (_, k) => mean(rows.map((r) => r[k])));

export function extractFeatures(y, sr) {
  const S = stft(y);
  const mag = magnitude(S);
  const power = mag.map((m) => m.map((v) => v * v));

  const zcr = frameStat(y, (fr) => {
    let c = 0;
    for (let i = 1; i < fr.length; i++) if ((fr[i] >= 0) !== (fr[i - 1] >= 0)) c++;
    return c / fr.length;
  });
  const rms = frameStat(y, (fr) => Math.sqrt(fr.reduce((s, v) => s + v * v, 0) / fr.length));

  const fb = melFilters(sr);
  const mel = power.map((p) => Float32Array.from(fb, (w) => w.reduce((s, [k, v]) => s + v * p[k], 0)));

  // log-mel in dB (top_db 80) -> DCT-II (ortho) -> 20 MFCCs
  const melDb = mel.map((m) => m.map((v) => 10 * Math.log10(Math.max(v, 1e-10))));
  const maxDb = Math.max(...melDb.map((m) => Math.max(...m)));
  melDb.forEach((m) => m.forEach((v, i) => (m[i] = Math.max(v, maxDb - 80))));
  const nM = 128, nC = 20;
  const mfcc = melDb.map((m) =>
    Float32Array.from({ length: nC }, (_, c) => {
      let s = 0;
      for (let i = 0; i < nM; i++) s += m[i] * Math.cos((Math.PI * c * (2 * i + 1)) / (2 * nM));
      return s * Math.sqrt((c === 0 ? 1 : 2) / nM);
    })
  );

  // chroma: magnitude mapped to the 12 pitch classes (C..B), each frame scaled to max 1
  const chroma = mag.map((m) => {
    const c = new Float32Array(12);
    for (let k = 1; k < m.length; k++) {
      const f = (k * sr) / N_FFT;
      if (f < 55 || f > 5000) continue;
      const pc = ((Math.round(12 * Math.log2(f / 440) + 69) % 12) + 12) % 12;
      c[pc] += m[k];
    }
    const mx = Math.max(...c) || 1;
    return c.map((v) => v / mx);
  });

  const specDb = mag.map((m) => m.map((v) => 20 * Math.log10(Math.max(v, 1e-6))));

  const vector = {
    zcr: [mean(zcr)],
    chroma: Array.from(colMeans(chroma, 12)),
    mfcc: Array.from(colMeans(mfcc, 20)),
    rms: [mean(rms)],
    mel: Array.from(colMeans(mel, 128)),
  };
  return { zcr, rms, chroma, mfcc, melDb, specDb, vector, frames: S.frames };
}
