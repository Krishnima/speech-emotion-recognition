import { useEffect, useMemo, useState } from "react";
import Waveform from "../components/Waveform.jsx";
import Heatmap from "../components/Heatmap.jsx";
import LinePlot from "../components/LinePlot.jsx";
import PlayButton from "../components/PlayButton.jsx";
import { extractFeatures, stft, magnitude, addNoise, timeStretch, shift, pitchShift } from "../lib/dsp.js";

const NOTEBOOK = { noise: 0.035, rate: 0.8, shiftSec: 0.12, steps: 0.7 };

function normalizeRows(frames) {
  const n = frames[0]?.length || 0;
  const lo = new Float32Array(n).fill(Infinity), hi = new Float32Array(n).fill(-Infinity);
  for (const f of frames) for (let r = 0; r < n; r++) { lo[r] = Math.min(lo[r], f[r]); hi[r] = Math.max(hi[r], f[r]); }
  return frames.map((f) => f.map((v, r) => (v - lo[r]) / (hi[r] - lo[r] || 1)));
}

function specOf(y) {
  return magnitude(stft(y)).map((m) => m.map((v) => 20 * Math.log10(Math.max(v, 1e-6))));
}

function FeatureVector({ vector }) {
  const groups = [
    ["ZCR", vector.zcr, "#E0A912"],
    ["Chroma", vector.chroma, "#3E9EA0"],
    ["MFCC", vector.mfcc, "#8E5BC9"],
    ["RMS", vector.rms, "#D9473B"],
    ["Mel", vector.mel, "#4C6EDB"],
  ];
  const W = 162 * 6, H = 132, mid = 60;
  let x = 0;
  const bars = [], labels = [];
  for (const [name, vals, color] of groups) {
    const mx = Math.max(...vals.map(Math.abs)) || 1;
    const start = x;
    vals.forEach((v) => {
      const h = (Math.abs(v) / mx) * (mid - 6);
      bars.push(<rect key={x} x={x * 6 + 1} y={v >= 0 ? mid - h : mid} width={4} height={Math.max(1, h)} fill={color} />);
      x++;
    });
    labels.push({ name, count: vals.length, start, end: x, color });
    bars.push(<rect key={`seg-${name}`} x={start * 6} y={H - 6} width={(x - start) * 6 - 2} height={5} rx={2} fill={color} />);
  }
  return (
    <div className="vector">
      <div className="vector-scroll">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="The 162 averaged feature values">
          <line x1="0" x2={W} y1={mid} y2={mid} stroke="var(--line)" />
          {bars}
        </svg>
      </div>
      <div className="vector-legend">
        {labels.map((l) => (
          <span key={l.name}>
            <i style={{ background: l.color }} />{l.name} <em>{l.count}</em>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function ExploreView({ clip, sr, onGoDetect, modelType }) {
  const [params, setParams] = useState(NOTEBOOK);
  const [debounced, setDebounced] = useState(NOTEBOOK);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(params), 250);
    return () => clearTimeout(t);
  }, [params]);

  const feats = useMemo(() => (clip ? extractFeatures(clip.samples, sr) : null), [clip, sr]);
  const mfccView = useMemo(() => (feats ? normalizeRows(feats.mfcc) : null), [feats]);

  const versions = useMemo(() => {
    if (!clip) return [];
    const y = clip.samples;
    const p = debounced;
    const list = [
      { id: "original", name: "Original", y, note: "The clip as recorded." },
      { id: "noise", name: "Noise", y: addNoise(y, p.noise), note: `Gaussian noise at ${(p.noise * 100).toFixed(1)}% of the peak.` },
      { id: "stretch", name: "Stretch", y: timeStretch(y, p.rate), note: `Speed × ${p.rate.toFixed(2)}, pitch unchanged.` },
      { id: "shift", name: "Shift", y: shift(y, p.shiftSec * sr), note: `Rolled ${p.shiftSec >= 0 ? "forward" : "back"} ${Math.abs(p.shiftSec).toFixed(2)} s.` },
      { id: "pitch", name: "Pitch", y: pitchShift(y, p.steps), note: `${p.steps >= 0 ? "Up" : "Down"} ${Math.abs(p.steps).toFixed(1)} semitones, same length.` },
    ];
    return list.map((v) => ({ ...v, spec: specOf(v.y) }));
  }, [clip, debounced, sr]);

  if (!clip) {
    return (
      <section className="panel empty-state" id="panel-explore">
        <h2>Explore your clip</h2>
        <p>Record or upload a clip first. This tab then shows its waveform and spectrogram, what each augmentation
          does to it, and the five features the model is trained on.</p>
        <button className="btn inline" onClick={onGoDetect}>Record or upload a clip</button>
      </section>
    );
  }

  const set = (k) => (e) => setParams((p) => ({ ...p, [k]: Number(e.target.value) }));
  const nyq = (sr / 2000).toFixed(1);

  return (
    <div className="stack" id="panel-explore">
      <section className="panel">
        <div className="section-head">
          <div>
            <h2>Waveform and spectrogram</h2>
            <p className="hint">{clip.label}, {clip.duration.toFixed(1)} s. Brighter colors mean more energy at that frequency.</p>
          </div>
          <PlayButton samples={clip.samples} sr={sr} label="original clip" />
        </div>
        <Waveform samples={clip.samples} color="var(--emo)" />
        <div className="axis-wrap">
          <Heatmap data={feats.specDb} height={200} range={[-60, 30]} label="Spectrogram of the clip" />
          <div className="axis-y"><span>{nyq} kHz</span><span>0</span></div>
        </div>
        <div className="axis-x"><span>0 s</span><span>{clip.duration.toFixed(1)} s</span></div>
      </section>

      <section className="panel">
        <h2>Data augmentation</h2>
        <p className="hint measure">
          Training adds two extra copies of every clip: one with noise, and one stretched and then pitch-shifted.
          That triples the training set. The notebook also defines a shift, but doesn’t use it in training.
        </p>
        <div className="sliders">
          <label><span className="sl-head">Noise level<output>{(params.noise * 100).toFixed(1)}%</output></span>
            <input type="range" min="0" max="0.1" step="0.005" value={params.noise} onChange={set("noise")} /></label>
          <label><span className="sl-head">Stretch rate<output>{params.rate.toFixed(2)}</output></span>
            <input type="range" min="0.6" max="1.4" step="0.05" value={params.rate} onChange={set("rate")} /></label>
          <label><span className="sl-head">Shift<output>{params.shiftSec.toFixed(2)} s</output></span>
            <input type="range" min="-0.23" max="0.23" step="0.01" value={params.shiftSec} onChange={set("shiftSec")} /></label>
          <label><span className="sl-head">Pitch<output>{params.steps.toFixed(1)} st</output></span>
            <input type="range" min="-4" max="4" step="0.1" value={params.steps} onChange={set("steps")} /></label>
          <button className="link" onClick={() => setParams(NOTEBOOK)}>Reset to the notebook’s values</button>
        </div>
        <div className="aug-grid">
          {versions.map((v) => (
            <article key={v.id} className="aug" data-testid={`aug-${v.id}`}>
              <div className="aug-head">
                <h3>{v.name}</h3>
                <PlayButton samples={v.y} sr={sr} label={v.name} />
              </div>
              <Heatmap data={v.spec} height={90} range={[-60, 30]} label={`${v.name} spectrogram`} />
              <p>{v.note} {(v.y.length / sr).toFixed(2)} s.</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Feature extraction</h2>
        <p className="hint measure">
          These are the five features from the notebook, computed over 93 ms frames. Each one is then averaged over
          time, so the whole clip becomes 162 numbers.
          {modelType === "ssl" && " They feed the baseline CNN. The current model listens with WavLM instead, a network pretrained on large amounts of unlabeled speech (see the Model tab)."}
        </p>
        <div className="feat-grid">
          <article className="feat">
            <h3>Zero crossing rate <em>1 value</em></h3>
            <p>How often the wave crosses zero. Higher for hissy, noisy sounds.</p>
            <LinePlot values={Array.from(feats.zcr)} color="#E0A912" label="Zero crossing rate over time" />
          </article>
          <article className="feat">
            <h3>RMS energy <em>1 value</em></h3>
            <p>Loudness of each frame.</p>
            <LinePlot values={Array.from(feats.rms)} color="#D9473B" label="RMS energy over time" />
          </article>
          <article className="feat">
            <h3>Chroma <em>12 values</em></h3>
            <p>Energy in each of the 12 pitch classes, C at the bottom to B at the top.</p>
            <Heatmap data={feats.chroma} rows={12} height={110} range={[0, 1]} label="Chroma over time" />
          </article>
          <article className="feat">
            <h3>MFCC <em>20 values</em></h3>
            <p>The overall shape of the spectrum, which captures voice timbre. Each row is scaled to its own range.</p>
            <Heatmap data={mfccView} rows={20} height={110} range={[0, 1]} label="MFCCs over time" />
          </article>
          <article className="feat wide">
            <h3>Mel spectrogram <em>128 values</em></h3>
            <p>Energy in 128 bands spaced the way human hearing spaces pitch.</p>
            <Heatmap data={feats.melDb} rows={128} height={150} label="Mel spectrogram" />
          </article>
        </div>
      </section>

      <section className="panel">
        <h2>The 162 numbers the model sees</h2>
        <p className="hint measure">
          Each group is scaled separately here so you can see its shape. Before training, every value is
          standardized with a StandardScaler.
        </p>
        <FeatureVector vector={feats.vector} />
      </section>
    </div>
  );
}
