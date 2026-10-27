import { useEffect, useState } from "react";
import { RAVDESS_COUNTS, parseRavdess } from "../lib/ravdess.js";
import { EMOTION_COLORS } from "../lib/emotions.js";

const EXAMPLE = "03-01-06-01-02-01-12.wav";

export default function DatasetView({ clipName }) {
  const fromClip = parseRavdess(clipName) ? clipName : null;
  const [name, setName] = useState(fromClip || EXAMPLE);
  useEffect(() => {
    if (fromClip) setName(fromClip);
  }, [fromClip]);

  const parsed = parseRavdess(name);
  const max = Math.max(...Object.values(RAVDESS_COUNTS));
  const total = Object.values(RAVDESS_COUNTS).reduce((a, b) => a + b, 0);

  return (
    <div className="stack" id="panel-dataset">
      <section className="panel">
        <h2>RAVDESS speech</h2>
        <p className="measure lead">
          The Ryerson Audio-Visual Database of Emotional Speech and Song. This project uses only its speech
          recordings: 24 professional actors, 12 female and 12 male, each saying two fixed sentences in eight
          emotions. Every emotion except neutral is recorded at normal and strong intensity, twice each.
        </p>
        <dl className="facts">
          <div><dt>Speech files</dt><dd>{total.toLocaleString()}</dd></div>
          <div><dt>Actors</dt><dd>24</dd></div>
          <div><dt>Emotions</dt><dd>8</dd></div>
          <div><dt>Sentences</dt><dd>2</dd></div>
        </dl>
      </section>

      <div className="grid">
        <section className="panel">
          <h2>Count of emotions</h2>
          <p className="hint">Neutral has half as many files because it has no strong-intensity version.</p>
          <ul className="bars counts">
            {Object.entries(RAVDESS_COUNTS).map(([e, n]) => (
              <li key={e}>
                <span className="name">{e}</span>
                <span className="track">
                  <span className="fill" style={{ "--w": `${(n / max) * 100}%`, background: EMOTION_COLORS[e] }} />
                </span>
                <span className="pct">{n}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <h2>How this project prepares it</h2>
          <ol className="steps">
            <li>Each file is loaded at 22,050 Hz, skipping the first 0.6 s of silence and keeping 2.5 s.</li>
            <li>Files are split 80/20 by stratified sampling: 1,152 for training and 288 for testing.</li>
            <li>Every training file gets a noisy copy and a stretched, pitch-shifted copy, giving 3,456 training samples.</li>
            <li>Each sample becomes 162 averaged features, standardized before training.</li>
          </ol>
          <p className="hint">One change from the notebook: the split happens before augmentation, so altered copies of a
            test clip never leak into training and the test score stays honest.</p>
        </section>
      </div>

      <section className="panel">
        <h2>Read a RAVDESS file name</h2>
        <p className="hint">The label is stored in the file name. Upload a RAVDESS file on the Detect tab and it shows up here.</p>
        <label className="field">
          <span>File name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} spellCheck="false" data-testid="ravdess-input" />
        </label>
        {parsed ? (
          <div className="decode" data-testid="decode">
            {parsed.fields.map((f) => (
              <div key={f.key} className={`seg${f.meaning ? "" : " bad"}${f.key === "emotion" ? " emo" : ""}`}
                style={f.key === "emotion" && f.meaning ? { "--seg": EMOTION_COLORS[f.meaning] } : undefined}>
                <code>{f.code}</code>
                <span className="seg-name">{f.name}</span>
                <span className="seg-val">{f.meaning || "Unknown code"}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="err">Not a RAVDESS name. It needs seven two-digit codes separated by dashes, like {EXAMPLE}.</p>
        )}
      </section>
    </div>
  );
}
