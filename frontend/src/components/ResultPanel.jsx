import { EMOTION_COLORS } from "../lib/emotions.js";

function Readout({ ac }) {
  if (!ac) return null;
  const items = [
    ["Pitch", ac.pitchHz ? `${Math.round(ac.pitchHz)} Hz` : "not detected"],
    ["Pitch variation", ac.pitchVarSt != null ? `${ac.pitchVarSt.toFixed(1)} semitones` : "not detected"],
    ["Brightness", ac.brightnessHz ? `${(ac.brightnessHz / 1000).toFixed(2)} kHz` : "not detected"],
    ["Pace", ac.pace ? `${ac.pace.toFixed(1)} syllables/s` : "not detected"],
  ];
  return (
    <div className="readout">
      <h3>Measured in this clip</h3>
      <dl>
        {items.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function ResultPanel({ result, status, trueLabel }) {
  const m = status.metrics || {};

  return (
    <section className="panel result" aria-labelledby="out-h" aria-live="polite">
      <div className="result-head">
        <h2 id="out-h">Detected emotion</h2>
        {result && (
          <span className={`chip${result.mode === "demo" ? " demo" : ""}`} data-testid="mode">
            {result.mode === "demo" ? "Demo estimate" : "Trained model"}
          </span>
        )}
      </div>

      {result ? (
        <div key={result._id}>
          <div className="big" data-testid="emotion">{result.emotion}</div>
          <p className="conf">
            {(result.confidence * 100).toFixed(1)}% confidence, from {result.duration} s of audio
            {result.windows > 1 ? ` averaged over ${result.windows} windows` : ""}
          </p>
          {trueLabel && (
            <p className={`truth${trueLabel === result.emotion ? " hit" : ""}`} data-testid="truth">
              {trueLabel === result.emotion
                ? `Matches the RAVDESS label in the file name (${trueLabel}).`
                : `The RAVDESS file name labels this clip as ${trueLabel}.`}
            </p>
          )}
          <ul className="bars">
            {Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([name, p]) => (
                <li key={name}>
                  <span className="name">{name}</span>
                  <span className="track">
                    <span
                      className="fill"
                      style={{ "--w": `${(p * 100).toFixed(1)}%`, background: EMOTION_COLORS[name] }}
                    />
                  </span>
                  <span className="pct">{(p * 100).toFixed(1)}%</span>
                </li>
              ))}
          </ul>
          {result.mode === "demo" && (
            <p className="note">
              Demo estimates come from fixed rules, not the trained network, so treat them as a rough guess.
            </p>
          )}
          <Readout ac={result.acoustics} />
        </div>
      ) : (
        <div className="placeholder">
          <p>Nothing analyzed yet.</p>
          <p>Record or upload a clip, then press Detect emotion to see the scores for all eight emotions.</p>
        </div>
      )}

      {m.test_accuracy != null && (
        <p className="meta">
          Model: {m.model_label || "Conv1D on 162 audio features"}, trained on {m.n_train_files} RAVDESS clips.
          Accuracy on {m.n_test_files} held-out clips: {(m.test_accuracy * 100).toFixed(1)}%.
        </p>
      )}
    </section>
  );
}
