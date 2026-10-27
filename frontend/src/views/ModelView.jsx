import { modelLayers } from "../lib/modelInfo.js";
import { EMOTIONS } from "../lib/emotions.js";

function Curves({ title, train, test, percent }) {
  const W = 420, H = 200, P = { l: 40, r: 10, t: 10, b: 26 };
  const all = [...train, ...test];
  let lo = Math.min(...all), hi = Math.max(...all);
  if (percent) { lo = Math.max(0, Math.floor(lo * 10) / 10); hi = Math.min(1, Math.ceil(hi * 10) / 10); }
  const span = hi - lo || 1;
  const n = Math.max(train.length, 2);
  const px = (i) => P.l + (i / (n - 1)) * (W - P.l - P.r);
  const py = (v) => P.t + (1 - (v - lo) / span) * (H - P.t - P.b);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join("");
  const ticks = [lo, lo + span / 2, hi];
  return (
    <figure className="curves">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} per epoch, train and test`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={P.l} x2={W - P.r} y1={py(t)} y2={py(t)} stroke="var(--line)" />
            <text x={P.l - 6} y={py(t) + 4} textAnchor="end">{percent ? `${Math.round(t * 100)}%` : t.toFixed(2)}</text>
          </g>
        ))}
        <text x={P.l} y={H - 6}>epoch 1</text>
        <text x={W - P.r} y={H - 6} textAnchor="end">epoch {train.length}</text>
        <path d={path(train)} fill="none" stroke="var(--muted)" strokeWidth="2" />
        <path d={path(test)} fill="none" stroke="var(--accent)" strokeWidth="2.5" />
      </svg>
      <div className="legend"><span className="k train" />Train <span className="k test" />Test</div>
    </figure>
  );
}

function Confusion({ matrix, labels }) {
  return (
    <div className="cm-wrap">
      <table className="cm">
        <thead>
          <tr><th className="corner">Actual ↓ Predicted →</th>{labels.map((l) => <th key={l} scope="col">{l}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={labels[i]}>
                <th scope="row">{labels[i]}</th>
                {row.map((v, j) => {
                  const share = v / total;
                  return (
                    <td key={j} className={i === j ? "diag" : ""}
                      style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(share * 85)}%, transparent)`,
                        color: share > 0.45 ? "#fff" : undefined }}
                      title={`${v} of ${total} ${labels[i]} clips predicted as ${labels[j]}`}>
                      {v}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Compare({ baseline, current, label }) {
  const rows = [
    ["Baseline CNN (notebook features)", baseline, "var(--muted)"],
    [label, current, "var(--accent)"],
  ];
  return (
    <ul className="bars compare" data-testid="compare">
      {rows.map(([name, v, color]) => (
        <li key={name}>
          <span className="name">{name}</span>
          <span className="track">
            <span className="fill" style={{ "--w": `${(v * 100).toFixed(1)}%`, background: color }} />
          </span>
          <span className="pct">{(v * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

function LayerChart({ acc, chosen }) {
  const W = 640, H = 170, P = { l: 34, r: 6, t: 8, b: 24 };
  const n = acc.length;
  const bw = (W - P.l - P.r) / n;
  const top = Math.min(1, Math.ceil(Math.max(...acc) * 10) / 10);
  const py = (v) => P.t + (1 - v / top) * (H - P.t - P.b);
  return (
    <figure className="curves layers-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Validation accuracy of each WavLM layer">
        {[0, top / 2, top].map((t) => (
          <g key={t}>
            <line x1={P.l} x2={W - P.r} y1={py(t)} y2={py(t)} stroke="var(--line)" />
            <text x={P.l - 6} y={py(t) + 4} textAnchor="end">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {acc.map((v, i) => (
          <rect key={i} x={P.l + i * bw + 1} y={py(v)} width={Math.max(1, bw - 2)} height={H - P.b - py(v)}
            rx="2" fill={chosen.includes(i) ? "var(--accent)" : "var(--line)"}>
            <title>{`Layer ${i}: ${(v * 100).toFixed(1)}%`}</title>
          </rect>
        ))}
        <text x={P.l} y={H - 6}>layer 0 (input)</text>
        <text x={W - P.r} y={H - 6} textAnchor="end">layer {n - 1}</text>
      </svg>
      <div className="legend"><span className="k test" />Layers the classifier uses</div>
    </figure>
  );
}

function SslModel({ m, labels }) {
  const a = m.architecture || {};
  const t = m.training || {};
  return (
    <div className="stack" id="panel-model">
      <section className="panel">
        <h2>Pipeline</h2>
        <ol className="pipeline">
          <li><strong>Load</strong><span>16 kHz, silence trimmed</span></li>
          <li><strong>Augment</strong><span>noise, stretch + pitch</span></li>
          <li><strong>{a.backbone_label || "WavLM"}</strong><span>pretrained, frozen</span></li>
          <li><strong>Pool</strong><span>mean + std per layer</span></li>
          <li><strong>Classify</strong><span>logistic regression</span></li>
          <li><strong>Softmax</strong><span>8 emotions</span></li>
        </ol>
      </section>

      <div className="grid">
        <section className="panel">
          <h2>Model</h2>
          <p className="hint">A large speech model listens; a small classifier learns emotions from what it hears.</p>
          <dl className="setup">
            <div><dt>Speech model</dt><dd>{a.backbone_label} <span className="muted">({a.backbone})</span></dd></div>
            {a.backbone_params ? <div><dt>Size</dt><dd>{Math.round(a.backbone_params / 1e6)} million parameters, kept frozen</dd></div> : null}
            <div><dt>Layers used</dt><dd data-testid="ssl-layers">{(a.layers || []).join(", ")} of {a.backbone_layers}</dd></div>
            <div><dt>Numbers per clip</dt><dd>{a.classifier_inputs?.toLocaleString()} ({t.pooling})</dd></div>
            <div><dt>Classifier</dt><dd>{t.classifier}, C = {t.C}</dd></div>
            <div><dt>Data split</dt><dd>{m.n_train_files} train, {m.n_val_files} validation, {m.n_test_files} test clips</dd></div>
          </dl>
        </section>

        <section className="panel">
          <h2>Accuracy on the same {m.n_test_files} test clips</h2>
          <p className="hint">Both models were scored on identical held-out clips that neither saw during training.</p>
          {m.baseline_accuracy != null ? (
            <Compare baseline={m.baseline_accuracy} current={m.test_accuracy} label={a.backbone_label || "WavLM"} />
          ) : (
            <p className="score" data-testid="score">Test accuracy <strong>{(m.test_accuracy * 100).toFixed(1)}%</strong></p>
          )}
          <p className="hint">Validation accuracy (used for tuning): {(m.validation_accuracy * 100).toFixed(1)}%. Random guessing: 12.5%.</p>
        </section>
      </div>

      {m.layer_val_accuracy && (
        <section className="panel">
          <h2>Which layers know about emotion</h2>
          <p className="hint measure">
            A quick classifier was trained on each layer alone and scored on the validation clips. The best layers were
            combined for the final model.
          </p>
          <LayerChart acc={m.layer_val_accuracy} chosen={a.layers || []} />
        </section>
      )}

      <section className="panel">
        <h2>Confusion matrix</h2>
        <p className="hint">Rows are the true emotion, columns are what the model predicted. Darker cells hold a bigger share of that row.</p>
        <Confusion matrix={m.confusion_matrix} labels={labels} />
      </section>
      <Report m={m} labels={labels} />
    </div>
  );
}

function Report({ m, labels }) {
  if (!m.report) return null;
  return (
    <section className="panel">
      <h2>Classification report</h2>
      <div className="table-wrap">
        <table className="layers report">
          <thead><tr><th>Emotion</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Clips</th></tr></thead>
          <tbody>
            {labels.map((l) => (
              <tr key={l}>
                <td className="cap">{l}</td>
                <td className="num">{m.report[l].precision.toFixed(2)}</td>
                <td className="num">{m.report[l].recall.toFixed(2)}</td>
                <td className="num">{m.report[l].f1.toFixed(2)}</td>
                <td className="num">{m.report[l].support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function ModelView({ status }) {
  const { layers, total } = modelLayers();
  const m = status.metrics || {};
  if (m.model_type === "ssl" && m.confusion_matrix) return <SslModel m={m} labels={m.labels || EMOTIONS} />;
  const hasResults = m.history && m.confusion_matrix;
  const labels = m.labels || EMOTIONS;

  return (
    <div className="stack" id="panel-model">
      <section className="panel">
        <h2>Pipeline</h2>
        <ol className="pipeline">
          <li><strong>Load</strong><span>2.5 s at 22,050 Hz</span></li>
          <li><strong>Augment</strong><span>noise, stretch + pitch</span></li>
          <li><strong>Extract</strong><span>162 features</span></li>
          <li><strong>Scale</strong><span>StandardScaler</span></li>
          <li><strong>Conv1D</strong><span>4 conv blocks</span></li>
          <li><strong>Softmax</strong><span>8 emotions</span></li>
        </ol>
      </section>

      <div className="grid">
        <section className="panel">
          <h2>Network</h2>
          <p className="hint">The notebook’s 1D-CNN. Output shapes and parameters are worked out for 162 inputs.</p>
          <div className="table-wrap">
            <table className="layers">
              <thead><tr><th>Layer</th><th>Output</th><th className="num">Params</th></tr></thead>
              <tbody>
                {layers.map((l, i) => (
                  <tr key={i}>
                    <td>{l.name}</td>
                    <td className="mono">({l.shape.join(", ")})</td>
                    <td className="num">{l.params ? l.params.toLocaleString() : "–"}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot><tr><td colSpan="2">Total trainable parameters</td><td className="num" data-testid="params">{total.toLocaleString()}</td></tr></tfoot>
            </table>
          </div>
        </section>

        <section className="panel">
          <h2>Training setup</h2>
          <dl className="setup">
            <div><dt>Optimizer</dt><dd>Adam</dd></div>
            <div><dt>Loss</dt><dd>Categorical cross-entropy</dd></div>
            <div><dt>Batch size</dt><dd>64</dd></div>
            <div><dt>Epochs</dt><dd>Up to 60, stops early if test accuracy stalls for 12</dd></div>
            <div><dt>Learning rate</dt><dd>Cut to 40% when test loss stalls for 3 epochs</dd></div>
            <div><dt>Labels</dt><dd>One-hot encoded, 8 classes</dd></div>
          </dl>
          {m.test_accuracy != null && (
            <p className="score" data-testid="score">
              Test accuracy <strong>{(m.test_accuracy * 100).toFixed(1)}%</strong> on {m.n_test_files} clips
              after {m.epochs_run} epochs.
            </p>
          )}
        </section>
      </div>

      {hasResults ? (
        <>
          <section className="panel">
            <h2>Training curves</h2>
            <div className="curves-grid">
              <Curves title="Accuracy" train={m.history.accuracy} test={m.history.val_accuracy} percent />
              <Curves title="Loss" train={m.history.loss} test={m.history.val_loss} />
            </div>
          </section>
          <section className="panel">
            <h2>Confusion matrix</h2>
            <p className="hint">Rows are the true emotion, columns are what the model predicted. Darker cells hold a bigger share of that row.</p>
            <Confusion matrix={m.confusion_matrix} labels={labels} />
          </section>
          <Report m={m} labels={labels} />
        </>
      ) : (
        <section className="panel empty-state" data-testid="no-results">
          <h2>Training results</h2>
          <p>
            Accuracy and loss curves, the confusion matrix and the classification report appear here after you run{" "}
            <code>python train.py --data data/ravdess</code> and start <code>python app.py</code>.
            {status.state === "standalone" ? " This hosted demo has no trained model, so this section stays empty." : ""}
          </p>
        </section>
      )}
    </div>
  );
}
