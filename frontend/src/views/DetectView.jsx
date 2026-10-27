import Waveform from "../components/Waveform.jsx";
import DropZone from "../components/DropZone.jsx";
import ResultPanel from "../components/ResultPanel.jsx";

export default function DetectView({ clip, rec, maxSeconds, preparing, busy, error, canAnalyze, emo,
  onToggleRecord, onFile, onAnalyze, result, status }) {
  let recTitle = "Start recording";
  let recSub = "Uses your microphone";
  if (rec.recording) {
    recTitle = "Recording, press to stop";
    recSub = `${rec.elapsed.toFixed(1)} s of ${maxSeconds}`;
  } else if (preparing) {
    recTitle = "Preparing audio";
    recSub = "Converting to WAV";
  } else if (clip) {
    recTitle = clip.label;
    recSub = `${clip.duration.toFixed(1)} s ready to analyze`;
  }

  return (
    <div className="grid">
      <section className="panel" aria-labelledby="in-h">
        <h2 id="in-h">Your voice</h2>
        <p className="hint">Speak for 2 to 5 seconds. Recording stops by itself after {maxSeconds}.</p>

        <Waveform samples={clip?.samples} analyser={rec.analyser} color={emo} />

        <div className="controls">
          <button
            className={`rec${rec.recording ? " on" : ""}`}
            onClick={onToggleRecord}
            disabled={preparing || busy}
            aria-label={rec.recording ? "Stop recording" : "Start recording"}
            data-testid="record"
          >
            <span className="dot" />
          </button>
          <div className="rec-meta">
            <strong>{recTitle}</strong>
            <span data-testid="rec-sub">{recSub}</span>
          </div>
        </div>

        <DropZone onFile={onFile} disabled={rec.recording || busy} />

        {clip && <audio key={clip.url} src={clip.url} controls />}

        <button className="btn" onClick={onAnalyze} disabled={!canAnalyze} data-testid="analyze">
          {busy ? (<><span className="spinner" />Detecting emotion</>) : "Detect emotion"}
        </button>
        <p className="err" role="alert">{error}</p>
      </section>

      <ResultPanel result={result} status={status} trueLabel={clip?.ravdess?.emotion} />
    </div>
  );
}
