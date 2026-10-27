import { useCallback, useEffect, useRef, useState } from "react";
import StatusBanner from "./components/StatusBanner.jsx";
import Tabs from "./components/Tabs.jsx";
import DetectView from "./views/DetectView.jsx";
import ExploreView from "./views/ExploreView.jsx";
import DatasetView from "./views/DatasetView.jsx";
import ModelView from "./views/ModelView.jsx";
import { useRecorder } from "./hooks/useRecorder.js";
import { blobToWav, TARGET_SR } from "./lib/audio.js";
import { getStatus, predictEmotion, STANDALONE } from "./lib/api.js";
import { measureAcoustics, demoPredict } from "./lib/acoustics.js";
import { colorFor } from "./lib/emotions.js";
import { parseRavdess } from "./lib/ravdess.js";

const MAX_SECONDS = 10;
const TABS = [
  { id: "detect", label: "Detect" },
  { id: "explore", label: "Explore" },
  { id: "dataset", label: "Dataset" },
  { id: "model", label: "Model" },
];
const tabFromHash = () => {
  const h = (typeof location !== "undefined" ? location.hash : "").replace("#", "");
  return TABS.some((t) => t.id === h) ? h : "detect";
};

export default function App() {
  const [tab, setTab] = useState(tabFromHash);
  const [status, setStatus] = useState({ state: STANDALONE ? "standalone" : "loading" });
  const [clip, setClip] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const urlRef = useRef(null);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const goTab = useCallback((id) => {
    setTab(id);
    try {
      history.replaceState(null, "", `#${id}`);
    } catch {
      /* sandboxed viewers may block URL changes */
    }
    window.scrollTo?.({ top: 0 });
  }, []);

  const checkStatus = useCallback(async () => {
    if (STANDALONE) return;
    setStatus({ state: "loading" });
    try {
      const s = await getStatus();
      setStatus(
        s.ready
          ? { state: "ready", metrics: s.metrics || {} }
          : { state: s.reason === "error" ? "error" : "untrained", message: s.message }
      );
    } catch {
      setStatus({ state: "offline" });
    }
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  useEffect(() => () => urlRef.current && URL.revokeObjectURL(urlRef.current), []);

  const loadClip = useCallback(async (blob, label) => {
    setError("");
    setResult(null);
    setPreparing(true);
    try {
      let c;
      try {
        c = await blobToWav(blob);
      } catch {
        setError("This file couldn’t be decoded. Try a WAV or MP3 file.");
        return;
      }
      if (c.duration < 0.5) {
        setError("That clip is under half a second. Record a full sentence.");
        return;
      }
      let acoustics;
      try {
        acoustics = measureAcoustics(c.samples, TARGET_SR);
      } catch {
        setError("The clip is silent. Check your microphone and try again.");
        return;
      }
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = URL.createObjectURL(c.wav);
      const rav = parseRavdess(label);
      setClip({ ...c, acoustics, label, url: urlRef.current, ravdess: rav?.valid ? rav : null });
    } finally {
      setPreparing(false);
    }
  }, []);

  const rec = useRecorder({
    maxSeconds: MAX_SECONDS,
    onComplete: (blob) => loadClip(blob, "Recording ready"),
  });

  async function toggleRecord() {
    setError("");
    if (rec.recording) {
      rec.stop();
      return;
    }
    try {
      setResult(null);
      await rec.start();
    } catch (e) {
      setError(e.message);
    }
  }

  async function analyze() {
    if (!clip) return;
    setBusy(true);
    setError("");
    try {
      let r;
      if (status.state === "ready") {
        r = { ...(await predictEmotion(clip.wav)), mode: "model" };
      } else {
        await new Promise((res) => setTimeout(res, 250));
        r = demoPredict(clip.acoustics, clip.duration);
      }
      setResult({ ...r, acoustics: clip.acoustics, _id: Date.now() });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const emo = result ? colorFor(result.emotion) : "var(--accent)";
  const canAnalyze = clip && !busy && !preparing && !rec.recording && status.state !== "loading";

  return (
    <div className="app" style={{ "--emo": emo }}>
      <div className="wrap">
        <header>
          <h1>How does this voice feel?</h1>
          <p className="sub">
            Speech emotion recognition on the RAVDESS dataset: record or upload a clip, see how the notebook’s
            pipeline processes it, and get scores for eight emotions.
          </p>
          <StatusBanner status={status} onRetry={STANDALONE ? null : checkStatus} />
        </header>

        <Tabs tabs={TABS} active={tab} onChange={goTab} />

        <main role="tabpanel" aria-labelledby={`tab-${tab}`} className="tabpanel">
          {tab === "detect" && (
            <DetectView
              clip={clip} rec={rec} maxSeconds={MAX_SECONDS} preparing={preparing} busy={busy} error={error}
              canAnalyze={canAnalyze} emo={emo} onToggleRecord={toggleRecord}
              onFile={(f) => loadClip(f, f.name)} onAnalyze={analyze} result={result} status={status}
            />
          )}
          {tab === "explore" && (
            <ExploreView clip={clip} sr={TARGET_SR} onGoDetect={() => goTab("detect")} modelType={status.metrics?.model_type} />
          )}
          {tab === "dataset" && <DatasetView clipName={clip?.ravdess ? clip.label : null} />}
          {tab === "model" && <ModelView status={status} />}
        </main>

        <p className="tips">
          For best results, speak expressively in a quiet room. The trained model learned from 24 professional actors
          speaking two English sentences, so everyday speech, other languages and noisy rooms will be less accurate.
        </p>
      </div>
    </div>
  );
}
