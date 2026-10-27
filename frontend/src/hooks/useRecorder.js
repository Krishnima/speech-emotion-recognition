import { useCallback, useEffect, useRef, useState } from "react";

function pickMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return types.find((t) => MediaRecorder.isTypeSupported?.(t)) || "";
}

export function useRecorder({ maxSeconds = 10, onComplete }) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [analyser, setAnalyser] = useState(null);
  const r = useRef({});
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const cleanup = useCallback(() => {
    const { stream, ctx, timer } = r.current;
    clearInterval(timer);
    stream?.getTracks().forEach((t) => t.stop());
    if (ctx && ctx.state !== "closed") ctx.close();
    r.current = {};
    setAnalyser(null);
    setRecording(false);
  }, []);

  const stop = useCallback(() => {
    const rec = r.current.recorder;
    if (rec && rec.state === "recording") rec.stop();
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      throw new Error("This browser can’t record audio. Upload a file instead.");
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
    } catch {
      throw new Error("Microphone access was blocked. Allow it in your browser’s site settings, or upload a file instead.");
    }

    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    const an = ctx.createAnalyser();
    an.fftSize = 2048;
    ctx.createMediaStreamSource(stream).connect(an);

    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      cleanup();
      onCompleteRef.current?.(blob);
    };

    const t0 = Date.now();
    const timer = setInterval(() => {
      const s = (Date.now() - t0) / 1000;
      setElapsed(s);
      if (s >= maxSeconds) stop();
    }, 100);

    r.current = { stream, ctx, recorder, timer };
    setElapsed(0);
    setAnalyser(an);
    setRecording(true);
    recorder.start(250);
  }, [cleanup, maxSeconds, stop]);

  // release the mic if the component unmounts mid-recording
  useEffect(() => () => {
    const rec = r.current.recorder;
    if (rec) rec.onstop = null;
    if (rec && rec.state === "recording") rec.stop();
    cleanup();
  }, [cleanup]);

  return { recording, elapsed, analyser, start, stop };
}
