import { useEffect, useRef, useState } from "react";

// Live oscilloscope while recording, static bar waveform once a clip is loaded.
export default function Waveform({ samples, analyser, color }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = wrapRef.current;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size.w * dpr;
    c.height = size.h * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { w, h } = size;
    const paint = getComputedStyle(c).color;

    if (analyser) {
      const buf = new Float32Array(analyser.fftSize);
      let raf;
      const loop = () => {
        analyser.getFloatTimeDomainData(buf);
        g.clearRect(0, 0, w, h);
        g.lineWidth = 2.5;
        g.strokeStyle = paint;
        g.beginPath();
        for (let i = 0; i < buf.length; i++) {
          const x = (i / buf.length) * w;
          const y = h / 2 + buf[i] * h * 1.6;
          i ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.stroke();
        raf = requestAnimationFrame(loop);
      };
      loop();
      return () => cancelAnimationFrame(raf);
    }

    g.clearRect(0, 0, w, h);
    if (!samples?.length) return;
    let peak = 0;
    for (let i = 0; i < samples.length; i += 32) peak = Math.max(peak, Math.abs(samples[i]));
    peak = peak || 1;
    const bars = Math.floor(w / 4);
    const step = Math.max(1, Math.floor(samples.length / bars));
    g.fillStyle = paint;
    for (let i = 0; i < bars; i++) {
      let m = 0;
      for (let j = i * step; j < (i + 1) * step && j < samples.length; j += 8) {
        m = Math.max(m, Math.abs(samples[j]));
      }
      const bh = Math.max(2, (m / peak) * h * 0.82);
      g.fillRect(i * 4, (h - bh) / 2, 2.5, bh);
    }
  }, [samples, analyser, color, size]);

  const empty = !analyser && !samples?.length;
  return (
    <div className="scope" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ color: analyser ? "#D9473B" : color }} aria-hidden="true" />
      {empty && <div className="scope-empty">Your waveform will show here</div>}
    </div>
  );
}
