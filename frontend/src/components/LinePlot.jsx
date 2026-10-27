import { useEffect, useRef, useState } from "react";

export default function LinePlot({ values, height = 90, color = "var(--emo)", label }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [w, setW] = useState(0);

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !w || !values?.length) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = height * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const paint = getComputedStyle(c).color;
    const max = Math.max(...values) || 1;
    g.clearRect(0, 0, w, height);
    g.beginPath();
    values.forEach((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * w;
      const y = height - 4 - (v / max) * (height - 10);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.strokeStyle = paint;
    g.lineWidth = 2;
    g.stroke();
    g.lineTo(w, height);
    g.lineTo(0, height);
    g.closePath();
    g.globalAlpha = 0.15;
    g.fillStyle = paint;
    g.fill();
    g.globalAlpha = 1;
  }, [values, w, height]);

  return (
    <div className="lineplot" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ height, color }} role="img" aria-label={label} />
    </div>
  );
}
