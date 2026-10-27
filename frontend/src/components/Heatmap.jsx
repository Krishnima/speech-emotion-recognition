import { useEffect, useRef, useState } from "react";

// inferno-like ramp
const STOPS = [
  [0.0, [13, 8, 35]], [0.2, [66, 10, 104]], [0.4, [147, 38, 103]],
  [0.6, [221, 81, 58]], [0.8, [252, 165, 10]], [1.0, [252, 255, 164]],
];
const LUT = Array.from({ length: 256 }, (_, i) => {
  const t = i / 255;
  let j = 0;
  while (j < STOPS.length - 2 && t > STOPS[j + 1][0]) j++;
  const [t0, c0] = STOPS[j], [t1, c1] = STOPS[j + 1];
  const a = (t - t0) / (t1 - t0);
  return c0.map((v, k) => Math.round(v + a * (c1[k] - v)));
});

// data: array of frames, each an array of `rows` values (row 0 drawn at the bottom)
export default function Heatmap({ data, rows, height = 140, range, label }) {
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
    if (!c || !w || !data?.length) return;
    const nRows = rows ?? data[0].length;
    let lo = Infinity, hi = -Infinity;
    if (range) [lo, hi] = range;
    else for (const f of data) for (let r = 0; r < nRows; r++) { lo = Math.min(lo, f[r]); hi = Math.max(hi, f[r]); }
    const span = hi - lo || 1;

    const off = document.createElement("canvas");
    off.width = data.length;
    off.height = nRows;
    const g = off.getContext("2d");
    const img = g.createImageData(data.length, nRows);
    for (let x = 0; x < data.length; x++) {
      for (let r = 0; r < nRows; r++) {
        const v = Math.max(0, Math.min(255, Math.round(((data[x][r] - lo) / span) * 255)));
        const [R, G, B] = LUT[v];
        const p = ((nRows - 1 - r) * data.length + x) * 4;
        img.data[p] = R; img.data[p + 1] = G; img.data[p + 2] = B; img.data[p + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);

    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = height * dpr;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = nRows > 24;
    ctx.drawImage(off, 0, 0, c.width, c.height);
  }, [data, rows, w, height, range]);

  return (
    <div className="heatmap" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ height }} role="img" aria-label={label} />
    </div>
  );
}
