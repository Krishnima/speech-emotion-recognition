import { useEffect, useState } from "react";

let ctx = null;
let current = null; // { source, stop }

export default function PlayButton({ samples, sr = 22050, label = "Play" }) {
  const [playing, setPlaying] = useState(false);

  useEffect(() => () => current?.owner === setPlaying && current.stop(), []);

  function toggle() {
    if (playing) {
      current?.stop();
      return;
    }
    current?.stop();
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = ctx || new AC();
    ctx.resume?.();
    const buf = ctx.createBuffer(1, samples.length, sr);
    buf.copyToChannel ? buf.copyToChannel(samples, 0) : buf.getChannelData(0).set(samples);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const entry = {
      owner: setPlaying,
      stop: () => {
        try { src.stop(); } catch { /* already stopped */ }
      },
    };
    src.onended = () => {
      setPlaying(false);
      if (current === entry) current = null;
    };
    current = entry;
    src.start();
    setPlaying(true);
  }

  return (
    <button className="play" onClick={toggle} aria-label={`${playing ? "Stop" : "Play"} ${label}`}>
      <span className={playing ? "ico stop" : "ico tri"} aria-hidden="true" />
      {playing ? "Stop" : "Play"}
    </button>
  );
}
