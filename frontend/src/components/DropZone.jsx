import { useState } from "react";

export default function DropZone({ onFile, disabled }) {
  const [over, setOver] = useState(false);

  const stop = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <label
      className={`drop${over ? " over" : ""}${disabled ? " disabled" : ""}`}
      onDragEnter={(e) => { stop(e); setOver(true); }}
      onDragOver={(e) => { stop(e); setOver(true); }}
      onDragLeave={(e) => { stop(e); setOver(false); }}
      onDrop={(e) => {
        stop(e);
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !disabled) onFile(f);
      }}
    >
      <input
        type="file"
        accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a,.webm"
        disabled={disabled}
        data-testid="file-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      Drop an audio file here or <u>choose one</u>
    </label>
  );
}
