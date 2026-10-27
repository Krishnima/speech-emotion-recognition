// RAVDESS file name format: modality-vocal-emotion-intensity-statement-repetition-actor.wav
export const RAVDESS_FIELDS = [
  { key: "modality", name: "Modality", codes: { "01": "Full audio-video", "02": "Video only", "03": "Audio only" } },
  { key: "vocal", name: "Vocal channel", codes: { "01": "Speech", "02": "Song" } },
  {
    key: "emotion", name: "Emotion",
    codes: { "01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised" },
  },
  { key: "intensity", name: "Intensity", codes: { "01": "Normal", "02": "Strong" } },
  { key: "statement", name: "Statement", codes: { "01": "“Kids are talking by the door”", "02": "“Dogs are sitting by the door”" } },
  { key: "repetition", name: "Repetition", codes: { "01": "First take", "02": "Second take" } },
  { key: "actor", name: "Actor", codes: null },
];

export function parseRavdess(name) {
  const m = /(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})/.exec(name || "");
  if (!m) return null;
  const parts = m.slice(1);
  const fields = RAVDESS_FIELDS.map((f, i) => {
    const code = parts[i];
    let meaning;
    if (f.key === "actor") {
      const n = Number(code);
      meaning = n >= 1 && n <= 24 ? `Actor ${n} (${n % 2 ? "male" : "female"})` : null;
    } else {
      meaning = f.codes[code] || null;
    }
    return { ...f, code, meaning };
  });
  const emotion = fields[2].meaning;
  return { fields, emotion, valid: fields.every((f) => f.meaning) };
}

// Speech files per emotion: 24 actors x 2 statements x 2 repetitions x 2 intensities (neutral has 1 intensity)
export const RAVDESS_COUNTS = {
  neutral: 96, calm: 192, happy: 192, sad: 192, angry: 192, fearful: 192, disgust: 192, surprised: 192,
};
