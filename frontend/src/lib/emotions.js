export const EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];

export const EMOTION_COLORS = {
  neutral: "#7D8799",
  calm: "#3E9EA0",
  happy: "#E0A912",
  sad: "#4C6EDB",
  angry: "#D9473B",
  fearful: "#8E5BC9",
  disgust: "#5E9632",
  surprised: "#E2679F",
};

export const colorFor = (emotion) => EMOTION_COLORS[emotion] || "var(--accent)";
