"""
Inference for the trained model. Also usable from the command line:
    python predict.py path/to/clip.wav
"""
import os as _os
# Keep librosa's compiled-code cache inside the project. Avoids crashes (SIGSEGV) from a
# cache that was corrupted by several processes writing to it at once.
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".numba_cache"))
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import os
import sys

import numpy as np
import joblib
import librosa

from features import SAMPLE_RATE, DURATION, extract_features

MIN_SECONDS = 0.4
MAX_WINDOWS = 12
LEAD_SECONDS = 0.3  # keep a little audio before the speech starts, like the training windows do


class EmotionPredictor:
    """Uses the WavLM model if train_ssl.py has been run (models/ssl_config.json), else the CNN."""

    def __init__(self, model_dir="models"):
        metrics_path = os.path.join(model_dir, "metrics.json")
        self.metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        if os.path.exists(os.path.join(model_dir, "ssl_config.json")):
            self._init_ssl(model_dir)
        else:
            self._init_cnn(model_dir)

    def _init_ssl(self, model_dir):
        import ssl_features as sf
        cfg = json.load(open(os.path.join(model_dir, "ssl_config.json")))
        head = joblib.load(os.path.join(model_dir, "ssl_head.joblib"))
        self.kind = "ssl"
        self.sf = sf
        self.labels = cfg["labels"]
        self.layers = head["layers"]
        self.clf = head["pipeline"]
        classes = list(self.clf.classes_)
        self._order = [classes.index(l) for l in self.labels]
        self.embedder = sf.Embedder(cfg["backbone"], device=os.environ.get("SER_DEVICE", "cpu"), verbose=False)

    def _init_cnn(self, model_dir):
        self.kind = "cnn"
        model_path = os.path.join(model_dir, "ser_model.keras")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        labels_path = os.path.join(model_dir, "labels.json")
        missing = [p for p in (model_path, scaler_path, labels_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Model files not found: " + ", ".join(missing) +
                ". Train first with: python train.py --data data/ravdess"
            )
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(labels_path) as f:
            self.labels = json.load(f)

    def _segments(self, y, sr):
        """Cut 2.5 s windows the same way training did.

        Training used librosa.load(offset=0.6, duration=2.5) on RAVDESS files, where speech starts
        around 0.8-1.2 s. So each training window holds the sentence plus a little quiet before and
        after it. Features are averaged over the window, so inference must match that: start the
        window shortly before the detected speech and keep a fixed 2.5 s length.
        """
        _, (start, end) = librosa.effects.trim(y, top_db=30)
        if end - start < MIN_SECONDS * sr:
            start, end = 0, len(y)
        if end - start < MIN_SECONDS * sr:
            raise ValueError("The clip is too short. Record or upload at least 1 second of speech.")
        win, hop = int(DURATION * sr), int(1.0 * sr)
        first = max(0, start - int(LEAD_SECONDS * sr))
        if end - first <= int(win * 1.3):
            return [y[first:first + win]]
        segs = [y[s:s + win] for s in range(first, end - win + 1, hop)]
        return segs[:MAX_WINDOWS] or [y[first:first + win]]

    def predict(self, path):
        return self._predict_ssl(path) if self.kind == "ssl" else self._predict_cnn(path)

    def _predict_ssl(self, path):
        sf = self.sf
        y, sr = librosa.load(path, sr=sf.SSL_SR, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-4:
            raise ValueError("The clip is silent. Check your microphone and try again.")
        speech = sf.trim_speech(y.astype(np.float32), sr)
        if len(speech) < MIN_SECONDS * sr:
            raise ValueError("The clip is too short. Record or upload at least 1 second of speech.")
        segs = sf.chunks(speech, sr)
        # same float16 rounding as the cached training embeddings
        X = np.stack([self.embedder.embed(s).astype(np.float16).astype(np.float32)[self.layers].reshape(-1)
                      for s in segs])
        probs = self.clf.predict_proba(X)[:, self._order].mean(axis=0)
        order = np.argsort(probs)[::-1]
        return {
            "emotion": self.labels[order[0]],
            "confidence": float(probs[order[0]]),
            "probabilities": {self.labels[i]: float(probs[i]) for i in order},
            "duration": round(len(y) / sr, 2),
            "windows": len(segs),
        }

    def _predict_cnn(self, path):
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-4:
            raise ValueError("The clip is silent. Check your microphone and try again.")
        segs = self._segments(y, sr)
        X = np.stack([extract_features(s, sr) for s in segs])
        X = self.scaler.transform(X)[..., np.newaxis]
        probs = self.model.predict(X, verbose=0).mean(axis=0)
        order = np.argsort(probs)[::-1]
        return {
            "emotion": self.labels[order[0]],
            "confidence": float(probs[order[0]]),
            "probabilities": {self.labels[i]: float(probs[i]) for i in order},
            "duration": round(len(y) / sr, 2),
            "windows": len(segs),
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict.py path/to/audio.wav")
    p = EmotionPredictor()
    r = p.predict(sys.argv[1])
    print(f"\nPredicted emotion: {r['emotion']} ({r['confidence'] * 100:.1f}%)\n")
    for k, v in r["probabilities"].items():
        print(f"  {k:<10} {'#' * int(v * 40):<40} {v * 100:5.1f}%")
