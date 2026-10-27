"""
Pretrained speech model (WavLM) embeddings for the high-accuracy model.

WavLM was pretrained by Microsoft on tens of thousands of hours of unlabeled speech, so its
internal layers already describe tone, pitch, energy and voice quality far better than
hand-made features. Each layer's output is averaged over time (mean + standard deviation),
giving one vector per layer; a small classifier is then trained on the best layers.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_HERE, ".numba_cache"))
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")       # torch + scikit-learn both ship OpenMP on macOS
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: fall back to CPU for missing ops
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import librosa

from features import add_noise, stretch, pitch

SSL_SR = 16000                       # WavLM expects 16 kHz audio
DEFAULT_BACKBONE = "microsoft/wavlm-large"
TRIM_DB = 30
MARGIN_S = 0.15                      # keep a little audio around the detected speech
MIN_S = 0.3
MAX_S = 8.0                          # longer clips are split into chunks
CHUNK_S = 6.0
HOP_S = 3.0
MAX_CHUNKS = 8


def trim_speech(y, sr=SSL_SR):
    """Cut leading/trailing silence, the same way for training and for the app."""
    if y.size == 0:
        return y
    _, (s, e) = librosa.effects.trim(y, top_db=TRIM_DB)
    if e - s < MIN_S * sr:
        return y
    m = int(MARGIN_S * sr)
    return y[max(0, s - m): min(len(y), e + m)]


def chunks(y, sr=SSL_SR):
    if len(y) <= MAX_S * sr:
        return [y]
    win, hop = int(CHUNK_S * sr), int(HOP_S * sr)
    return [y[i:i + win] for i in range(0, len(y) - win + 1, hop)][:MAX_CHUNKS]


def load_clip(path):
    y, _ = librosa.load(path, sr=SSL_SR, mono=True)
    return trim_speech(y.astype(np.float32))


def augmented_versions(y):
    """Same augmentations as the notebook: original, noise, and stretch followed by pitch shift."""
    return {
        "orig": y,
        "noise": np.asarray(add_noise(y), dtype=np.float32),
        "stretchpitch": np.asarray(pitch(stretch(y), sr=SSL_SR), dtype=np.float32),
    }


class Embedder:
    """Turns a 16 kHz clip into an array of shape (layers, 2 * hidden_size)."""

    def __init__(self, backbone=DEFAULT_BACKBONE, device="auto", verbose=True):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch = torch
        self.backbone = backbone
        if verbose:
            print(f"Loading {backbone} (the first run downloads it, about 1.2 GB for WavLM Large) ...", flush=True)
        self.fe = AutoFeatureExtractor.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone)
        self.model.eval()
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers + 1
        self.hidden = cfg.hidden_size
        self.dim = 2 * self.hidden
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.device = self._pick_device(device, verbose)
        if verbose:
            print(f"  {self.n_params / 1e6:.0f}M parameters, {self.n_layers} layers, running on {self.device}", flush=True)

    def _pick_device(self, device, verbose):
        torch = self.torch
        if device == "auto":
            cands = []
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                cands.append("mps")
            cands.append("cpu")
        else:
            cands = [device]
        probe = (np.random.RandomState(0).randn(SSL_SR) * 0.01).astype(np.float32)
        for d in cands:
            try:
                self.model.to(d)
                self._embed_on(probe, d)
                return d
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"  {d} not usable ({type(e).__name__}: {e}); trying the next option", flush=True)
        raise RuntimeError("Could not run the speech model on any device.")

    def _embed_on(self, y, device):
        torch = self.torch
        inputs = self.fe(y, sampling_rate=SSL_SR, return_tensors="pt")
        with torch.inference_mode():
            out = self.model(inputs["input_values"].to(device), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)[:, 0].float()      # (layers, frames, hidden)
        feat = torch.cat([hs.mean(1), hs.std(1)], dim=-1)          # (layers, 2 * hidden)
        return feat.cpu().numpy().astype(np.float32)

    def embed(self, y):
        return self._embed_on(np.asarray(y, dtype=np.float32), self.device)
