"""
Audio loading, augmentation and feature extraction.

Same feature set as the Kaggle notebook (shivamburnwal/speech-emotion-recognition):
ZCR (1) + Chroma STFT (12) + MFCC (20) + RMS (1) + Mel spectrogram (128) = 162 features.
This module is shared by training and inference so both see identical features.
"""
import os as _os
# Keep librosa's compiled-code cache inside the project. Avoids crashes (SIGSEGV) from a
# cache that was corrupted by several processes writing to it at once.
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".numba_cache"))
import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 2.5      # seconds of audio used per training clip
OFFSET = 0.6        # skip the silent lead-in of RAVDESS clips
N_FEATURES = 162

# RAVDESS filename: modality-vocal-EMOTION-intensity-statement-repetition-actor.wav
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


# ---------- augmentation (same ideas as the notebook) ----------
def add_noise(y):
    amp = 0.035 * np.random.uniform() * np.amax(np.abs(y) + 1e-9)
    return y + amp * np.random.normal(size=y.shape[0])


def shift(y):
    """Roll the signal by a random amount (notebook: +/-5000 samples)."""
    return np.roll(y, int(np.random.uniform(low=-5, high=5) * 1000))


def stretch(y, rate=0.8):
    return librosa.effects.time_stretch(y=y, rate=rate)


def pitch(y, sr=SAMPLE_RATE, n_steps=0.7):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


# ---------- features ----------
def extract_features(y, sr=SAMPLE_RATE):
    y = np.asarray(y, dtype=np.float32)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    feats = np.hstack([zcr, chroma, mfcc, rms, mel]).astype(np.float32)
    assert feats.shape[0] == N_FEATURES, feats.shape
    return feats


def load_training_clip(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
    return y, sr


def features_for_file(path, augment=False):
    """Returns a list of feature vectors: original (+ noisy + stretched/pitched if augment)."""
    y, sr = load_training_clip(path)
    out = [extract_features(y, sr)]
    if augment:
        out.append(extract_features(add_noise(y), sr))
        out.append(extract_features(pitch(stretch(y), sr), sr))
    return out
