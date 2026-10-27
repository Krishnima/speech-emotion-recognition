"""
Notebook-style data exploration for RAVDESS. Saves figures to ./reports:

  emotion_counts.png        count of each emotion (notebook: "Count of Emotions")
  wave_<emotion>.png        waveplot + spectrogram for one clip of each emotion
  augmentation.png          the same clip with noise, stretch, shift and pitch

Usage:  python explore.py --data data/ravdess
"""
import os as _os
# Keep librosa's compiled-code cache inside the project. Avoids crashes (SIGSEGV) from a
# cache that was corrupted by several processes writing to it at once.
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".numba_cache"))
import argparse
import os
from collections import Counter

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import add_noise, stretch, shift, pitch, SAMPLE_RATE
from train import find_ravdess_files, LABELS


def waveplot_and_spectrogram(path, emotion, out_dir):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].set_title(f"Waveplot for audio with {emotion} emotion")
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(np.abs(X))
    ax[1].set_title(f"Spectrogram for audio with {emotion} emotion")
    img = librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz", ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"wave_{emotion}.png"), dpi=110)
    plt.close(fig)


def augmentation_figure(path, out_dir):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    versions = {
        "Original": y,
        "Noise": add_noise(y),
        "Stretch (rate 0.8)": stretch(y),
        "Shift (np.roll)": shift(y),
        "Pitch (+0.7 semitones)": pitch(y, sr),
    }
    fig, axes = plt.subplots(len(versions), 1, figsize=(10, 2.2 * len(versions)), sharex=True)
    for ax, (name, sig) in zip(axes, versions.items()):
        librosa.display.waveshow(sig, sr=sr, ax=ax)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "augmentation.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths, labels = find_ravdess_files(args.data)
    if not paths:
        raise SystemExit(f"No RAVDESS files under '{args.data}'. See README for the download step.")

    counts = Counter(labels)
    print("Files per emotion:")
    for e in LABELS:
        print(f"  {e:<10} {counts.get(e, 0)}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(LABELS, [counts.get(e, 0) for e in LABELS], color="#4757D6")
    ax.set_title("Count of Emotions")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "emotion_counts.png"), dpi=110)
    plt.close(fig)

    for e in LABELS:
        first = next(p for p, l in zip(paths, labels) if l == e)
        waveplot_and_spectrogram(first, e, args.out)

    augmentation_figure(paths[0], args.out)
    print(f"Saved figures to '{args.out}/'")


if __name__ == "__main__":
    main()
