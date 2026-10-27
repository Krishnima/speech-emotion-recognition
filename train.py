"""
Train the Speech Emotion Recognition model on the RAVDESS dataset only.

Usage:
    python train.py --data data/ravdess
    python train.py --data data/ravdess --epochs 80 --batch 64
    python train.py --data data/ravdess --jobs 4      # parallel feature extraction (optional)

Extracted features are cached in models/features_cache.npz, so a second run skips
that slow step. Use --no-cache to extract again.

Outputs (in ./models):
    ser_model.keras, scaler.pkl, labels.json, metrics.json,
    confusion_matrix.png, training_curves.png
"""
import os as _os
# Keep librosa's compiled-code cache inside the project. Avoids crashes (SIGSEGV) from a
# cache that was corrupted by several processes writing to it at once.
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".numba_cache"))
import argparse
import faulthandler
import hashlib
import json
import os
import glob
import time

import numpy as np
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from features import RAVDESS_EMOTIONS, N_FEATURES, features_for_file

LABELS = list(RAVDESS_EMOTIONS.values())
faulthandler.enable()  # if a native library crashes, print where


def find_ravdess_files(root):
    files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    paths, labels = [], []
    for f in files:
        parts = os.path.basename(f).split(".")[0].split("-")
        # speech files only: modality 03 (audio-only), vocal channel 01 (speech)
        if len(parts) != 7 or parts[1] != "01":
            continue
        emo = RAVDESS_EMOTIONS.get(parts[2])
        if emo:
            paths.append(f)
            labels.append(emo)
    # de-duplicate (the Kaggle zip contains the actors twice in some versions)
    seen, P, L = set(), [], []
    for p, l in zip(paths, labels):
        key = os.path.basename(p)
        if key not in seen:
            seen.add(key)
            P.append(p)
            L.append(l)
    return P, L


def extract(paths, labels, augment, desc, jobs=1):
    t = time.time()
    n = len(paths)
    print(f"Extracting features for {n} {desc} files (augment={augment}, jobs={jobs}) ...", flush=True)
    if jobs == 1:
        results = []
        for i, p in enumerate(paths, 1):
            results.append(features_for_file(p, augment))
            if i % 25 == 0 or i == n:
                el = time.time() - t
                print(f"  {desc}: {i}/{n} files, {el:.0f}s elapsed, about {el / i * (n - i):.0f}s left", flush=True)
    else:
        features_for_file(paths[0], augment)  # compile librosa's code once before starting workers
        results = Parallel(n_jobs=jobs, verbose=5)(
            delayed(features_for_file)(p, augment) for p in paths
        )
    X, y = [], []
    for feats, lab in zip(results, labels):
        for f in feats:
            X.append(f)
            y.append(lab)
    print(f"  -> {len(X)} samples in {time.time() - t:.1f}s")
    return np.array(X, dtype=np.float32), np.array(y)


def build_model(n_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(N_FEATURES, 1)),
        layers.Conv1D(256, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(5, strides=2, padding="same"),
        layers.Conv1D(256, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(5, strides=2, padding="same"),
        layers.Conv1D(128, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(5, strides=2, padding="same"),
        layers.Dropout(0.2),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(5, strides=2, padding="same"),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_plots(history, cm, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["accuracy"], label="train")
    ax[1].plot(history["val_accuracy"], label="test")
    ax[1].set_title("Accuracy"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess", help="folder containing RAVDESS Actor_XX folders")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1, help="processes for feature extraction (1 = safest)")
    ap.add_argument("--no-cache", action="store_true", help="re-extract features even if a cache exists")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    paths, labels = find_ravdess_files(args.data)
    if not paths:
        raise SystemExit(
            f"No RAVDESS .wav files found under '{args.data}'.\n"
            "Download the dataset (see README) and extract it so it contains Actor_01 ... Actor_24."
        )
    print(f"Found {len(paths)} RAVDESS speech files.")

    # Split by FILE before augmentation so augmented copies of a test clip never leak into training.
    p_tr, p_te, l_tr, l_te = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )
    # record exactly which clips were held out, so they can be used for demos and evaluate.py
    with open(os.path.join(args.out, "test_files.txt"), "w") as fh:
        for p_, l_ in zip(p_te, l_te):
            fh.write(f"{l_}\t{p_}\n")

    cache_path = os.path.join(args.out, "features_cache.npz")
    key = hashlib.md5(("|".join(sorted(os.path.basename(p) for p in paths)) + f"|seed={args.seed}").encode()).hexdigest()
    X_tr = None
    if os.path.exists(cache_path) and not args.no_cache:
        try:
            c = np.load(cache_path, allow_pickle=False)
            if str(c["key"]) == key:
                X_tr, y_tr, X_te, y_te = c["X_tr"], c["y_tr"], c["X_te"], c["y_te"]
                print(f"Loaded cached features from {cache_path} ({len(X_tr)} train, {len(X_te)} test samples).")
        except Exception as e:  # noqa: BLE001
            print(f"Ignoring unreadable feature cache ({e}).")
    if X_tr is None:
        X_tr, y_tr = extract(p_tr, l_tr, augment=True, desc="train", jobs=args.jobs)
        X_te, y_te = extract(p_te, l_te, augment=False, desc="test", jobs=args.jobs)
        np.savez(cache_path, key=key, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
        print(f"Saved features to {cache_path}")

    # like the notebook: keep the raw 162-feature table as CSV
    import csv
    with open(os.path.join(args.out, "features.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"f{i}" for i in range(N_FEATURES)] + ["labels", "split"])
        for row, lab in zip(X_tr, y_tr):
            w.writerow([f"{v:.6g}" for v in row] + [lab, "train"])
        for row, lab in zip(X_te, y_te):
            w.writerow([f"{v:.6g}" for v in row] + [lab, "test"])

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)[..., np.newaxis]
    X_te = scaler.transform(X_te)[..., np.newaxis]

    idx = {l: i for i, l in enumerate(LABELS)}
    Y_tr = np.eye(len(LABELS))[[idx[l] for l in y_tr]]
    Y_te = np.eye(len(LABELS))[[idx[l] for l in y_te]]

    import tensorflow as tf
    tf.random.set_seed(args.seed)
    from tensorflow import keras

    model = build_model(len(LABELS))
    model.summary()
    model_path = os.path.join(args.out, "ser_model.keras")
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
    ]
    hist = model.fit(X_tr, Y_tr, validation_data=(X_te, Y_te),
                     epochs=args.epochs, batch_size=args.batch, callbacks=callbacks, verbose=2)

    loss, acc = model.evaluate(X_te, Y_te, verbose=0)
    pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    true = np.argmax(Y_te, axis=1)

    # notebook-style "Predicted vs Actual" table
    print("\nPredicted vs actual (first 10 test clips):")
    for p_, t_ in list(zip(pred, true))[:10]:
        print(f"  predicted {LABELS[p_]:<10} actual {LABELS[t_]}")
    report = classification_report(true, pred, labels=range(len(LABELS)), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    print(classification_report(true, pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0))
    print(f"Test accuracy: {acc * 100:.2f}%")

    model.save(model_path)
    joblib.dump(scaler, os.path.join(args.out, "scaler.pkl"))
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(LABELS, f)
    cm = confusion_matrix(true, pred, labels=range(len(LABELS)))
    metrics = {
        "dataset": "RAVDESS (speech)",
        "labels": LABELS,
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "n_train_files": len(p_tr),
        "n_test_files": len(p_te),
        "n_train_samples": int(len(X_tr)),
        "epochs_run": len(hist.history["loss"]),
        "per_class_f1": {l: float(report[l]["f1-score"]) for l in LABELS},
        "report": {l: {"precision": float(report[l]["precision"]), "recall": float(report[l]["recall"]),
                       "f1": float(report[l]["f1-score"]), "support": int(report[l]["support"])} for l in LABELS},
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(v) for v in hist.history[k]]
                    for k in ("loss", "val_loss", "accuracy", "val_accuracy") if k in hist.history},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_plots(hist.history, cm, args.out)
    print(f"Saved model and artifacts to '{args.out}/'. Now run: python app.py")


if __name__ == "__main__":
    main()
