"""
Train the high-accuracy model: WavLM (pretrained speech model) + a classifier on its best layers.

    python train_ssl.py --data data/ravdess
    python train_ssl.py --data data/ravdess --backbone microsoft/wavlm-base-plus   # ~3x faster, a bit less accurate

Same 288 test clips as train.py, so the two models can be compared fairly. The 1,152 training
files are split again into 1,036 for training and 116 for validation; all tuning uses the
validation set only, and the test set is scored once at the end.

Embeddings are saved per clip in models/ssl_cache/, so an interrupted run resumes where it
stopped and later runs skip the slow part. When this finishes, the web app uses this model
automatically (delete models/ssl_config.json to go back to the CNN).
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_HERE, ".numba_cache"))
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import os
import shutil
import time
import warnings

import numpy as np
import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ssl_features as sf
from train import find_ravdess_files, LABELS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
KINDS_AUG = ["orig", "noise", "stretchpitch"]


class LazyEmbedder:
    """Loads WavLM only if some clip still needs embedding."""

    def __init__(self, backbone, device, meta_path):
        self.backbone, self.device, self.meta_path, self.e = backbone, device, meta_path, None

    def get(self):
        if self.e is None:
            self.e = sf.Embedder(self.backbone, device=self.device)
            with open(self.meta_path, "w") as fh:
                json.dump({"backbone": self.backbone, "n_params": self.e.n_params,
                           "n_layers": self.e.n_layers, "hidden": self.e.hidden}, fh)
        return self.e


def embed_files(lazy, paths, augment, cdir, desc):
    kinds = KINDS_AUG if augment else ["orig"]
    X, owner = [], []
    t, new, n = time.time(), 0, len(paths)
    print(f"Embedding {n} {desc} files ({', '.join(kinds)}) ...", flush=True)
    for i, p in enumerate(paths):
        base = os.path.splitext(os.path.basename(p))[0]
        files = {k: os.path.join(cdir, f"{base}__{k}.npy") for k in kinds}
        if all(os.path.exists(f) for f in files.values()):
            feats = {k: np.load(f) for k, f in files.items()}
        else:
            y = sf.load_clip(p)
            versions = sf.augmented_versions(y) if augment else {"orig": y}
            feats = {}
            for k in kinds:
                if os.path.exists(files[k]):
                    feats[k] = np.load(files[k])
                    continue
                f = lazy.get().embed(versions[k]).astype(np.float16)
                np.save(files[k], f)
                feats[k] = f
            new += 1
        for k in kinds:
            X.append(feats[k])
            owner.append(i)
        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.time() - t
            left = el / new * (n - i - 1) if new else 0
            print(f"  {desc}: {i + 1}/{n} files, {el:.0f}s elapsed, about {left:.0f}s left", flush=True)
    return np.stack(X), np.array(owner)


def select(X, layers):
    return X[:, layers].reshape(len(X), -1).astype(np.float32)


def fit(X, y, C):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(X, y)
    return clf


def save_figures(layer_acc, chosen, cm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ["#4757D6" if i in chosen else "#B8BFD6" for i in range(len(layer_acc))]
    ax.bar(range(len(layer_acc)), [a * 100 for a in layer_acc], color=colors)
    ax.set_xlabel("WavLM layer"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("How much each layer knows about emotion (chosen layers in blue)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ssl_layer_accuracy.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set), WavLM model")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(out, "ssl_confusion_matrix.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess")
    ap.add_argument("--out", default="models")
    ap.add_argument("--backbone", default=sf.DEFAULT_BACKBONE)
    ap.add_argument("--device", default="auto", help="auto (Apple GPU if it works, else CPU), cpu or mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-layers", type=int, default=3)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    paths, labels = find_ravdess_files(args.data)
    if not paths:
        raise SystemExit(f"No RAVDESS .wav files found under '{args.data}'.")
    print(f"Found {len(paths)} RAVDESS speech files.")

    # identical test split to train.py, then a validation split carved from the training files
    p_all, p_te, l_all, l_te = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=args.seed)
    p_tr, p_va, l_tr, l_va = train_test_split(p_all, l_all, test_size=0.1, stratify=l_all, random_state=args.seed)
    print(f"Split: {len(p_tr)} train, {len(p_va)} validation, {len(p_te)} test files (test = same as train.py).")
    with open(os.path.join(args.out, "test_files.txt"), "w") as fh:
        for p_, l_ in zip(p_te, l_te):
            fh.write(f"{l_}\t{p_}\n")

    cdir = os.path.join(args.out, "ssl_cache", args.backbone.replace("/", "__"))
    os.makedirs(cdir, exist_ok=True)
    meta_path = os.path.join(cdir, "meta.json")
    lazy = LazyEmbedder(args.backbone, args.device, meta_path)

    t0 = time.time()
    Xtr, otr = embed_files(lazy, p_tr, not args.no_augment, cdir, "train")
    Xva, ova = embed_files(lazy, p_va, False, cdir, "validation")
    Xte, ote = embed_files(lazy, p_te, False, cdir, "test")
    ytr, yva, yte = np.array(l_tr)[otr], np.array(l_va)[ova], np.array(l_te)[ote]
    print(f"Embeddings ready in {time.time() - t0:.0f}s: {Xtr.shape[0]} train samples, "
          f"{Xtr.shape[1]} layers x {Xtr.shape[2]} values each.\n")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    # 1) which layers carry emotion? one quick classifier per layer, scored on validation
    n_layers = Xtr.shape[1]
    print("Step 1/3: scoring each layer on the validation set ...", flush=True)
    layer_acc = []
    for l in range(n_layers):
        clf = fit(select(Xtr, [l]), ytr, C=0.1)
        layer_acc.append(float(clf.score(select(Xva, [l]), yva)))
        print(f"  layer {l:2d}: {layer_acc[-1] * 100:5.1f}%", flush=True)
    ranked = sorted(range(n_layers), key=lambda l: -layer_acc[l])

    # 2) combine the best layers, choose regularisation on validation
    print("\nStep 2/3: combining the best layers ...", flush=True)
    search, best = [], None
    for k in range(1, max(1, args.top_layers) + 1):
        layers = sorted(ranked[:k])
        for C in (0.01, 0.1, 1.0):
            clf = fit(select(Xtr, layers), ytr, C)
            acc = float(clf.score(select(Xva, layers), yva))
            search.append({"layers": layers, "C": C, "val_accuracy": acc})
            print(f"  layers {layers}, C={C}: {acc * 100:5.1f}%", flush=True)
            if best is None or acc > best[0] + 1e-9:
                best = (acc, layers, C, clf)
    val_acc, layers, C, clf = best
    print(f"  -> using layers {layers} with C={C} (validation {val_acc * 100:.1f}%)")

    # 3) score the untouched test set once
    print("\nStep 3/3: test set ...", flush=True)
    classes = list(clf.classes_)
    probs = clf.predict_proba(select(Xte, layers))
    pred = np.array(classes)[probs.argmax(1)]
    acc = float(np.mean(pred == yte))
    print("\nPredicted vs actual (first 10 test clips):")
    for p_, t_ in list(zip(pred, yte))[:10]:
        print(f"  predicted {p_:<10} actual {t_}")
    print(classification_report(yte, pred, labels=LABELS, zero_division=0))
    report = classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=LABELS)

    # keep the CNN's numbers as the baseline for comparison
    metrics_path = os.path.join(args.out, "metrics.json")
    cnn_path = os.path.join(args.out, "metrics_cnn.json")
    if os.path.exists(metrics_path):
        old = json.load(open(metrics_path))
        if old.get("model_type") != "ssl":
            shutil.copy(metrics_path, cnn_path)
    baseline = json.load(open(cnn_path)).get("test_accuracy") if os.path.exists(cnn_path) else None

    short = args.backbone.split("/")[-1].replace("wavlm", "WavLM").replace("-", " ").title().replace("Wavlm", "WavLM")
    metrics = {
        "model_type": "ssl",
        "model_label": f"{short} + logistic regression",
        "dataset": "RAVDESS (speech)",
        "labels": LABELS,
        "test_accuracy": acc,
        "validation_accuracy": val_acc,
        "baseline_accuracy": baseline,
        "n_train_files": len(p_tr),
        "n_val_files": len(p_va),
        "n_test_files": len(p_te),
        "n_train_samples": int(len(Xtr)),
        "report": {l: {"precision": float(report[l]["precision"]), "recall": float(report[l]["recall"]),
                       "f1": float(report[l]["f1-score"]), "support": int(report[l]["support"])} for l in LABELS},
        "per_class_f1": {l: float(report[l]["f1-score"]) for l in LABELS},
        "confusion_matrix": cm.tolist(),
        "layer_val_accuracy": layer_acc,
        "search": search,
        "architecture": {
            "backbone": args.backbone, "backbone_label": short,
            "backbone_params": meta.get("n_params"), "backbone_layers": n_layers,
            "values_per_layer": int(Xtr.shape[2]), "layers": layers,
            "classifier_inputs": int(len(layers) * Xtr.shape[2]),
        },
        "training": {"classifier": "Logistic regression (L2)", "C": C, "augment": not args.no_augment,
                     "pooling": "mean + standard deviation over time, per layer"},
    }
    joblib.dump({"layers": layers, "pipeline": clf}, os.path.join(args.out, "ssl_head.joblib"))
    with open(os.path.join(args.out, "ssl_config.json"), "w") as fh:
        json.dump({"backbone": args.backbone, "sample_rate": sf.SSL_SR, "layers": layers, "labels": LABELS}, fh, indent=2)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(args.out, "labels.json"), "w") as fh:
        json.dump(LABELS, fh)
    try:
        save_figures(layer_acc, layers, cm, args.out)
    except Exception as e:  # noqa: BLE001
        print(f"(skipped figures: {e})")

    print(f"Test accuracy: {acc * 100:.2f}%" + (f"   (CNN baseline: {baseline * 100:.2f}%)" if baseline else ""))
    print(f"Saved to '{args.out}/'. Restart python app.py to use the new model.")


if __name__ == "__main__":
    main()
