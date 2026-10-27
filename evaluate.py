"""
Evidence that the trained model works, for a report or a live demo.

  python evaluate.py --data data/ravdess

What it does:
  1. Re-scores the saved model on the held-out test features from training
     (reproduces the test accuracy printed by train.py).
  2. Runs the exact pipeline the web app uses (load -> cut 2.5 s -> 162 features -> CNN)
     on every held-out clip and reports that accuracy and confusion matrix.
  3. Copies held-out clips into demo_clips/, named with their true emotion, for uploading
     during a demo. The model never saw these clips while training.

Writes models/evaluation.txt with the results.
"""
import os as _os
_os.environ.setdefault("NUMBA_CACHE_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".numba_cache"))
import argparse
import os
import shutil
import time

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from train import find_ravdess_files, LABELS


def held_out_files(args):
    listed = os.path.join(args.models, "test_files.txt")
    if os.path.exists(listed):
        rows = [ln.rstrip("\n").split("\t") for ln in open(listed) if ln.strip()]
        return [r[1] for r in rows], [r[0] for r in rows], "models/test_files.txt"
    paths, labels = find_ravdess_files(args.data)
    if not paths:
        raise SystemExit(f"No RAVDESS files under '{args.data}'.")
    _, p_te, _, l_te = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=args.seed)
    return p_te, l_te, "the same split as train.py (seed 42)"


def table(cm, labels):
    w = max(len(l) for l in labels)
    lines = [" " * (w + 2) + " ".join(f"{l[:4]:>4}" for l in labels) + "   recall"]
    for i, l in enumerate(labels):
        row = cm[i]
        rec = row[i] / row.sum() if row.sum() else 0
        lines.append(f"{l:>{w}}  " + " ".join(f"{v:>4}" for v in row) + f"   {rec * 100:5.1f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo-dir", default="demo_clips")
    ap.add_argument("--per-emotion", type=int, default=2, help="held-out clips per emotion to copy for demos")
    args = ap.parse_args()

    from predict import EmotionPredictor
    pred = EmotionPredictor(args.models)
    out = []
    say = lambda s="": (print(s, flush=True), out.append(s))

    p_te, l_te, source = held_out_files(args)
    say(f"Held-out test clips: {len(p_te)} (from {source}). None of these were used for training.\n")

    # 1) saved test features -> should reproduce the training script's number
    cache = os.path.join(args.models, "features_cache.npz")
    if pred.kind == "ssl":
        m = pred.metrics
        say(f"1) Model: {m.get('model_label', 'WavLM')}. Test accuracy reported by train_ssl.py: "
            f"{m.get('test_accuracy', float('nan')) * 100:.1f}%"
            + (f" (CNN baseline on the same clips: {m['baseline_accuracy'] * 100:.1f}%)" if m.get("baseline_accuracy") else ""))
        say()
    elif os.path.exists(cache):
        c = np.load(cache, allow_pickle=False)
        X_te, y_te = c["X_te"], c["y_te"]
        same = len(y_te) == len(l_te) and all(a == b for a, b in zip(y_te, l_te))
        Xs = pred.scaler.transform(X_te)[..., np.newaxis]
        yhat = np.argmax(pred.model.predict(Xs, verbose=0), axis=1)
        acc = float(np.mean([pred.labels[i] == t for i, t in zip(yhat, y_te)]))
        say(f"1) Saved model on the training-time test features: {acc * 100:.1f}% accuracy"
            f" (train.py reported {pred.metrics.get('test_accuracy', float('nan')) * 100:.1f}%)")
        if not same:
            say("   Note: the file order differs from training, so steps 2-3 may include a few training clips.")
        say()

    # 2) the web app's pipeline on every held-out clip
    say("2) Web-app pipeline on every held-out clip ...")
    t, preds = time.time(), []
    for i, p in enumerate(p_te, 1):
        preds.append(pred.predict(p)["emotion"])
        if i % 50 == 0 or i == len(p_te):
            print(f"   {i}/{len(p_te)} clips, {time.time() - t:.0f}s", flush=True)
    correct = sum(a == b for a, b in zip(preds, l_te))
    say(f"   Accuracy: {correct}/{len(p_te)} = {correct / len(p_te) * 100:.1f}%   (random guessing: 12.5%)\n")
    say("   Confusion matrix (rows = true emotion, columns = predicted):")
    say(table(confusion_matrix(l_te, preds, labels=LABELS), LABELS))
    say()

    # 3) demo clips
    os.makedirs(args.demo_dir, exist_ok=True)
    say(f"3) Demo clips copied to {args.demo_dir}/ (upload these in the app; the true label is in the name):")
    n = 0
    for k, emo in enumerate(LABELS, 1):
        picked = [(p, pr) for p, l, pr in zip(p_te, l_te, preds) if l == emo][: args.per_emotion]
        for p, pr in picked:
            name = f"{k:02d}_{emo}__{os.path.basename(p)}"
            shutil.copy(p, os.path.join(args.demo_dir, name))
            mark = "correct" if pr == emo else f"predicted {pr}"
            say(f"   {name:<48} {mark}")
            n += 1
    say(f"   {n} clips copied.")

    with open(os.path.join(args.models, "evaluation.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nSaved this report to {os.path.join(args.models, 'evaluation.txt')}")


if __name__ == "__main__":
    main()
