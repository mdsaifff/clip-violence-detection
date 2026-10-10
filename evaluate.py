"""Measure the detector on the Real Life Violence Situations dataset (1,000 violent + 1,000 non-violent clips).

    python evaluate.py --data-dir "path/to/Real Life Violence Dataset"    # folder with Violence/ and NonViolence/

Frame embeddings are cached in results/embeddings.npz, so re-runs only redo the cheap part.
"""

import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from detector import PROMPTS, SINGLE_PROMPT, ViolenceDetector, even_indices, read_frames

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS, ASSETS = os.path.join(ROOT, "results"), os.path.join(ROOT, "assets")
FRAMES_PER_CLIP = 8
SEED = 0

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def clips(data_dir):
    """(path, label) for every clip; label 1 = violent. Folder names are matched case-insensitively."""
    found = []
    for folder in os.listdir(data_dir):
        name = folder.lower().replace("-", "").replace("_", "")
        label = 1 if name == "violence" else 0 if name == "nonviolence" else None
        if label is not None:
            found += [(p, label) for p in sorted(glob.glob(os.path.join(data_dir, folder, "*")))]
    if not found:
        raise SystemExit(f"no Violence/ or NonViolence/ folder in {data_dir}")
    return found


def embed_dataset(detector, data_dir, cache):
    if os.path.exists(cache):
        data = np.load(cache)
        return data["embeddings"], data["labels"]
    embeddings, labels = [], []
    items = clips(data_dir)
    for i, (path, label) in enumerate(items, 1):
        capture = cv2.VideoCapture(path)
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        frames, _ = read_frames(path, even_indices(n, FRAMES_PER_CLIP))
        if len(frames) != FRAMES_PER_CLIP:
            print(f"skipping {os.path.basename(path)}: {len(frames)} readable frames")
            continue
        embeddings.append(detector.embed_images(frames))
        labels.append(label)
        if i % 100 == 0:
            print(f"{i}/{len(items)} clips embedded")
    embeddings, labels = np.stack(embeddings).astype(np.float16), np.array(labels)
    np.savez_compressed(cache, embeddings=embeddings, labels=labels)
    return embeddings, labels


def scores(name, y, p):
    return {"method": name, "auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5),
            "f1": f1_score(y, p >= 0.5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)

    detector = ViolenceDetector()
    frames, y = embed_dataset(detector, args.data_dir, os.path.join(RESULTS, "embeddings.npz"))
    frames = frames.astype(np.float32)                         # (clips, frames, dim), unit-length rows
    middle = frames[:, FRAMES_PER_CLIP // 2]
    print(f"{len(y)} clips: {y.sum()} violent, {(1 - y).sum()} non-violent")

    rows = []
    for prompt_name, prompts in (("single prompt", SINGLE_PROMPT), ("prompt ensemble", PROMPTS)):
        classes = detector.embed_prompts(prompts)
        rows.append(scores(f"zero-shot, {prompt_name}, 1 frame", y, detector.violence_probability(middle, classes)))
        per_frame = detector.violence_probability(frames.reshape(-1, frames.shape[-1]), classes)
        rows.append(scores(f"zero-shot, {prompt_name}, {FRAMES_PER_CLIP}-frame mean", y,
                           per_frame.reshape(len(y), -1).mean(axis=1)))

    clip_vectors = frames.mean(axis=1)                          # one vector per clip
    probe = LogisticRegression(C=1.0, max_iter=5000)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows.append(scores("linear probe (5-fold CV)", y, cross_val_predict(probe, clip_vectors, y, cv=cv,
                                                                       method="predict_proba")[:, 1]))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False, float_format="%.3f")
    print(table.to_string(index=False))

    # how many labelled clips does the probe need to beat zero-shot?
    rng, curve = np.random.default_rng(SEED), []
    for train, test in cv.split(clip_vectors, y):
        for per_class in (5, 10, 25, 50, 100, 200, 400):
            pick = np.concatenate([rng.choice(train[y[train] == c], per_class, replace=False) for c in (0, 1)])
            p = probe.fit(clip_vectors[pick], y[pick]).predict_proba(clip_vectors[test])[:, 1]
            curve.append({"clips_per_class": per_class, "auc": roc_auc_score(y[test], p)})
    curve = pd.DataFrame(curve).groupby("clips_per_class", as_index=False).auc.mean()
    curve.to_csv(os.path.join(RESULTS, "probe_curve.csv"), index=False, float_format="%.3f")
    plot(table, curve)


def plot(table, curve):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(table))[::-1]
    left.barh(y, table["auc"], height=0.55, color=BLUE)
    for yi, value in zip(y, table["auc"]):
        left.text(value + 0.01, yi, f"{value:.3f}", va="center", color=MUTED, fontsize=9)
    left.set_yticks(y, table["method"])
    left.set_xlim(0.5, 1.02)
    left.grid(axis="y", visible=False)
    left.set_xlabel("ROC-AUC")
    left.set_title("Video-level results on RLVS (2,000 clips)", loc="left", fontsize=11)

    zero_shot = table.loc[table["method"].str.startswith("zero-shot"), "auc"].max()
    right.plot(curve.clips_per_class, curve.auc, color=BLUE, linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=2, label="linear probe")
    right.axhline(zero_shot, color=ORANGE, linewidth=2, label="best zero-shot")
    right.set_xscale("log")
    right.set_xlabel("labelled clips per class (log scale)")
    right.set_ylabel("ROC-AUC")
    right.legend(frameon=False, loc="lower right")
    right.set_title("Labels needed to beat zero-shot", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "results.png"), dpi=150)


if __name__ == "__main__":
    main()
