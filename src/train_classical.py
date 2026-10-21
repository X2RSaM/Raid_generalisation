"""
train_classical.py - simple baseline and TF-IDF + logistic regression detectors.

For every (model, augmentation proportion, seed):
  1. Builds the training set: train_clean, with a proportion `p` of rows (per class)
     replaced by their seen-attack version from train_attacked.
  2. Tunes the regularisation strength C on the clean validation set (AUROC).
  3. Sets the decision threshold at 5% FPR on validation human texts.
  4. Evaluates on test_seen, test_unseen, test_reppen, test_seen_attacks and
     test_heldout_attacks (attack sets overall and per attack), and measures training time, inference speed
     and model size.

Models
  baseline         log(token count) + repeated-word ratio -> logistic regression
  tfidf_word       word 1-2 gram TF-IDF -> logistic regression
  tfidf_word_char  + character 3-5 gram TF-IDF (ablation: character features)

Usage (from the project root)
  python src/train_classical.py
  python src/train_classical.py --seeds 42 --aug 0      # quick first run
"""
import argparse
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from metrics import evaluate, threshold_at_fpr

C_GRID = [0.1, 1.0, 10.0]
TEST_SETS = ["test_seen", "test_unseen", "test_reppen",
             "test_seen_attacks", "test_heldout_attacks"]
ATTACK_SETS = {"test_seen_attacks", "test_heldout_attacks"}


# ---------------------------------------------------------------------------
# Feature extractors (each has fit_transform / transform)
# ---------------------------------------------------------------------------
class LengthRepetition:
    """Simple baseline features: log token count and share of repeated words."""

    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def _raw(texts):
        out = np.zeros((len(texts), 2))
        for i, t in enumerate(texts):
            words = t.split()
            n = len(words)
            out[i, 0] = np.log1p(n)
            out[i, 1] = 1 - len({w.lower() for w in words}) / n if n else 0.0
        return out

    def fit_transform(self, texts):
        return self.scaler.fit_transform(self._raw(texts))

    def transform(self, texts):
        return self.scaler.transform(self._raw(texts))


class Tfidf:
    """Word (and optionally character) n-gram TF-IDF features.

    Note: lowercase=True (the standard setting) also neutralises the upper_lower
    attack for these features - mention this when interpreting per-attack results.
    """

    def __init__(self, use_char):
        self.vecs = [TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=200_000, sublinear_tf=True)]
        if use_char:
            self.vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                             min_df=3, max_features=200_000,
                                             sublinear_tf=True))

    def fit_transform(self, texts):
        return sp.hstack([v.fit_transform(texts) for v in self.vecs]).tocsr()

    def transform(self, texts):
        return sp.hstack([v.transform(texts) for v in self.vecs]).tocsr()

    @property
    def n_features(self):
        return sum(len(v.vocabulary_) for v in self.vecs)


def make_features(name):
    if name == "baseline":
        return LengthRepetition()
    if name == "tfidf_word":
        return Tfidf(use_char=False)
    if name == "tfidf_word_char":
        return Tfidf(use_char=True)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_sets(data_dir):
    names = ["train_clean", "train_attacked", "val"] + TEST_SETS
    return {n: pd.read_parquet(Path(data_dir) / f"{n}.parquet") for n in names}


def augmented_train(clean, attacked, p, seed):
    """Replace a proportion p of clean rows (per class) with their attacked version."""
    if p == 0:
        return clean
    swap = clean.groupby("label", group_keys=False).sample(frac=p, random_state=seed)
    att = attacked.set_index("orig_id").loc[swap["id"]].reset_index(drop=True)
    kept = clean[~clean["id"].isin(swap["id"])]
    return pd.concat([kept, att], ignore_index=True).sample(frac=1, random_state=seed)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(model_name, p, seed, sets, pred_dir):
    train = augmented_train(sets["train_clean"], sets["train_attacked"], p, seed)
    val = sets["val"]

    # Fit features once, then tune C on validation AUROC
    feats = make_features(model_name)
    t0 = time.perf_counter()
    X_tr = feats.fit_transform(train["text"].tolist())
    feat_time = time.perf_counter() - t0
    X_val = feats.transform(val["text"].tolist())

    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000,
                                 random_state=seed)
        t0 = time.perf_counter()
        clf.fit(X_tr, train["label"])
        fit_time = time.perf_counter() - t0
        val_auc = roc_auc_score(val["label"], clf.predict_proba(X_val)[:, 1])
        if best is None or val_auc > best["val_auroc"]:
            best = {"C": C, "clf": clf, "val_auroc": val_auc, "fit_time": fit_time}
    clf = best["clf"]

    # Threshold at 5% FPR on validation human texts
    val_scores = clf.predict_proba(X_val)[:, 1]
    thr = threshold_at_fpr(val_scores[val["label"].values == 0], 0.05)

    rows = []
    base = {"model": model_name, "aug": p, "seed": seed, "C": best["C"],
            "val_auroc": best["val_auroc"], "threshold": thr}
    for name in TEST_SETS:
        df = sets[name]
        t0 = time.perf_counter()
        scores = clf.predict_proba(feats.transform(df["text"].tolist()))[:, 1]
        infer_time = time.perf_counter() - t0

        rows.append({**base, "condition": name, "attack": "all",
                     **evaluate(scores, df["label"], thr)})
        if name in ATTACK_SETS:
            for atk, idx in df.groupby("attack").groups.items():
                rows.append({**base, "condition": name, "attack": atk,
                             **evaluate(scores[df.index.get_indexer(idx)],
                                        df.loc[idx, "label"], thr)})
        if name == "test_seen":
            efficiency = {
                **base,
                "train_time_s": feat_time + best["fit_time"],
                "infer_texts_per_s": len(df) / infer_time,
                "model_size_mb": len(pickle.dumps((feats, clf))) / 1e6,
                "n_features": getattr(feats, "n_features", X_tr.shape[1]),
            }
        # Save per-text predictions for error analysis (first seed only)
        if pred_dir is not None:
            out = df[["id", "label", "model", "domain", "attack"]].copy()
            out["score"] = scores
            out["pred"] = (scores > thr).astype(int)
            out.to_parquet(pred_dir / f"{model_name}_aug{p}_seed{seed}_{name}.parquet",
                           index=False)
    return rows, efficiency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+",
                    default=["baseline", "tfidf_word", "tfidf_word_char"])
    ap.add_argument("--aug", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    out = Path(args.out)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "hardware_classical.json").write_text(json.dumps({
        "platform": platform.platform(), "processor": platform.processor(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "python": platform.python_version()}, indent=1))

    sets = load_sets(args.data)
    all_rows, all_eff = [], []
    for model_name in args.models:
        augs = [0.0] if model_name == "baseline" else args.aug  # proposal: no aug for baseline
        for p in augs:
            for seed in args.seeds:
                t0 = time.time()
                save_preds = pred_dir if seed == args.seeds[0] else None
                rows, eff = run_one(model_name, p, seed, sets, save_preds)
                all_rows += rows
                all_eff.append(eff)
                seen = next(r for r in rows if r["condition"] == "test_seen")
                unseen = next(r for r in rows if r["condition"] == "test_unseen")
                print(f"{model_name:<16} aug={p:<4} seed={seed}  "
                      f"TPR@5%FPR seen={seen['tpr_at_5fpr']:.3f} "
                      f"unseen={unseen['tpr_at_5fpr']:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

                # write after every run so progress is never lost
                pd.DataFrame(all_rows).to_csv(out / "classical_results.csv", index=False)
                pd.DataFrame(all_eff).to_csv(out / "classical_efficiency.csv", index=False)

    # Summary: mean +- std over seeds
    res = pd.DataFrame(all_rows)
    summary = (res.groupby(["model", "aug", "condition", "attack"])
               [["tpr_at_5fpr", "fpr_observed", "auroc", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out / "classical_summary.csv")
    main_view = res[res["attack"] == "all"].pivot_table(
        index=["model", "aug"], columns="condition", values="tpr_at_5fpr", aggfunc="mean")
    print("\nMean TPR at 5% FPR (over seeds):")
    print(main_view.round(3).to_string())
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()