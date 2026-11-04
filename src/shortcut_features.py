"""
shortcut_features.py - what surface clues separate human and AI text in the training data?

Part A (pattern recognition): fits TF-IDF (word + character n-grams) + logistic
regression on train_clean and lists the clues with the largest weights towards
"AI" and towards "human".
Part B (formatting): compares simple formatting statistics (line breaks, blank lines,
non-ASCII characters, lists, length) between humans and each AI generator.

If the strongest clues are formatting or boilerplate rather than writing style,
very high detection scores may partly rest on shortcuts.

Usage (from the project root)
  python src/shortcut_features.py                                   # original data
  python src/shortcut_features.py --data data/processed_norm --tag normalized
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from train_classical import Tfidf


def chosen_c(res_dir):
    """C picked on validation for TF-IDF (word+char), seed 42, no augmentation."""
    path = Path(res_dir) / "classical_results.csv"
    if path.exists():
        r = pd.read_csv(path)
        r = r[(r["model"] == "tfidf_word_char") & (r["aug"] == 0) & (r["seed"] == 42)]
        if len(r):
            return float(r["C"].iloc[0])
    return 1.0


def top_features(train, C, n):
    feats = Tfidf(use_char=True)
    X = feats.fit_transform(train["text"].tolist())
    clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=42)
    clf.fit(X, train["label"])
    names = np.concatenate([
        [f"word: {t}" for t in feats.vecs[0].get_feature_names_out()],
        [f"char: {t!r}" for t in feats.vecs[1].get_feature_names_out()]])
    w = clf.coef_[0]
    order = np.argsort(w)
    rows = [{"direction": "AI", "rank": i + 1, "feature": names[j], "weight": round(w[j], 3)}
            for i, j in enumerate(order[::-1][:n])]
    rows += [{"direction": "human", "rank": i + 1, "feature": names[j], "weight": round(w[j], 3)}
             for i, j in enumerate(order[:n])]
    return pd.DataFrame(rows)


LIST_LINE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")


def formatting_stats(df):
    t = df["text"]
    words = t.str.split().str.len().clip(lower=1)
    return pd.DataFrame({
        "has_line_break": t.str.contains("\n", regex=False),
        "line_breaks_per_100_words": t.str.count("\n") / words * 100,
        "has_blank_line": t.str.contains("\n\n", regex=False),
        "has_list_lines": t.str.contains(LIST_LINE),
        "has_non_ascii": t.str.contains(r"[^\x00-\x7F]", regex=True),
        "words": words,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="original")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    tab = Path(args.results) / "tables"
    tab.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(Path(args.data) / "train_clean.parquet")

    # Part B: formatting by class and by generator
    stats = formatting_stats(train)
    stats["class"] = np.where(train["label"] == 1, "AI", "human")
    stats["source"] = train["model"]
    by_class = stats.groupby("class").mean(numeric_only=True)
    by_source = stats.groupby("source").mean(numeric_only=True)
    for df in (by_class, by_source):
        for c in ["has_line_break", "has_blank_line", "has_list_lines", "has_non_ascii"]:
            df[c] = (df[c] * 100).round(1)
        df["line_breaks_per_100_words"] = df["line_breaks_per_100_words"].round(2)
        df["words"] = df["words"].round(0)
    by_class.to_csv(tab / f"formatting_by_class_{args.tag}.csv")
    by_source.to_csv(tab / f"formatting_by_source_{args.tag}.csv")
    print("Formatting by class (% of texts, except per-100-words and words):")
    print(by_class.to_string(), "\n")
    print("Formatting by source:")
    print(by_source.to_string(), "\n")

    # Part A: top TF-IDF clues
    C = chosen_c(args.results)
    top = top_features(train, C, args.top)
    top.to_csv(tab / f"tfidf_top_features_{args.tag}.csv", index=False)
    print(f"Top TF-IDF clues (logistic regression C={C}):")
    for d in ["AI", "human"]:
        print(f"\n  towards {d}:")
        for _, r in top[top["direction"] == d].iterrows():
            print(f"    {r['rank']:>2}. {r['feature']:<40} {r['weight']:>8.3f}")
    print(f"\nSaved tables to {tab}/ (tag: {args.tag})")


if __name__ == "__main__":
    main()
