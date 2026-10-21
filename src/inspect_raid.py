"""
Quick completeness check for the RAID training file (train.csv).
Reads in chunks so a multi-GB file doesn't exhaust memory.

Usage:  python inspect_raid.py path/to/train.csv
"""
import sys
from collections import Counter

import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
CHUNK = 200_000

# 1. Peek at structure first - confirm column names before trusting anything else
head = pd.read_csv(PATH, nrows=5)
print("Columns:", list(head.columns))
print(head.head(3).T, "\n")

# Columns we'd like to profile (only those that actually exist are used)
wanted = ["model", "domain", "attack", "decoding", "repetition_penalty"]
cols = [c for c in wanted if c in head.columns]
text_col = "generation" if "generation" in head.columns else None
print("Profiling:", cols, "| text column:", text_col, "\n")

counts = {c: Counter() for c in cols}
model_domain = Counter()
total_rows = 0
missing_text = 0
lengths = []

for chunk in pd.read_csv(PATH, chunksize=CHUNK, usecols=cols + ([text_col] if text_col else [])):
    total_rows += len(chunk)
    for c in cols:
        counts[c].update(chunk[c].fillna("MISSING").astype(str))
    if "model" in cols and "domain" in cols:
        model_domain.update(zip(chunk["model"].astype(str), chunk["domain"].astype(str)))
    if text_col:
        txt = chunk[text_col]
        missing_text += int(txt.isna().sum() + (txt.astype(str).str.strip() == "").sum())
        lengths.extend(txt.dropna().astype(str).str.split().str.len().sample(
            min(1000, txt.notna().sum()), random_state=0).tolist())
    print(f"  ...read {total_rows:,} rows", end="\r")

print(f"\nTotal rows: {total_rows:,}")
print(f"Missing/empty texts: {missing_text:,}")
if lengths:
    s = pd.Series(lengths)
    print(f"Word length (sample): median={s.median():.0f}, 5th pct={s.quantile(.05):.0f}, 95th pct={s.quantile(.95):.0f}")

for c in cols:
    print(f"\n=== {c} ({len(counts[c])} values) ===")
    for k, v in counts[c].most_common():
        print(f"  {k:<30} {v:>12,}")

if model_domain:
    md = pd.Series(model_domain).unstack(fill_value=0)
    print("\n=== Rows per model x domain ===")
    print(md.to_string())
    md.to_csv("raid_model_by_domain.csv")
    print("\nSaved model x domain table to raid_model_by_domain.csv")