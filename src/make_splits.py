"""
make_splits.py - build leakage-free, class-balanced experiment splits from RAID.

What it does
  1. Splits the HUMAN source documents 70/15/15 (train/val/test), stratified by
     domain. Every AI text and attacked text inherits the split of its source
     document (via `source_id`), so no source document crosses splits.
  2. Builds balanced evaluation sets: one human text + one AI text per source doc.
  3. Saves attacked versions for the augmentation experiment (seen attacks, train)
     and for the robustness test (held-out attacks, test).
  4. Pulls the actual texts in a final pass and saves everything as Parquet.

Usage
  python make_splits.py --raid data/train.csv --out data/processed --seed 42

Requires: pandas, numpy, scikit-learn, pyarrow
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401  - fail fast here, not after a 15-minute run
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Experimental design (fixed before experimentation, as stated in the proposal)
# ---------------------------------------------------------------------------
SEEN_GENERATORS = ["gpt2", "gpt3", "chatgpt", "cohere", "cohere-chat",
                   "mistral", "mistral-chat"]
HELDOUT_GENERATORS = ["gpt4", "llama-chat", "mpt", "mpt-chat"]

SEEN_ATTACKS = ["article_deletion", "number", "paraphrase", "synonym",
                "perplexity_misspelling", "alternative_spelling"]
HELDOUT_ATTACKS = ["homoglyph", "zero_width_space", "whitespace",
                   "upper_lower", "insert_paragraphs"]

META_COLS = ["id", "adv_source_id", "source_id", "model", "decoding",
             "repetition_penalty", "attack", "domain"]
CHUNK = 500_000


def read_meta(path):
    """Stream the metadata columns only (no text) to keep memory low."""
    return pd.read_csv(path, usecols=META_COLS, chunksize=CHUNK, dtype=str)


def rep_mask(df, setting):
    """Repetition-penalty filter. Human/closed-model rows count as 'no'."""
    if setting == "no":
        return df["repetition_penalty"].isna() | (df["repetition_penalty"] == "no")
    return df["repetition_penalty"] == "yes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raid", default="data/train.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Pass 1: find clean human source documents and split them
    # -----------------------------------------------------------------------
    print("Pass 1/3: reading human source documents...")
    parts = []
    for c in read_meta(args.raid):
        parts.append(c[(c["model"] == "human") & (c["attack"] == "none")]
                     [["id", "source_id", "domain"]])
    humans = pd.concat(parts, ignore_index=True)
    print(f"  clean human documents: {len(humans):,}")
    if not (humans["id"] == humans["source_id"]).all():
        print("  WARNING: some human rows have id != source_id - check linking.")

    train_h, temp_h = train_test_split(
        humans, test_size=0.30, stratify=humans["domain"], random_state=seed)
    val_h, test_h = train_test_split(
        temp_h, test_size=0.50, stratify=temp_h["domain"], random_state=seed)

    split_of = {}
    for name, part in [("train", train_h), ("val", val_h), ("test", test_h)]:
        split_of.update(dict.fromkeys(part["source_id"], name))
    print(f"  source docs -> train {len(train_h):,} | val {len(val_h):,} | "
          f"test {len(test_h):,}")

    # -----------------------------------------------------------------------
    # Pass 2: keep only the metadata rows any experiment needs
    # -----------------------------------------------------------------------
    print("Pass 2/3: filtering metadata...")
    keep, orphans = [], 0
    for c in read_meta(args.raid):
        c["split"] = c["source_id"].map(split_of)
        orphans += int(c["split"].isna().sum())
        c = c[c["split"].notna()]

        is_human = c["model"] == "human"
        clean = c["attack"] == "none"
        seen_ai_no_pen = c["model"].isin(SEEN_GENERATORS) & rep_mask(c, "no")

        # train/val: humans + seen generators without penalty;
        # train additionally keeps seen-attack versions for augmentation
        trval = c["split"].isin(["train", "val"])
        m_trval = trval & (is_human | seen_ai_no_pen) & (
            clean | ((c["split"] == "train") & c["attack"].isin(SEEN_ATTACKS)))

        # test: all clean rows (seen, held-out, penalty conditions) +
        # seen- and held-out-attack versions of humans and seen generators
        test = c["split"] == "test"
        m_test = test & (clean | (c["attack"].isin(SEEN_ATTACKS + HELDOUT_ATTACKS)
                                  & (is_human | seen_ai_no_pen)))
        keep.append(c[m_trval | m_test])
    meta = pd.concat(keep, ignore_index=True)
    print(f"  kept {len(meta):,} metadata rows "
          f"(rows without a human source in this file: {orphans:,})")

    # -----------------------------------------------------------------------
    # Build the experiment sets (balanced: 1 human + 1 AI per source document)
    # -----------------------------------------------------------------------
    clean = meta[meta["attack"] == "none"]
    hum = clean[clean["model"] == "human"]
    ai = clean[clean["model"] != "human"]
    att = meta[meta["attack"] != "none"]

    def balanced(split, gens, penalty):
        h = hum[hum["split"] == split]
        a = ai[(ai["split"] == split) & ai["model"].isin(gens) & rep_mask(ai, penalty)]
        a = a.groupby("source_id", group_keys=False).sample(n=1, random_state=seed)
        return pd.concat([h, a], ignore_index=True)

    sets = {
        "train_clean": balanced("train", SEEN_GENERATORS, "no"),
        "val": balanced("val", SEEN_GENERATORS, "no"),
        "test_seen": balanced("test", SEEN_GENERATORS, "no"),
        "test_unseen": balanced("test", HELDOUT_GENERATORS, "no"),
        # penalty condition: seen open-source generators (gpt2, mistral, mistral-chat)
        "test_reppen": balanced("test", SEEN_GENERATORS, "yes"),
    }

    # Augmentation pool: one random SEEN-attack version of every train_clean row
    # (both human and AI, so "attacked" cannot become a label shortcut).
    # `orig_id` links each attacked row to the clean row it replaces.
    tr_att = att[att["adv_source_id"].isin(sets["train_clean"]["id"])
                 & att["attack"].isin(SEEN_ATTACKS)]
    tr_att = tr_att.groupby("adv_source_id", group_keys=False).sample(
        n=1, random_state=seed)
    sets["train_attacked"] = tr_att.rename(columns={"adv_source_id": "orig_id"})
    coverage = len(tr_att) / len(sets["train_clean"])
    print(f"  attacked versions found for {coverage:.1%} of train_clean rows")
    if coverage < 0.99:
        print("  WARNING: low coverage - check that adv_source_id links attacked "
              "rows to their clean original.")

    # Robustness tests: ALL attack versions of the test_seen rows.
    # test_seen_attacks   -> attacks used in augmentation (does augmentation work?)
    # test_heldout_attacks -> attacks never trained on (does robustness transfer?)
    in_test_seen = att["adv_source_id"].isin(sets["test_seen"]["id"])
    sets["test_seen_attacks"] = att[
        in_test_seen & att["attack"].isin(SEEN_ATTACKS)].reset_index(drop=True)
    sets["test_heldout_attacks"] = att[
        in_test_seen & att["attack"].isin(HELDOUT_ATTACKS)].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Pass 3: attach the texts
    # -----------------------------------------------------------------------
    print("Pass 3/3: collecting texts...")
    needed = set().union(*(set(df["id"]) for df in sets.values()))
    texts = {}
    for c in pd.read_csv(args.raid, usecols=["id", "generation"],
                         chunksize=CHUNK, dtype=str):
        sub = c[c["id"].isin(needed)]
        texts.update(zip(sub["id"], sub["generation"]))
    print(f"  texts collected: {len(texts):,} / {len(needed):,}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    summary = {}
    for name, df in sets.items():
        df = df.copy()
        df["text"] = df["id"].map(texts)
        df["label"] = (df["model"] != "human").astype(int)  # 1 = AI, 0 = human
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df.to_parquet(out / f"{name}.parquet", index=False)
        summary[name] = {"rows": len(df), "ai": int(df["label"].sum()),
                         "human": int((df["label"] == 0).sum()),
                         "missing_text": int(df["text"].isna().sum())}

    manifest = {
        "seed": seed,
        "seen_generators": SEEN_GENERATORS,
        "heldout_generators": HELDOUT_GENERATORS,
        "seen_attacks": SEEN_ATTACKS,
        "heldout_attacks": HELDOUT_ATTACKS,
        "summary": summary,
        "source_ids": {name: sorted(part["source_id"])
                       for name, part in [("train", train_h), ("val", val_h),
                                          ("test", test_h)]},
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))

    print("\nSaved sets:")
    for name, s in summary.items():
        print(f"  {name:<22} rows={s['rows']:>7,}  AI={s['ai']:>6,}  "
              f"human={s['human']:>6,}  missing_text={s['missing_text']}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {out}/")


if __name__ == "__main__":
    main()