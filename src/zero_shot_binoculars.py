"""
zero_shot_binoculars.py - zero-shot Binoculars detector (Hans et al., ICML 2024). GPU.

No training at all. Two related LLMs read each text:
  * log-perplexity:  how surprising the text is to the "performer" LLM
  * cross-perplexity: how surprising the performer's predictions are to the
                      "observer" LLM (a measure of how predictable text is in general)
Binoculars score = log-perplexity / cross-perplexity. LOW score = likely AI-written.

The original paper uses Falcon-7B (observer) and Falcon-7B-instruct (performer).
To fit a free Kaggle T4, this uses a smaller pair with a shared tokenizer:
Qwen2.5-1.5B (observer) and Qwen2.5-1.5B-Instruct (performer). With two GPUs
(Kaggle "T4 x2") each model gets its own GPU.

For a fair comparison with the trained detectors, the decision threshold is set
the same way: at 5% FPR on the validation human texts. Every generator is unseen
for this detector, because it never trains on any.

Scores are cached per set in <out>/zeroshot_scores/, so a restart skips finished sets.

Usage (from the project root)
  python src/zero_shot_binoculars.py --limit 200 --out results/smoke_binoculars   # smoke test
  python src/zero_shot_binoculars.py                                              # full run
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import evaluate, threshold_at_fpr
from train_classical import ATTACK_SETS, TEST_SETS, load_sets
from train_roberta import limit_sets, pick_device, sync, tokenize_all

KEY = "binoculars"  # model name used in results and prediction files


def dtype_kwargs(dtype):
    """`dtype=` in new transformers, `torch_dtype=` in older versions."""
    major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    return {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype}


def pick_devices():
    """Observer and performer on separate GPUs when two are available."""
    if torch.cuda.device_count() >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    d = pick_device()
    if d.type == "cuda":
        d = torch.device("cuda:0")
    return d, d


def load_lm(name, device, fp32=False):
    dtype = torch.float16 if device.type == "cuda" and not fp32 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, **dtype_kwargs(dtype))
    return model.to(device).eval()


@torch.no_grad()
def binoculars(seqs, observer, performer, dev_o, dev_p, batch_size, pad_id, log_every):
    """Binoculars score for each token sequence (same formula as the official code)."""
    scores = np.full(len(seqs), np.nan)
    order = np.argsort([len(s) for s in seqs])  # similar lengths together = less padding
    n_batches = -(-len(seqs) // batch_size)
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, len(seqs), batch_size)):
        idx = order[start:start + batch_size]
        width = max(len(seqs[i]) for i in idx)
        x = np.full((len(idx), width), pad_id, dtype=np.int64)
        m = np.zeros((len(idx), width), dtype=np.int64)
        for row, i in enumerate(idx):
            x[row, :len(seqs[i])] = seqs[i]
            m[row, :len(seqs[i])] = 1
        xo, mo = torch.as_tensor(x, device=dev_o), torch.as_tensor(m, device=dev_o)
        lo = observer(input_ids=xo, attention_mask=mo).logits
        lp = performer(input_ids=xo.to(dev_p), attention_mask=mo.to(dev_p)).logits.to(dev_o)
        if lo.shape[-1] != lp.shape[-1]:
            raise ValueError("Observer and performer must share the same tokenizer/vocabulary.")

        batch_scores = []
        for row, i in enumerate(idx):  # one row at a time keeps memory low (vocab is large)
            n = len(seqs[i])
            if n < 2:
                batch_scores.append(torch.tensor(float("nan"), device=dev_o))
                continue
            perf_logp = torch.log_softmax(lp[row, :n].float(), dim=-1)
            # log-perplexity of the text under the performer (next-token prediction)
            log_ppl = -perf_logp[:-1].gather(-1, xo[row, 1:n].unsqueeze(-1)).mean()
            # cross-perplexity: observer's distribution vs performer's log-probabilities
            obs_p = torch.softmax(lo[row, :n].float(), dim=-1)
            x_ppl = -(obs_p * perf_logp).sum(-1).mean()
            batch_scores.append(log_ppl / x_ppl)
            del perf_logp, obs_p
        scores[idx] = torch.stack(batch_scores).cpu().numpy()
        del lo, lp

        if (b + 1) % log_every == 0 or b + 1 == n_batches:
            done_s = time.perf_counter() - t0
            eta = done_s / (b + 1) * (n_batches - b - 1) / 60
            print(f"      batch {b + 1}/{n_batches}  {(start + len(idx)) / done_s:.1f} texts/s  "
                  f"ETA {eta:.0f} min", flush=True)
    sync(dev_o)
    sync(dev_p)
    return scores, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--observer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--performer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="subsample sets (smoke test)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--fp32", action="store_true",
                    help="full precision (use if many scores come out as NaN in fp16)")
    args = ap.parse_args()

    dev_o, dev_p = pick_devices()
    print(f"Observer on {dev_o}, performer on {dev_p}")
    out = Path(args.out)
    pred_dir, cache_dir = out / "predictions", out / "zeroshot_scores"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    timing_path = cache_dir / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    sets = load_sets(args.data)
    sets.pop("train_clean")      # zero-shot: no training data needed
    sets.pop("train_attacked")
    if args.limit:
        sets = limit_sets(sets, args.limit)

    print("Tokenizing...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.observer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cache = tokenize_all(tokenizer, sets, args.max_len)
    print(f"  {len(cache):,} texts tokenized", flush=True)

    observer = performer = None
    raw = {}
    for name in ["val"] + TEST_SETS:
        path = cache_dir / f"{name}.parquet"
        if path.exists():
            raw[name] = pd.read_parquet(path).set_index("id")["binoculars"].reindex(
                sets[name]["id"]).to_numpy()
            print(f"{name}: loaded cached scores")
            continue
        if observer is None:
            print("Loading models...", flush=True)
            observer = load_lm(args.observer, dev_o, args.fp32)
            performer = load_lm(args.performer, dev_p, args.fp32)
        print(f"{name}: scoring {len(sets[name]):,} texts", flush=True)
        seqs = [cache[i] for i in sets[name]["id"]]
        scores, secs = binoculars(seqs, observer, performer, dev_o, dev_p,
                                  args.batch, pad_id, args.log_every)
        n_nan = int(np.isnan(scores).sum())
        if n_nan > 0.01 * len(scores):
            print(f"  WARNING: {n_nan} NaN scores - consider re-running with --fp32")
        raw[name] = scores
        timing[name] = {"seconds": secs, "texts": len(seqs)}
        timing_path.write_text(json.dumps(timing, indent=1))
        pd.DataFrame({"id": sets[name]["id"], "binoculars": scores}).to_parquet(path, index=False)

    # Higher = more likely AI, so the AI score is the negative Binoculars score
    # (the rare text with fewer than 2 tokens gets the median, i.e. a neutral score)
    ai = {name: -np.nan_to_num(s, nan=np.nanmedian(s)) for name, s in raw.items()}
    val = sets["val"]
    val_auc = roc_auc_score(val["label"], ai["val"])
    thr = threshold_at_fpr(ai["val"][val["label"].to_numpy() == 0], 0.05)
    print(f"Validation AUROC {val_auc:.4f}, threshold (AI score) {thr:.4f}")

    base = {"model": KEY, "aug": 0.0, "seed": 42, "val_auroc": val_auc, "threshold": thr}
    rows = []
    for name in TEST_SETS:
        df, scores = sets[name], ai[name]
        rows.append({**base, "condition": name, "attack": "all",
                     "truncated_share": float(np.mean([len(cache[i]) >= args.max_len
                                                       for i in df["id"]])),
                     **evaluate(scores, df["label"], thr)})
        if name in ATTACK_SETS:
            for atk, idx in df.groupby("attack").groups.items():
                pos = df.index.get_indexer(idx)
                rows.append({**base, "condition": name, "attack": atk,
                             **evaluate(scores[pos], df.loc[idx, "label"], thr)})
        pred = df[["id", "label", "model", "domain", "attack"]].copy()
        pred["score"] = scores
        pred["binoculars"] = raw[name]
        pred["pred"] = (scores > thr).astype(int)
        pred.to_parquet(pred_dir / f"{KEY}_aug0.0_seed42_{name}.parquet", index=False)

    if observer is None:  # everything came from the cache - still record model sizes
        observer = AutoModelForCausalLM.from_pretrained(args.observer, **dtype_kwargs(torch.float16))
        performer = AutoModelForCausalLM.from_pretrained(args.performer, **dtype_kwargs(torch.float16))
    params = [t for mdl in (observer, performer) for t in mdl.parameters()]
    t_seen = timing.get("test_seen", {})
    efficiency = {**base, "train_time_s": 0.0,
                  "infer_texts_per_s": (t_seen["texts"] / t_seen["seconds"]
                                        if t_seen else np.nan),
                  "model_size_mb": sum(t.numel() * t.element_size() for t in params) / 1e6,
                  "n_params": sum(t.numel() for t in params),
                  "observer": args.observer, "performer": args.performer}

    pd.DataFrame(rows).to_csv(out / "zeroshot_results.csv", index=False)
    pd.DataFrame([efficiency]).to_csv(out / "zeroshot_efficiency.csv", index=False)
    gpus = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [platform.processor() or platform.machine()])
    (out / "hardware_zeroshot.json").write_text(json.dumps({
        "device": dev_o.type, "gpu": " + ".join(gpus), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "platform": platform.platform(), **vars(args)}, indent=1))

    res = pd.DataFrame(rows)
    print("\nTPR at 5% FPR / observed FPR / AUROC:")
    print(res[["condition", "attack", "tpr_at_5fpr", "fpr_observed", "auroc"]]
          .round(3).to_string(index=False))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()