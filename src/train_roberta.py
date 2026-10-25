"""
train_roberta.py - fine-tuned RoBERTa detector. Run on a GPU (Kaggle or Colab).

Runs on an NVIDIA GPU (fastest), an Apple Silicon Mac (MPS) or, very slowly, CPU.
Uses exactly the same splits, augmentation, threshold rule, metrics and test sets
as train_classical.py, so the results line up directly.

For every (augmentation proportion, learning rate, seed):
  1. Builds the training set (train_clean with proportion p swapped for attacked).
  2. Fine-tunes RoBERTa for up to --epochs, keeping the epoch with the best
     validation AUROC (clean validation set only).
  3. Sets the threshold at 5% FPR on validation human texts.
  4. Evaluates on all test sets (attack sets also per attack) and measures
     training time, inference speed and model size.

Results are written after every run, and finished runs are skipped on restart,
so a disconnected Colab/Kaggle session can simply be re-run.

Usage (from the project root)
  # 1) smoke test - a few minutes, checks everything works
  python src/train_roberta.py --limit 500 --seeds 42 --aug 0 --epochs 1 --out results/smoke

  # 2) learning-rate tuning on validation only (seed 42, no augmentation)
  python src/train_roberta.py --seeds 42 --aug 0 --lrs 1e-5 2e-5 3e-5 --tune-only --out results/tuning

  # 3) main experiments with the chosen learning rate
  python src/train_roberta.py --lrs 2e-5
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from metrics import evaluate, threshold_at_fpr
from train_classical import ATTACK_SETS, TEST_SETS, augmented_train, load_sets


def sync(device):
    """Wait for the GPU to finish, so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def pick_device():
    """NVIDIA GPU (Kaggle/Colab) > Apple Silicon GPU (MPS) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_sets(sets, n):
    """Small random subsets for a quick smoke test."""
    out = {}
    for name, df in sets.items():
        if name == "train_attacked":
            out[name] = df  # looked up by id, so keep complete
        else:
            out[name] = df.sample(n=min(n, len(df)), random_state=0).reset_index(drop=True)
    return out


def tokenize_all(tokenizer, sets, max_len):
    """Tokenize every unique text once; returns {text id: int32 token array}."""
    texts = pd.concat([df[["id", "text"]] for df in sets.values()]).drop_duplicates("id")
    cache = {}
    ids, docs = texts["id"].tolist(), texts["text"].tolist()
    for start in range(0, len(docs), 2000):
        enc = tokenizer(docs[start:start + 2000], truncation=True, max_length=max_len)
        for i, tok in zip(ids[start:start + 2000], enc["input_ids"]):
            cache[i] = np.asarray(tok, dtype=np.int32)
    return cache


def batches(seqs, labels, batch_size, pad_id, device, shuffle, rng=None):
    """Yield (positions, input_ids, attention_mask, labels) with dynamic padding.

    Inference batches are sorted by length, which makes prediction much faster.
    """
    n = len(seqs)
    order = rng.permutation(n) if shuffle else np.argsort([len(s) for s in seqs])
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        width = max(len(seqs[i]) for i in idx)
        x = np.full((len(idx), width), pad_id, dtype=np.int64)
        m = np.zeros((len(idx), width), dtype=np.int64)
        for row, i in enumerate(idx):
            x[row, :len(seqs[i])] = seqs[i]
            m[row, :len(seqs[i])] = 1
        y = torch.as_tensor(labels[idx], device=device) if labels is not None else None
        yield idx, torch.as_tensor(x, device=device), torch.as_tensor(m, device=device), y


@torch.no_grad()
def predict(model, seqs, batch_size, pad_id, device, amp):
    """Probability that each text is AI-generated, plus elapsed seconds."""
    model.eval()
    scores = np.zeros(len(seqs), dtype=np.float64)
    sync(device)
    t0 = time.perf_counter()
    for idx, x, m, _ in batches(seqs, None, batch_size, pad_id, device, shuffle=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(input_ids=x, attention_mask=m).logits
        scores[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    sync(device)
    return scores, time.perf_counter() - t0


def run_one(args, tokenizer, cache, sets, p, lr, seed, device, pred_dir):
    set_seed(seed)
    amp = device.type == "cuda"  # mixed precision on NVIDIA only; MPS/CPU use fp32
    pad_id = tokenizer.pad_token_id
    rng = np.random.default_rng(seed)

    train = augmented_train(sets["train_clean"], sets["train_attacked"], p, seed)
    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=seed)
    tr_seqs = [cache[i] for i in train["id"]]
    tr_labels = train["label"].to_numpy()
    val = sets["val"]
    val_seqs = [cache[i] for i in val["id"]]

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    updates_per_epoch = -(-len(tr_seqs) // (args.batch * args.accum))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_updates), total_updates)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_auc, best_state, best_epoch, train_time = -1.0, None, 0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, n_batches = 0.0, -(-len(tr_seqs) // args.batch)
        for step, (_, x, m, y) in enumerate(batches(
                tr_seqs, tr_labels, args.batch, pad_id, device, shuffle=True, rng=rng)):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (step + 1) % args.accum == 0 or step + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if (step + 1) % args.log_every == 0 or step + 1 == n_batches:
                done_s = time.perf_counter() - t0
                eta_min = done_s / (step + 1) * (n_batches - step - 1) / 60
                print(f"      epoch {epoch} batch {step + 1}/{n_batches}  "
                      f"loss {running / (step + 1):.4f}  "
                      f"{(step + 1) * args.batch / done_s:.1f} texts/s  "
                      f"epoch ETA {eta_min:.0f} min", flush=True)
        sync(device)
        train_time += time.perf_counter() - t0

        val_scores, _ = predict(model, val_seqs, args.eval_batch, pad_id, device, amp)
        val_auc = roc_auc_score(val["label"], val_scores)
        print(f"    epoch {epoch}: train loss {running / n_batches:.4f}  "
              f"val AUROC {val_auc:.4f}", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_scores, _ = predict(model, val_seqs, args.eval_batch, pad_id, device, amp)
    thr = threshold_at_fpr(val_scores[val["label"].to_numpy() == 0], 0.05)

    short = args.model_name.split("/")[-1]
    base = {"model": short, "aug": p, "seed": seed, "lr": lr, "best_epoch": best_epoch,
            "val_auroc": best_auc, "threshold": thr}
    if args.tune_only:  # tuning uses validation only - skip the test sets
        del model, optimizer, best_state
        return [], {**base, "train_time_s": train_time}
    rows, efficiency = [], None
    for name in TEST_SETS:
        df = sets[name]
        seqs = [cache[i] for i in df["id"]]
        scores, infer_time = predict(model, seqs, args.eval_batch, pad_id, device, amp)
        truncated = float(np.mean([len(s) >= args.max_len for s in seqs]))

        rows.append({**base, "condition": name, "attack": "all",
                     "truncated_share": truncated, **evaluate(scores, df["label"], thr)})
        if name in ATTACK_SETS:
            for atk, idx in df.groupby("attack").groups.items():
                pos = df.index.get_indexer(idx)
                rows.append({**base, "condition": name, "attack": atk,
                             **evaluate(scores[pos], df.loc[idx, "label"], thr)})
        if name == "test_seen":
            n_params = sum(t.numel() for t in model.parameters())
            efficiency = {**base, "train_time_s": train_time,
                          "infer_texts_per_s": len(df) / infer_time,
                          "model_size_mb": sum(t.numel() * t.element_size()
                                               for t in model.parameters()) / 1e6,
                          "n_params": n_params,
                          "train_truncated_share": float(np.mean(
                              [len(s) >= args.max_len for s in tr_seqs]))}
        if pred_dir is not None:
            out = df[["id", "label", "model", "domain", "attack"]].copy()
            out["score"] = scores
            out["pred"] = (scores > thr).astype(int)
            out.to_parquet(pred_dir / f"{short}_aug{p}_lr{lr}_seed{seed}_{name}.parquet",
                           index=False)

    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return rows, efficiency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-name", default="roberta-base")
    ap.add_argument("--aug", nargs="+", type=float, default=[0.0, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--lrs", nargs="+", type=float, default=[2e-5])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="subsample sets (smoke test)")
    ap.add_argument("--log-every", type=int, default=50, help="progress line every N batches")
    ap.add_argument("--tune-only", action="store_true",
                    help="train + validate only (for learning-rate tuning)")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU found - this will be extremely slow.")

    out = Path(args.out)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "hardware_roberta.json").write_text(json.dumps({
        "device": device.type,
        "gpu": (torch.cuda.get_device_name(0) if device.type == "cuda"
                else platform.processor() or platform.machine()),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), **vars(args)}, indent=1))

    res_path, eff_path = out / "roberta_results.csv", out / "roberta_efficiency.csv"
    all_rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    all_eff = pd.read_csv(eff_path).to_dict("records") if eff_path.exists() else []
    done = {(r["aug"], r["lr"], r["seed"]) for r in all_eff}

    sets = load_sets(args.data)
    if args.limit:
        sets = limit_sets(sets, args.limit)
    print("Tokenizing...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cache = tokenize_all(tokenizer, sets, args.max_len)
    print(f"  {len(cache):,} texts tokenized", flush=True)

    for p in args.aug:
        for lr in args.lrs:
            for seed in args.seeds:
                if (p, lr, seed) in done:
                    print(f"skip aug={p} lr={lr} seed={seed} (already done)")
                    continue
                print(f"Run aug={p} lr={lr} seed={seed}", flush=True)
                t0 = time.time()
                save_preds = pred_dir if seed == args.seeds[0] else None
                rows, eff = run_one(args, tokenizer, cache, sets, p, lr, seed,
                                    device, save_preds)
                all_rows += rows
                all_eff.append(eff)
                pd.DataFrame(all_rows).to_csv(res_path, index=False)
                pd.DataFrame(all_eff).to_csv(eff_path, index=False)
                if args.tune_only:
                    print(f"  val AUROC {eff['val_auroc']:.4f} (best epoch "
                          f"{eff['best_epoch']})  ({(time.time() - t0) / 60:.1f} min)",
                          flush=True)
                    continue
                get = {r["condition"]: r for r in rows if r["attack"] == "all"}
                print(f"  TPR@5%FPR seen={get['test_seen']['tpr_at_5fpr']:.3f} "
                      f"unseen={get['test_unseen']['tpr_at_5fpr']:.3f} "
                      f"seen_atk={get['test_seen_attacks']['tpr_at_5fpr']:.3f} "
                      f"heldout_atk={get['test_heldout_attacks']['tpr_at_5fpr']:.3f}  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    if args.tune_only or not all_rows:
        print("\nValidation AUROC per learning rate:")
        print(pd.DataFrame(all_eff).groupby("lr")["val_auroc"].mean().round(4).to_string())
        return
    res = pd.DataFrame(all_rows)
    summary = (res.groupby(["model", "aug", "lr", "condition", "attack"])
               [["tpr_at_5fpr", "fpr_observed", "auroc", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out / "roberta_summary.csv")
    view = res[res["attack"] == "all"].pivot_table(
        index=["aug", "lr"], columns="condition", values="tpr_at_5fpr", aggfunc="mean")
    print("\nMean TPR at 5% FPR (over seeds):")
    print(view.round(3).to_string())
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()