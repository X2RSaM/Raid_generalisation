"""
report_utils.py - shared helpers for the analysis notebooks (EDA, results, errors).

Loads the experiment outputs, applies one consistent figure style sized for the
IEEE two-column format, and writes tables as both CSV and LaTeX.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Names and experimental design (must match make_splits.py)
# ---------------------------------------------------------------------------
DETECTOR_NAMES = {"baseline": "Baseline", "tfidf_word": "TF-IDF (word)",
                  "tfidf_word_char": "TF-IDF (word+char)", "roberta-base": "RoBERTa",
                  "Qwen2.5-0.5B": "Qwen2.5 (LoRA)", "binoculars": "Binoculars (zero-shot)"}
DETECTOR_ORDER = list(DETECTOR_NAMES.values())
FIGURE_KEYS = ["tfidf_word_char", "roberta-base", "Qwen2.5-0.5B", "binoculars"]
TRAINED_KEYS = ["tfidf_word_char", "roberta-base", "Qwen2.5-0.5B"]  # have augmentation runs

CONDITION_NAMES = {"test_seen": "Seen gen.", "test_unseen": "Held-out gen.",
                   "test_reppen": "Rep. penalty", "test_seen_attacks": "Seen attacks",
                   "test_heldout_attacks": "Held-out attacks"}
CONDITION_ORDER = list(CONDITION_NAMES.values())

SEEN_GENERATORS = ["gpt2", "gpt3", "chatgpt", "cohere", "cohere-chat",
                   "mistral", "mistral-chat"]
HELDOUT_GENERATORS = ["gpt4", "llama-chat", "mpt", "mpt-chat"]
GENERATOR_TYPE = {"gpt2": "base", "gpt3": "base", "cohere": "base", "mistral": "base",
                  "mpt": "base", "chatgpt": "chat", "gpt4": "chat", "cohere-chat": "chat",
                  "mistral-chat": "chat", "mpt-chat": "chat", "llama-chat": "chat"}

SEEN_ATTACKS = ["article_deletion", "number", "paraphrase", "synonym",
                "perplexity_misspelling", "alternative_spelling"]
HELDOUT_ATTACKS = ["homoglyph", "zero_width_space", "whitespace",
                   "upper_lower", "insert_paragraphs"]

# ---------------------------------------------------------------------------
# Figure style (validated categorical colours; IEEE column widths in inches)
# ---------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Plotted detectors, in figure order, pass the adjacent-pair colour-blind check:
# blue, orange, violet, green. Baseline and TF-IDF (word) only appear in tables.
COLORS = {"TF-IDF (word+char)": "#2a78d6", "RoBERTa": "#eb6834",
          "Qwen2.5 (LoRA)": "#6250d6", "Binoculars (zero-shot)": "#008300",
          "Baseline": "#e87ba4", "TF-IDF (word)": "#eda100"}
HATCH = {"RoBERTa": "////", "Qwen2.5 (LoRA)": "\\\\\\\\", "Binoculars (zero-shot)": "...."}
MARKERS = {"TF-IDF (word+char)": "o", "RoBERTa": "s", "Qwen2.5 (LoRA)": "D",
           "Binoculars (zero-shot)": "v", "Baseline": "^", "TF-IDF (word)": "P"}
PAIR = ("#1baf7a", "#e34948")          # two non-detector series (seen/held-out, human/AI)
COL_W, PAGE_W = 3.5, 7.16


def set_style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def project_root():
    """Find the project folder (works from the root or from notebooks/)."""
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "src").is_dir() and (p / "results").is_dir():
            return p
    raise FileNotFoundError("Run the notebook from inside the project folder.")


def output_dirs(root):
    fig, tab = root / "results" / "figures", root / "results" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab


def save_fig(fig, name, fig_dir):
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    print(f"saved {fig_dir.name}/{name}.pdf and .png")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def pm(mean, std, scale=100, digits=1):
    """'99.5 ± 0.1' (std omitted when there is only one run)."""
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def _tex(value):
    s = str(value)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("±", r"$\pm$")]:
        s = s.replace(a, b)
    return s


def save_table(df, name, tab_dir, caption="", index=True):
    """Write a table as CSV (for checking) and as a LaTeX table (for the report)."""
    df.to_csv(tab_dir / f"{name}.csv", index=index)
    data = df.reset_index() if index else df
    cols = list(data.columns)
    lines = [r"\begin{table}[!t]", r"\centering",
             rf"\caption{{{_tex(caption)}}}", rf"\label{{tab:{name}}}", r"\footnotesize",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(rf"\textbf{{{_tex(c)}}}" for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex(v) for v in row) + r" \\" for row in data.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tab_dir / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"saved {tab_dir.name}/{name}.csv and .tex")
    return df


# ---------------------------------------------------------------------------
# Results and predictions
# ---------------------------------------------------------------------------
def chosen_lr(eff):
    """Learning rate with the best validation AUROC (seed 42, no augmentation)."""
    tune = eff[(eff["aug"] == 0) & (eff["seed"] == 42)]
    return float(tune.loc[tune["val_auroc"].idxmax(), "lr"])


def tuning_runs(res_dir, prefix):
    """Learning-rate tuning runs: results/tuning/ if it exists, else the main file."""
    for path in [res_dir / "tuning" / f"{prefix}_efficiency.csv",
                 res_dir / f"{prefix}_efficiency.csv"]:
        if path.exists():
            return pd.read_csv(path)
    return None


def load_results(res_dir):
    """All detectors' results; neural models restricted to the validation-chosen lr.

    Returns (results, lrs) where lrs maps model key -> chosen learning rate.
    Missing result files (e.g. before the LLM runs) are simply skipped.
    """
    parts, lrs = [pd.read_csv(res_dir / "classical_results.csv")], {}
    for prefix in ["roberta", "llm"]:
        path = res_dir / f"{prefix}_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        lr = chosen_lr(tuning_runs(res_dir, prefix))
        if not np.isclose(df["lr"], lr).any():   # chosen lr not run yet: use what exists
            lr = float(df["lr"].mode()[0])
        df = df[np.isclose(df["lr"], lr)]
        lrs.update({m: lr for m in df["model"].unique()})
        parts.append(df)
    if (res_dir / "zeroshot_results.csv").exists():
        parts.append(pd.read_csv(res_dir / "zeroshot_results.csv"))
    df = pd.concat(parts, ignore_index=True)
    df["detector"] = df["model"].map(DETECTOR_NAMES).fillna(df["model"])
    df["condition_name"] = df["condition"].map(CONDITION_NAMES)
    return df, lrs


def detector_order(df):
    """Detectors present in the results, in report order."""
    present = set(df["detector"])
    return [d for d in DETECTOR_ORDER if d in present]


def keys_present(df, keys):
    return [k for k in keys if k in set(df["model"])]


def summarise(df, by, metric):
    """Mean and std over seeds."""
    return df.groupby(by)[metric].agg(["mean", "std"])


def pred_file(res_dir, key, condition, aug=0.0, seed=42, lr=None):
    """Path of a per-text prediction file written by the training scripts.

    Neural models include the learning rate in the name (pass lr=lrs[key]).
    """
    if lr is not None:
        return res_dir / "predictions" / f"{key}_aug{float(aug)}_lr{lr}_seed{seed}_{condition}.parquet"
    return res_dir / "predictions" / f"{key}_aug{float(aug)}_seed{seed}_{condition}.parquet"


def load_predictions(res_dir, data_dir, key, condition, aug=0.0, seed=42, lr=None,
                     with_text=False):
    df = pd.read_parquet(pred_file(res_dir, key, condition, aug, seed, lr))
    if with_text:
        texts = pd.read_parquet(data_dir / f"{condition}.parquet", columns=["id", "text"])
        df = df.merge(texts, on="id", how="left")
    return df


def word_count(texts):
    return texts.str.split().str.len()


def repeated_word_ratio(text):
    words = text.split()
    return 1 - len({w.lower() for w in words}) / len(words) if words else 0.0