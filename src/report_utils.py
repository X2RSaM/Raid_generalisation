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
                  "tfidf_word_char": "TF-IDF (word+char)", "roberta-base": "RoBERTa"}
DETECTOR_ORDER = list(DETECTOR_NAMES.values())

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
COLORS = {"Baseline": "#1baf7a", "TF-IDF (word)": "#6250d6",
          "TF-IDF (word+char)": "#2a78d6", "RoBERTa": "#eb6834"}
HATCH = {"RoBERTa": "////"}            # texture so the figures also work in greyscale
MARKERS = {"TF-IDF (word+char)": "o", "RoBERTa": "s", "Baseline": "^", "TF-IDF (word)": "D"}
PAIR = ("#1baf7a", "#6250d6")          # two non-detector series (e.g. human vs AI)
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
def chosen_lr(roberta_eff):
    """Learning rate with the best validation AUROC in the tuning runs."""
    tune = roberta_eff[(roberta_eff["aug"] == 0) & (roberta_eff["seed"] == 42)]
    return float(tune.loc[tune["val_auroc"].idxmax(), "lr"])


def load_results(res_dir):
    """Classical + RoBERTa results; RoBERTa restricted to the validation-chosen lr."""
    cl = pd.read_csv(res_dir / "classical_results.csv")
    rb = pd.read_csv(res_dir / "roberta_results.csv")
    best_lr = chosen_lr(pd.read_csv(res_dir / "roberta_efficiency.csv"))
    rb = rb[np.isclose(rb["lr"], best_lr)]
    df = pd.concat([cl, rb], ignore_index=True)
    df["detector"] = df["model"].map(DETECTOR_NAMES)
    df["condition_name"] = df["condition"].map(CONDITION_NAMES)
    return df, best_lr


def summarise(df, by, metric):
    """Mean and std over seeds."""
    return df.groupby(by)[metric].agg(["mean", "std"])


def pred_file(res_dir, key, condition, aug=0.0, seed=42, lr=None):
    """Path of a per-text prediction file written by the training scripts."""
    if key == "roberta-base":
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