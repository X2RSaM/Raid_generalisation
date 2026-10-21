# Do AI-Text Detectors Generalise? Unseen Generators, Held-Out Attacks and Adversarial Augmentation on RAID

MSc in Artificial Intelligence, National College of Ireland: Machine Learning CA project.
Author: Sourav Arun Mahajan

## Research question

How much does the accuracy at a 5% false positive rate of TF-IDF and RoBERTa
detectors drop when tested on unseen generators and held-out attacks in RAID,
and does adversarial augmentation reduce this drop?

## Data

This project uses the RAID benchmark (Dugan et al., ACL 2024).
The data is **not** included in this repository. Download the labelled training file:

```bash
mkdir -p data
curl -o data/train.csv https://dataset.raid-bench.xyz/train.csv
```

Source and licence: https://github.com/liamdugan/raid

## Environment

```bash
conda create -n raid python=3.11 -y
conda activate raid
pip install -r requirements.txt
```

Note: `pyarrow<19` is required because the environment uses NumPy 1.x.

## Reproducing the experiments

1. Inspect the dataset (optional):
   ```bash
   python src/inspect_raid.py data/train.csv
   ```
2. Build leakage-free, class-balanced splits (seed 42):
   ```bash
   python src/make_splits.py --raid data/train.csv --out data/processed --seed 42
   ```

More steps will be added as the experiments are implemented.

## Experimental design

- **Seen generators (training):** gpt2, gpt3, chatgpt, cohere, cohere-chat, mistral, mistral-chat
- **Held-out generators (test only):** gpt4, llama-chat, mpt, mpt-chat
- **Seen attacks (augmentation):** article_deletion, number, paraphrase, synonym, perplexity_misspelling, alternative_spelling
- **Held-out attacks (test only):** homoglyph, zero_width_space, whitespace, upper_lower, insert_paragraphs
- **Splits:** 70/15/15 of human source documents, stratified by domain; every AI and attacked text follows its source document's split.

## Repository structure

```
data/        raw and processed data (not tracked by Git)
notebooks/   exploration
src/         scripts
configs/     seeds and hyperparameters
results/     metrics and figures
```

## AI usage

Code in this repository was produced with the help of Claude (Anthropic), as declared on the project's AI acknowledgement cover sheet.
