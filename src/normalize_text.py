"""
normalize_text.py - remove formatting and character-level clues from every text.

Used for the shortcut check: if detectors partly rely on layout or odd characters
rather than writing style, their accuracy changes once these clues are removed.
The same cleaning is also a simple defence against character-level attacks.

Steps, applied to every text in every set (human and AI alike):
  1. Unicode NFKC normalisation (e.g. full-width letters -> normal letters)
  2. delete invisible characters (zero-width spaces/joiners, BOM, soft hyphen)
     and the replacement character "\ufffd" left by broken text encoding
  3. map look-alike letters (Cyrillic/Greek homoglyphs) to their Latin letters
  4. remove HTML leftovers (e.g. "<br />" in scraped reviews) and markdown
     markup at the start of lines (headings "##", quotes ">", bullets, "1." numbers)
  5. collapse all whitespace (line breaks, tabs, repeated spaces) into one space
Letter case and punctuation are kept, because they are part of normal writing style.

Usage (from the project root)
  python src/normalize_text.py                       # data/processed -> data/processed_norm
  python src/normalize_text.py --src A --dst B
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

INVISIBLE = ("\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064"
             "\ufeff\u00ad\u180e"   # zero-width / BOM / soft hyphen / Mongolian vowel sep.
             "\ufffd")                # replacement char left by broken encoding
_DELETE = dict.fromkeys(map(ord, INVISIBLE), None)

# Look-alike letters (Cyrillic, Greek, Latin variants) -> plain Latin letters
HOMOGLYPHS = {
    # Cyrillic lower case
    "а": "a", "е": "e", "ё": "e", "һ": "h", "і": "i", "ї": "i", "ј": "j", "ӏ": "l",
    "о": "o", "р": "p", "с": "c", "ѕ": "s", "у": "y", "х": "x", "ԁ": "d", "ԛ": "q",
    "ԝ": "w", "ѵ": "v", "ү": "y",
    # Latin / Armenian look-alikes
    "ɡ": "g", "ɑ": "a", "ı": "i", "ɩ": "i", "ʋ": "v", "ո": "n", "ս": "u", "օ": "o",
    # Cyrillic upper case
    "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ї": "I",
    "Ј": "J", "Ѕ": "S", "Ү": "Y", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W", "Ӏ": "I",
    # Greek
    "ο": "o", "ν": "v", "ρ": "p", "ι": "i",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_MAP = str.maketrans(HOMOGLYPHS)
_SPACE = re.compile(r"\s+")
_HTML = re.compile(r"<\s*/?\s*(?:br|p|div|i|b|em|strong|span)\b[^>]*>", re.IGNORECASE)
_LINE_MARKUP = re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]+|>+[ \t]?|[-*•+][ \t]+|\d{1,3}[.)][ \t]+)")


def normalize(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DELETE).translate(_MAP)
    text = _HTML.sub(" ", text)
    text = _LINE_MARKUP.sub("", text)   # must run before line breaks are collapsed
    return _SPACE.sub(" ", text).strip()


def leftover_report(texts, top=15):
    """Most common non-ASCII characters still present (to spot unmapped homoglyphs)."""
    counts = Counter(ch for t in texts for ch in t if ord(ch) > 127)
    return [(ch, f"U+{ord(ch):04X}", unicodedata.name(ch, "?"), n)
            for ch, n in counts.most_common(top)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed")
    ap.add_argument("--dst", default="data/processed_norm")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files in {src}")
    for f in files:
        df = pd.read_parquet(f)
        before = df["text"].copy()
        df["text"] = df["text"].map(normalize)
        changed = (before != df["text"]).mean()
        newline_before = before.str.contains("\n", regex=False).mean()
        df.to_parquet(dst / f.name, index=False)
        print(f"{f.name:<30} {len(df):>7,} texts | changed {changed:6.1%} | "
              f"had line breaks {newline_before:6.1%}")

    att = pd.read_parquet(dst / "test_heldout_attacks.parquet") \
        if (dst / "test_heldout_attacks.parquet").exists() else None
    if att is not None:
        print("\nMost common non-ASCII characters left in held-out-attack texts")
        print("(homoglyphs that are still here are not in the mapping):")
        for row in leftover_report(att.loc[att["attack"] == "homoglyph", "text"]):
            print("  ", row)
    print(f"\nSaved normalised sets to {dst}/")


if __name__ == "__main__":
    main()