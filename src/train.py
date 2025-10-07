# src/train.py
r"""
Training script for the customer feedback classifier (TF-IDF + Naive Bayes).

Adds flexible flags:
  --analyzer {word,char,char_wb}
  --char-min/--char-max (for char n-grams)
  --min-df / --max-df / --sublinear-tf / --max-features
  --model-type {multinomial,complement}
  --alpha

Examples (PowerShell):

# WORD unigrams+bigrams, ComplementNB, stronger min_df
python .\src\train.py `
  --train-path ".\data\processed\train.csv" `
  --vectorizer-out ".\models\vectorizer.joblib" `
  --model-out ".\models\classifier.joblib" `
  --model-type complement `
  --alpha 0.3 `
  --analyzer word `
  --min-df 10 `
  --max-ngram 2 `
  --sublinear-tf

# CHAR n-grams (char_wb 3–5), ComplementNB, capped features
python .\src\train.py `
  --train-path ".\data\processed\train.csv" `
  --vectorizer-out ".\models\vectorizer.joblib" `
  --model-out ".\models\classifier.joblib" `
  --model-type complement `
  --alpha 0.3 `
  --analyzer char_wb `
  --char-min 3 `
  --char-max 5 `
  --min-df 5 `
  --sublinear-tf `
  --max-features 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB


def _coerce_min_df(val):
    """
    Accept both proportions (0<val<=1.0) and counts (>=1).
    If a float > 1.0 is given (e.g., 10.0), cast to int 10 so sklearn is happy.
    """
    if isinstance(val, int):
        return val
    try:
        v = float(val)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            "min_df must be int>=1 or float in (0,1]."
        ) from e
    if v > 1.0:
        return int(round(v))
    if 0.0 < v <= 1.0:
        return v
    raise argparse.ArgumentTypeError("min_df must be int>=1 or float in (0,1].")


def build_vectorizer(
    analyzer: str,
    max_ngram: int,
    char_min: int,
    char_max: int,
    min_df,
    max_df: float,
    sublinear_tf: bool,
    max_features: int | None,
) -> TfidfVectorizer:
    # make sure min_df obeys sklearn contract
    min_df = _coerce_min_df(min_df)

    if analyzer == "word":
        ngram_range = (1, max_ngram)
    elif analyzer in ("char", "char_wb"):
        ngram_range = (char_min, char_max)
    else:
        raise ValueError("analyzer must be one of: word, char, char_wb")

    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=True,
        min_df=min_df,  # int>=1 or 0<float<=1
        max_df=max_df,  # 0<float<=1 for corpus-specific stop-words
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )


def load_training_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    if "text" not in df.columns or "category" not in df.columns:
        raise SystemExit(
            f"Expected columns 'text' and 'category' in {path}. Got: {list(df.columns)[:10]} ..."
        )
    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["text", "category"]).reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser(
        description="Train TF-IDF + Naive Bayes text classifier."
    )

    # IO
    p.add_argument(
        "--train-path",
        required=True,
        help="Path to training CSV (must have 'text','category').",
    )
    p.add_argument(
        "--vectorizer-out", required=True, help="Where to save vectorizer .joblib"
    )
    p.add_argument(
        "--model-out", required=True, help="Where to save classifier .joblib"
    )

    # Vectorizer options
    p.add_argument("--analyzer", choices=["word", "char", "char_wb"], default="word")
    p.add_argument(
        "--max-ngram",
        type=int,
        default=2,
        help="For word analyzer: use (1, max_ngram).",
    )
    p.add_argument(
        "--char-min", type=int, default=3, help="For char analyzers: min n-gram length."
    )
    p.add_argument(
        "--char-max", type=int, default=5, help="For char analyzers: max n-gram length."
    )
    p.add_argument(
        "--min-df",
        type=float,
        default=2,
        help="Doc-frequency cutoff: int>=1 or 0<frac<=1.",
    )
    p.add_argument(
        "--max-df",
        type=float,
        default=1.0,
        help="Drop too-common terms: fraction in (0,1].",
    )
    p.add_argument("--sublinear-tf", action="store_true", help="Use 1+log(tf) scaling.")
    p.add_argument(
        "--max-features", type=int, default=None, help="Cap vocabulary size."
    )

    # Model options
    p.add_argument(
        "--model-type",
        choices=["multinomial", "complement"],
        default="multinomial",
        help="ComplementNB often helps on imbalanced text.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.3, help="NB smoothing (try 0.1, 0.3, 1.0)."
    )

    args = p.parse_args()

    # Load data
    train_df = load_training_csv(args.train_path)

    # Vectorize
    vectorizer = build_vectorizer(
        analyzer=args.analyzer,
        max_ngram=args.max_ngram,
        char_min=args.char_min,
        char_max=args.char_max,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
        max_features=args.max_features,
    )
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["category"].values

    # Model
    if args.model_type == "complement":
        clf = ComplementNB(alpha=args.alpha)
    else:
        clf = MultinomialNB(alpha=args.alpha)

    clf.fit(X, y)

    # Save
    Path(args.vectorizer_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.vectorizer_out)
    joblib.dump(clf, args.model_out)

    # Summary
    vocab_size = getattr(vectorizer, "vocabulary_", None)
    vocab_n = len(vocab_size) if vocab_size is not None else X.shape[1]
    print("=== Training summary ===")
    print(f"Rows: {X.shape[0]:,}")
    print(f"Features (vocab): {vocab_n:,}")
    print(f"Analyzer: {args.analyzer}")
    if args.analyzer == "word":
        print(f"Word n-grams: (1, {args.max_ngram})")
    else:
        print(f"Char n-grams: ({args.char_min}, {args.char_max})")
    print(
        f"min_df={_coerce_min_df(args.min_df)}, max_df={args.max_df}, sublinear_tf={args.sublinear_tf}, max_features={args.max_features}"
    )
    print(f"Model: {args.model_type}, alpha={args.alpha}")
    print(f"Saved vectorizer → {args.vectorizer_out}")
    print(f"Saved classifier → {args.model_out}")


if __name__ == "__main__":
    main()
