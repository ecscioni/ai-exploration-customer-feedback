"""
Training script for the customer feedback classifier.

This script reads processed training data, vectorises the text with TF‑IDF and
trains a Naive Bayes classifier.  The vectoriser and classifier are saved to
disk as joblib files.
"""

from __future__ import annotations

import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, accuracy_score, f1_score


def train_model(
    train_path: str,
    vectorizer_out: str,
    model_out: str,
    model_type: str = 'multinomial',
    alpha: float = 0.1,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
) -> None:
    """Train a TF‑IDF + Naive Bayes classifier and save artefacts."""
    df = pd.read_csv(train_path)
    X_train = df['text']
    y_train = df['category']

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=min_df)
    X_train_vec = vectorizer.fit_transform(X_train)

    if model_type == 'complement':
        clf = ComplementNB(alpha=alpha)
    else:
        clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train_vec, y_train)

    # Save artefacts
    os.makedirs(os.path.dirname(vectorizer_out), exist_ok=True)
    joblib.dump(vectorizer, vectorizer_out)
    joblib.dump(clf, model_out)
    print(f"Saved vectorizer to {vectorizer_out} and classifier to {model_out}")


def main():
    parser = argparse.ArgumentParser(description='Train a Naive Bayes classifier on customer feedback.')
    parser.add_argument('--train-path', default='data/processed/train.csv', help='Path to the processed training CSV')
    parser.add_argument('--vectorizer-out', default='models/vectorizer.joblib', help='Path to save the vectorizer')
    parser.add_argument('--model-out', default='models/classifier.joblib', help='Path to save the classifier')
    parser.add_argument('--model-type', choices=['multinomial', 'complement'], default='multinomial', help='Type of Naive Bayes model')
    parser.add_argument('--alpha', type=float, default=0.1, help='Smoothing parameter for Naive Bayes')
    parser.add_argument('--min-df', type=int, default=2, help='Minimum document frequency for TF‑IDF')
    parser.add_argument('--max-ngram', type=int, default=2, help='Maximum n‑gram size (min n‑gram is 1)')
    args = parser.parse_args()

    ngram_range = (1, args.max_ngram)
    train_model(
        train_path=args.train_path,
        vectorizer_out=args.vectorizer_out,
        model_out=args.model_out,
        model_type=args.model_type,
        alpha=args.alpha,
        ngram_range=ngram_range,
        min_df=args.min_df,
    )


if __name__ == '__main__':
    main()
