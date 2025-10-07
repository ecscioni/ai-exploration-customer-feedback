"""
Evaluation script for the customer feedback classifier.

This script loads a saved vectorizer and classifier and evaluates them on a
processed dataset.  It prints a classification report and can optionally
generate a confusion matrix image.
"""

from __future__ import annotations

import argparse
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from .utils import plot_confusion_matrix


def evaluate(
    data_path: str,
    vectorizer_path: str,
    model_path: str,
    output_fig: str | None = None,
) -> None:
    """Evaluate a saved model on a dataset and optionally save a confusion matrix."""
    df = pd.read_csv(data_path)
    X = df["text"]
    y = df["category"]

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}\n")
    print(classification_report(y, y_pred))

    if output_fig:
        labels = model.classes_
        cm = confusion_matrix(y, y_pred, labels=labels)
        plot_confusion_matrix(cm, labels, "Confusion matrix", output_fig)
        print(f"Confusion matrix saved to {output_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier on a dataset."
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/test.csv",
        help="Path to processed CSV file",
    )
    parser.add_argument(
        "--vectorizer",
        default="models/vectorizer.joblib",
        help="Path to saved vectorizer",
    )
    parser.add_argument(
        "--model", default="models/classifier.joblib", help="Path to saved classifier"
    )
    parser.add_argument(
        "--output-fig", default=None, help="Path to save confusion matrix PNG"
    )
    args = parser.parse_args()

    evaluate(
        data_path=args.data_path,
        vectorizer_path=args.vectorizer,
        model_path=args.model,
        output_fig=args.output_fig,
    )


if __name__ == "__main__":
    main()
