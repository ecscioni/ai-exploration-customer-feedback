"""
Utility functions for data loading, preprocessing and visualization.

These helpers are shared across training and evaluation scripts.  They are kept
simple to make the pipeline easy to follow.  For more advanced projects you
might consider using a library like `hydra` or `click` for configuration.
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def load_raw_data(path: str) -> pd.DataFrame:
    """Load a CSV file containing customer feedback.

    The file is expected to have at least two columns: 'text' and 'category'.
    Returns a pandas DataFrame.
    """
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    """Basic text cleaning: strip whitespace and lowercase.

    You could extend this function to remove punctuation, lemmatise, etc.
    """
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def prepare_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply basic cleaning to a DataFrame and optionally drop duplicate rows."""
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_dataframe(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of a DataFrame.

    Returns two DataFrames: train and test.  Uses sklearn under the hood but
    keeps the dependency local to this function.
    """
    from sklearn.model_selection import train_test_split

    X = df['text']
    y = df['category']
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=y
    )
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def plot_confusion_matrix(
    cm,
    labels,
    title: str,
    output_path: str | None = None,
    cmap: str = "Blues",
):
    """Plot and optionally save a confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    return plt.gcf()
