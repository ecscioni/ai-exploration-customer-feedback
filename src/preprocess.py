"""
Preprocessing utilities for the customer feedback dataset.

This module provides functions to load the raw data, clean it and save
processed splits for reproducibility.  Running this script directly will
produce train and validation CSV files under `data/processed/`.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_raw_data, clean_text


def preprocess_data(
    input_path: str,
    drop_duplicates: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, optionally drop duplicates, clean text and split into train/test."""
    df = load_raw_data(input_path)
    df['text'] = df['text'].apply(clean_text)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['category']
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save train and test DataFrames to CSV files in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess the customer feedback dataset.")
    parser.add_argument(
        '--input', default='data/raw/customer_feedback.csv', help='Path to the raw CSV file'
    )
    parser.add_argument(
        '--output-dir', default='data/processed', help='Directory where processed splits will be saved'
    )
    parser.add_argument('--no-dedup', action='store_true', help='Do not drop duplicate rows')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data for the test set')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()

    train_df, test_df = preprocess_data(
        args.input, drop_duplicates=not args.no_dedup, test_size=args.test_size, random_state=args.random_state
    )
    save_splits(train_df, test_df, args.output_dir)
    print(f"Saved {len(train_df)} training rows and {len(test_df)} test rows to {args.output_dir}")


if __name__ == '__main__':
    main()
