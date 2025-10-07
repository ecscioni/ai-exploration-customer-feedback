# src/make_sample.py
import argparse
from pathlib import Path
import pandas as pd


def make_sample(
    src: Path,
    dst: Path,
    cap: int,
    cap_other: int,
    seed: int = 42,
):
    df = pd.read_csv(src)
    if "text" not in df.columns or "category" not in df.columns:
        raise SystemExit("Expected columns 'text' and 'category' in the CSV.")

    print("Class counts (full):")
    print(df["category"].value_counts())
    print("\nClass % (full):")
    print((df["category"].value_counts(normalize=True) * 100).round(2))

    parts = []
    for k, g in df.groupby("category", group_keys=False):
        if k == "other":
            n = min(cap_other, len(g))
        else:
            n = min(cap, len(g))
        parts.append(g.sample(n=n, random_state=seed))

    sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print("\nSampled size:", len(sample))
    print(sample["category"].value_counts())
    print("\nClass % (sample):")
    print((sample["category"].value_counts(normalize=True) * 100).round(2))

    dst.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dst, index=False, encoding="utf-8")
    print(f"\nSaved sample → {dst}")


def main():
    ap = argparse.ArgumentParser(
        description="Create a class-capped sample from customer_feedback.csv"
    )
    ap.add_argument("--src", type=Path, default=Path("data/raw/customer_feedback.csv"))
    ap.add_argument(
        "--dst", type=Path, default=Path("data/raw/customer_feedback_sample.csv")
    )
    ap.add_argument(
        "--cap", type=int, default=30000, help="Per-class cap (non-'other')"
    )
    ap.add_argument("--cap-other", type=int, default=12000, help="Cap for 'other'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_sample(args.src, args.dst, args.cap, args.cap_other, args.seed)


if __name__ == "__main__":
    main()
