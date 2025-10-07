import pandas as pd
from pathlib import Path

SRC = Path("data/raw/customer_feedback.csv")
DST = Path("data/raw/customer_feedback_sample.csv")

# target per-class cap (tune as you like)
CAP = 40000  # ~<= 200k total if 5 classes are that large


def main():
    df = pd.read_csv(SRC)
    if "text" not in df.columns or "category" not in df.columns:
        raise SystemExit("Expected columns 'text' and 'category' in the CSV.")

    # Show class counts
    print("Class counts (full):")
    print(df["category"].value_counts())

    # Cap per class to balance a bit and keep dataset manageable
    parts = []
    for k, g in df.groupby("category", group_keys=False):
        n = min(CAP, len(g))
        parts.append(g.sample(n=n, random_state=42))
    sample = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\nSampled size:", len(sample))
    print(sample["category"].value_counts())

    DST.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(DST, index=False, encoding="utf-8")
    print(f"\nSaved sample → {DST}")


if __name__ == "__main__":
    main()
