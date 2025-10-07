# src/ingest_cfpb.py
import re
from pathlib import Path

import pandas as pd

RAW_IN = Path("data/raw/complaints.csv")  # your download
RAW_OUT = Path("data/raw/customer_feedback.csv")  # project-standard file name

# ---- CONFIG: tune these if needed ----
# Keep only CFPB products where our 5 buckets make sense
ALLOWED_PRODUCTS_NORM = {
    "credit_card_or_prepaid_card",
    "bank_account_or_service",
    "money_transfer_virtual_currency_or_money_service",
    "checking_or_savings_account",  # sometimes used in newer dumps
    "credit_card",  # older naming variants
    "prepaid_card",
    "money_transfers",  # older variant
}

MIN_TEXT_LEN = 15  # drop super-short narratives


# ---- helpers ----
def normalize(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def contains(patterns, s: str) -> bool:
    s = s or ""
    s = s.lower()
    return any(re.search(p, s) for p in patterns)


# ISSUE-based rules (most precise → highest priority)
ISSUE_REFUND_PATTERNS = [
    r"\brefund",
    r"chargeback",
    r"reimburse",
    r"money back",
    r"credit balance refund",
    r"reverse(d)? charge",
    r"paid by mistake",
]
ISSUE_BILLING_PATTERNS = [
    r"billing",
    r"\bcharged\b",
    r"overcharg",
    r"fee",
    r"interest",
    r"late fee",
    r"incorrect amount",
    r"statement",
    r"charged twice",
]
ISSUE_DELIVERY_PATTERNS = [
    r"card not received",
    r"never received",
    r"did not receive",
    r"in the mail",
    r"mail delivery",
    r"delayed (?:delivery|mail|card)",
]

# TEXT-based rules
TEXT_APP_PATTERNS = [
    r"\bapp\b",
    r"application",
    r"\bmobile\b",
    r"website",
    r"online portal",
    r"\bonline\b",
    r"portal",
    r"login",
    r"password",
    r"technical",
    r"error",
    r"crash",
    r"bug",
    r"glitch",
    r"update",
    r"site (?:down|error)",
    r"unable to (?:log|login|sign in)",
    r"two[- ]?factor",
    r"otp",
]
TEXT_REFUND_PATTERNS = [
    r"\brefund",
    r"chargeback",
    r"reimburse",
    r"money back",
    r"reverse(d)? charge",
]
TEXT_BILLING_PATTERNS = [
    r"\bcharged\b",
    r"billing",
    r"fee",
    r"interest",
    r"late fee",
    r"overcharg",
    r"statement",
    r"charged twice",
]
TEXT_DELIVERY_PATTERNS = [
    r"card (?:not|never) received",
    r"never received",
    r"did not receive",
    r"in the mail",
    r"delayed",
    r"arriv(e|al)",
    r"mail delivery",
    r"debit card not received",
    r"credit card not received",
]


def label_row(narrative: str, issue: str, product: str) -> str:
    issue = (issue or "").lower()
    text = (narrative or "").lower()
    prod = (product or "").lower()

    # 1) ISSUE first (more stable than free-text)
    if contains(ISSUE_REFUND_PATTERNS, issue):
        return "refund_request"
    if contains(ISSUE_BILLING_PATTERNS, issue):
        return "billing_problem"
    if contains(ISSUE_DELIVERY_PATTERNS, issue):
        return "delivery_issue"

    # 2) TEXT app/site/login/etc.
    if contains(TEXT_APP_PATTERNS, text):
        return "app_bug"

    # 3) TEXT refund/billing/delivery
    if contains(TEXT_REFUND_PATTERNS, text):
        return "refund_request"
    if contains(TEXT_BILLING_PATTERNS, text):
        return "billing_problem"
    if contains(TEXT_DELIVERY_PATTERNS, text):
        return "delivery_issue"

    # 4) fallback
    return "other"


def main():
    if not RAW_IN.exists():
        raise SystemExit(f"Input file not found: {RAW_IN}")

    # Read minimal columns to save RAM; handle BOM and weird encodings
    try:
        df = pd.read_csv(RAW_IN, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(RAW_IN, encoding="utf-8", low_memory=False)

    # Normalize all column names
    col_map = {c: normalize(c) for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Figure out available names across dumps
    text_col = "consumer_complaint_narrative"
    product_col = "product"
    issue_col = "issue"

    have = set(df.columns)
    missing = [c for c in [text_col, product_col, issue_col] if c not in have]
    if text_col not in have:
        raise SystemExit(
            f"Column '{text_col}' not found. Available = {list(df.columns)[:20]} ..."
        )

    # Keep only what we need
    keep = [c for c in [text_col, product_col, issue_col] if c in df.columns]
    df = df[keep].copy()

    # Filter to allowed products (after normalizing values)
    if product_col in df.columns:
        prod_vals = df[product_col].astype(str).map(normalize)
        df = df[prod_vals.isin(ALLOWED_PRODUCTS_NORM)].copy()

    # Drop empty narratives + trim
    df = df[df[text_col].notnull()].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() >= MIN_TEXT_LEN].copy()

    # Label
    df["category"] = [
        label_row(
            narrative=row.get(text_col, ""),
            issue=row.get(issue_col, "") if issue_col in df.columns else "",
            product=row.get(product_col, "") if product_col in df.columns else "",
        )
        for _, row in df.iterrows()
    ]

    # Standardize schema
    df.rename(columns={text_col: "text"}, inplace=True)
    df = df[["text", "category"]]

    # Drop dupes and shuffle
    df = df.drop_duplicates().sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Write
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUT, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows → {RAW_OUT}")
    print("\nClass counts:")
    print(df["category"].value_counts())
    print("\nClass %:")
    print((df["category"].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    main()
