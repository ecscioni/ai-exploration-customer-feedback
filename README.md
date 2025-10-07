# AI Exploration — Customer Feedback Classification (Plain-English Guide)

This project is a **toy sorter** for short customer messages. You type a message like _“I was charged twice”_, and it predicts one of five boxes:

- `delivery_issue`
- `refund_request`
- `billing_problem`
- `app_bug`
- `other`

It’s small on purpose, so you can see the **whole machine-learning pipeline** end-to-end:

1) get data → 2) clean it → 3) make a training/test split → 4) train a model → 5) check results → 6) use it to predict new messages.

No heavy math. No fancy code. Copy–paste the commands and watch it work.

---

## What’s in this repo (folders)

```
ai-exploration-customer-feedback/
├─ data/
│  ├─ raw/          # big, original files (don’t edit)
│  └─ processed/    # small, cleaned files used for training/testing
├─ models/          # saved model files you can reuse (vectorizer + classifier)
├─ reports/         # pictures and text reports (evaluation results)
│  └─ figures/
├─ src/             # small scripts you run
├─ notes/           # plain-language notes (what we tried, what worked, failures too)
└─ README.md        # this guide
```

---

## Key idea (in kid-friendly words)

- We turn text into numbers with **TF‑IDF** (common words like “the/and” matter less; special words like “refund/crash” matter more).
- A simple classifier (**Naive Bayes** by default) learns which words push toward which box.
- We save two files so we can reuse the model instantly:
  - `models/vectorizer.joblib` — how the text becomes numbers
  - `models/classifier.joblib` — the trained model itself

> If you train again, these files get **overwritten** with the new model. If you want to keep multiple models, save to different filenames (see below).

---

## One‑time setup

Open a terminal in the project folder and create a Python “bubble” (virtual environment).

**Windows PowerShell**
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## About the datasets we use

You will see **three sizes** of data in this project:

- **Small sample** (recommended while learning): `data/raw/customer_feedback_sample.csv`  
  – A balanced-ish subset created from the big CSV. Fast to train/evaluate.
- **Bigger sample**: same file but with higher caps (see below). Good once the pipeline works.
- **Full raw CSV**: `data/raw/complaints.csv` from CFPB (very big). We don’t train on the full file yet; it’s slow and noisy.

> **Change of plan (learning outcome):** we keep using `customer_feedback_sample.csv` so failures and fixes are clear. You can still scale up later.

---

## Workflow overview (plain words)

- **ingest** → map the big complaints CSV to our two columns: `text`, `category` (5 boxes).  
- **make_sample** → create a smaller, balanced training set (cap each class; keep “other” smaller).  
- **preprocess** → split the sample into **train** and **test** CSVs.  
- **train** → learn TF‑IDF + Naive Bayes (or Logistic Regression).  
- **evaluate** → print metrics and save a confusion‑matrix picture.  
- **predict** → try single messages from the terminal.

You can run all of this with copy–paste commands below.

---

## 0) Put your big CSV in place (only for real data)

Put the downloaded CFPB CSV here:
```
data/raw/complaints.csv
```

If you don’t have this file, skip to **2) Make a training sample** (maybe you already have `customer_feedback_sample.csv` in the repo).

---

## 1) Ingest (map to our 5 boxes) — real data only

**Creates**: `data/raw/customer_feedback.csv` with **two columns**: `text`, `category`

```powershell
python .\src\ingest_cfpb.py
```

This prints class counts so you can see what you just built.

> If you only want to work with the sample already in the repo, you can **skip** ingest and go straight to **2) Make a training sample**.

---

## 2) Make a training sample

**Creates**: `data/raw/customer_feedback_sample.csv`

This step balances the data and keeps “other” smaller (otherwise it dominates). You choose the size with caps.

### A) Small sample (fast + recommended)
```powershell
python .\src\make_sample.py --cap 30000 --cap-other 12000
```

### B) Bigger sample (once everything works)
```powershell
python .\src\make_sample.py --cap 50000 --cap-other 20000
```

> The numbers above are **maximum per class**. If a class has fewer rows than the cap, it will just take all available rows.

---

## 3) Preprocess (split into train/test)

**Creates**: `data/processed/train.csv` and `data/processed/test.csv`

```powershell
python -m src.preprocess --input ".\data\raw\customer_feedback_sample.csv" --output-dir ".\data\processed" --test-size 0.2 --random-state 42
```

> We run it as a **module** (`python -m src.preprocess`) to avoid Python’s relative‑import errors.

---

## 4) Train a model (saves to `models/`)

This learns from `train.csv` and writes two files:
- `models/vectorizer.joblib`
- `models/classifier.joblib`

### Option 1 — Naive Bayes (ComplementNB) on word 1–2 n‑grams
Fast and solid for imbalanced text.

```powershell
python .\src\train.py --train-path ".\data\processed\train.csv" --vectorizer-out ".\models\vectorizer.joblib" --model-out ".\models\classifier.joblib" --model-type complement --alpha 0.3 --analyzer word --min-df 10 --max-df 0.95 --max-ngram 2 --sublinear-tf
```

### Option 2 — (Optional) Logistic Regression on word 1–2 n‑grams
Often stronger on messy, real text. Use this **only if** you have `src/train_logreg.py` in the repo.

```powershell
python .\src\train_logreg.py --train-path ".\data\processed\train.csv" --vectorizer-out ".\models\vectorizer.joblib" --model-out ".\models\classifier.joblib" --analyzer word --min-df 10 --max-df 0.95 --max-ngram 2 --sublinear-tf --C 2.0
```

> **Keeping multiple models**: change the output names, e.g.  
> `--vectorizer-out ".\models\vectorizer_nb.joblib" --model-out ".\models\classifier_nb.joblib"` and similarly for logreg.

---

## 5) Evaluate (see accuracy/F1 + confusion matrix)

**Reads**: the files in `models/`  
**Writes**: a PNG confusion matrix and a text report

```powershell
python -m src.evaluate --data-path ".\data\processed\test.csv" --vectorizer ".\models\vectorizer.joblib" --model ".\models\classifier.joblib" --output-fig ".\reports\figures\confusion_matrix.png" | Tee-Object -FilePath ".\reports\classification_report.txt"
```

Open `reports/` to view results.

> Tip: Also check the **majority baseline** (how big the largest class is). If the largest class is 0.52 and your accuracy is 0.63, that’s a **real** improvement.

---

## 6) Predict single messages (quick demo)

```powershell
python .\src\predict.py --text "The mobile app crashes when I try to log in"
python .\src\predict.py --text "I was charged twice and need a refund"
python .\src\predict.py --text "I never received my new card in the mail"
```

You’ll see the predicted label and per‑class probabilities.

---

## Training on different sizes (what to use when)

- **Small sample** (`--cap 30000 --cap-other 12000`) — best for learning + quick iteration.  
- **Bigger sample** (`--cap 50000 --cap-other 20000`) — when the pipeline works and you want a bit more data.  
- **Full CSV** — not recommended yet. It’s huge and noisy; fix mapping and evaluation first. If you later need to scale, switch to a streaming approach (HashingVectorizer + online/partial_fit) to avoid running out of memory.

---

## What’s actually inside the saved files?

- `vectorizer.joblib` — the TF‑IDF settings (e.g., word vs char n‑grams, min_df, etc.).  
- `classifier.joblib` — the trained model (Naive Bayes or Logistic Regression).  
Whichever training command you ran **last** wrote those files. That’s what `predict.py` will use.

If you keep multiple versions (NB vs LogReg), pass their paths explicitly when evaluating/predicting.

---

## Common problems (and fixes)

- **Relative import error** (`attempted relative import with no known parent package`)  
  → Run scripts that live in `src/` as **modules**:
  ```powershell
  python -m src.preprocess ...
  python -m src.evaluate ...
  ```

- **pandas not found**  
  → You ran `py -3 ...` (system Python) instead of your **venv**. Activate venv and use `python`:
  ```powershell
  . .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- **Command split across lines** (PowerShell)  
  → If using backticks, the backtick must be the **last character** on the line. Easiest fix: run commands on **one line**.

- **Models get overwritten**  
  → Change `--vectorizer-out` / `--model-out` to new filenames when you want to keep multiple versions.

---

## FAQ

**Q: Why is accuracy around ~0.6 on real data when the tiny demo looked better?**  
A: Real messages are messy (multiple issues in one message) and our auto‑labels are noisy. Clean labels matter more than “more data.”

**Q: Can I train on all 2M+ rows?**  
A: Not yet. It’s slow and won’t fix label noise. First make the model work well on a balanced sample. Then consider streaming/online training.

**Q: What if the model seems unsure?**  
A: In production, use a **confidence threshold** (if max probability is low, route to “unknown”/human review) to avoid bad misroutes.

---

## Short glossary (no jargon)

- **TF‑IDF**: a way to score words so common words count less and special words count more.
- **Naive Bayes**: each word “votes” for a box; we add votes and pick the biggest pile.
- **Confusion matrix**: a picture of where the model is mixing up boxes.
- **Majority baseline**: accuracy if you always guess the most common box.

---

## Repro checklist

- Same Python version in a virtual env
- `pip install -r requirements.txt`
- Same random seed (`--random-state 42`)
- Same caps in `make_sample.py`
- Use the same train/evaluate commands

Good luck—and remember: **small, clean steps beat big, messy leaps**.
