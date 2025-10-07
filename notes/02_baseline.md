# 02 – Baselines and feature experiments

This stage establishes simple baselines for the classification problem and explores a few variations in feature extraction.  The goal is to see how much improvement a Bag‑of‑Words model delivers over a trivial majority predictor and whether basic preprocessing choices matter.

## Majority‑class baseline

The simplest possible model is to always predict the most common class in the training set.  In the deduplicated dataset the majority class is **delivery_issue**.  When evaluated on the validation split this approach reaches only about **22 % accuracy** and a **macro F1 of ~0.07**.  This low macro F1 is expected: predicting one class results in zero F1 for all other classes.  This baseline sets a very low bar to beat.

## Bag‑of‑Words + Multinomial Naive Bayes

I then trained a Multinomial Naive Bayes classifier on simple Bag‑of‑Words features using `CountVectorizer` with default settings (lowercasing only).  After deduplication and an 80/20 stratified split the model achieved **~83 % accuracy** and **~0.82 macro F1** on the validation set—an order of magnitude improvement over the majority predictor.  The confusion matrix below shows that most categories are correctly classified, although a few `billing_problem` messages were mistaken for other classes.

![Confusion matrix for baseline NB model]({{file:file-M6D4UepvbABrB9YR91TYcL}})

The corresponding classification report (saved to `reports/classification_report_baseline.txt`) reveals precision and recall per class.  For example, `delivery_issue` messages achieved 0.91 F1, while `billing_problem` only achieved 0.75 due to confusion with similar categories.

## Feature comparison

I experimented with a few simple text‑processing toggles:

| Feature variant | Description | Macro F1 | Accuracy | Notes |
|---|---|---|---|---|
| **unigram** | Lowercase word unigrams (default `CountVectorizer`) | 0.820 | 0.826 | Baseline model. |
| **punctuation removed** | Applied a custom preprocessor to strip punctuation before vectorisation | 0.820 | 0.826 | No change – punctuation was not informative here. |
| **stopwords removed** | Used built‑in English stopword list | 0.820 | 0.826 | Also no change; the simple sentences seldom contained common stopwords. |
| **word n‑grams (1–2)** | Included unigrams and bigrams | 0.782 | 0.783 | Performance dropped slightly, likely due to sparse bigram features that the small dataset cannot support. |
| **char n‑grams (3–5)** | Character 3–5‑grams | 0.820 | 0.826 | Similar to the unigram baseline; character features did not hurt because messages are short and well‑formed. |

In this project none of these preprocessing tweaks significantly outperformed the baseline unigrams.  The small, synthetic dataset contains clear keywords for each category (e.g. *refund*, *package*, *charge*, *app*), so a simple bag‑of‑words model captures enough signal.

## Lessons learned

* **Baseline matters:** The majority‑class baseline is extremely weak here, so any real model must do better.  Always compute a trivial baseline to contextualise improvements.
* **Simple models work well on simple data:** A Bag‑of‑Words representation with Naive Bayes already yields strong performance on this toy dataset.  For more complex or noisy real‑world data one might need more sophisticated features or models.
* **Tweaks aren’t always helpful:** Removing punctuation or stopwords and adding bigrams had little or negative effect.  It is important to test assumptions rather than assume that “more preprocessing” always helps.
* **Failure:** I expected word bigrams to help by capturing phrases like “extra fee” or “app crashes”, but the small sample size meant bigram counts were too sparse and actually reduced performance.  Documenting this failure helps set expectations for future experiments.