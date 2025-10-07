# 05 – Final takeaways and next steps

This project walked through a complete, if tiny, text classification pipeline.  Starting from a synthetic dataset of customer feedback messages, I conducted exploratory analysis, established baselines, built and tuned a Naive Bayes classifier, analysed errors, and suggested improvements.  Below are the key lessons learned.

## What worked well

* **Simple features suffice on simple data:** A Bag‑of‑Words model with Naive Bayes achieved around 83 % accuracy and 0.82 macro F1 on the deduplicated dataset.  TF‑IDF features and hyperparameter tuning offered no substantial gains.  This underscores that for small, well‑structured text problems simple models can be effective.
* **ComplementNB can mitigate imbalance:** When training data was limited (20 % of samples), switching from `MultinomialNB` to `ComplementNB` modestly improved macro F1 from ~0.63 to ~0.68.  ComplementNB estimates statistics on the complement of each class and is better suited to imbalanced data.
* **Synthetic data is useful for prototyping:** Generating a dataset allowed me to complete all stages of the pipeline without relying on external sources.  It provided a controlled environment to explore modelling decisions and document the process.

## What didn’t work or was surprising

* **TF‑IDF didn’t help:** I expected TF‑IDF to improve discrimination by down‑weighting common words, but performance was identical to the CountVectorizer baseline.  The vocabulary is dominated by a handful of distinctive words (e.g. “refund,” “late,” “charged”), so reweighting had little effect.
* **Word bigrams hurt:** Adding bigrams slightly decreased macro F1, likely because the dataset is too small to estimate bigram probabilities reliably.  More data or character‑based features might be needed.
* **Limited error analysis:** Even after injecting ambiguous messages, the full model made very few mistakes.  I had to deliberately reduce the training set to produce enough misclassifications for analysis.  Real‑world datasets would expose more nuanced failure modes.

## What I would do next time

* **Collect real data:** Synthetic data cannot capture the full diversity, nuance, and noise of genuine customer feedback.  Sourcing a real dataset with varied phrasing, typos, sarcasm and multi‑issue messages is the most important next step.
* **Explore multi‑label and hierarchical classification:** Many messages span multiple issues.  Extending the model to predict multiple categories or a hierarchy (e.g. billing → refund vs billing → general enquiry) would improve routing accuracy.
* **Try linear SVM or logistic regression:** These linear models often outperform Naive Bayes on text by capturing feature interactions implicitly.  They remain efficient on sparse data.
* **Integrate basic NLP preprocessing:** Lemmatization, handling negations (“not charged”), and recognising synonyms could improve robustness.  Character n‑grams may capture misspellings and morphological variants.
* **Evaluate on balanced and imbalanced splits:** Reporting both macro and micro F1 helped reveal the impact of class imbalance.  Future work should include stratified cross‑validation and perhaps resampling techniques.

## Answers to critical analysis questions

1. **Classification challenges:** Ambiguity (messages mentioning several issues), multi‑issue texts, sarcasm/irony (“thanks for the great bug”), evolving jargon (new slang or abbreviations) and domain drift (support topics change over time) all complicate classification.  A model trained on historical data may misclassify novel complaints.
2. **Misclassification impact:** Routing a request to the wrong team delays resolution, frustrates customers and can lead to churn.  Billing mistakes might be escalated to technical support or vice versa.  Mitigation strategies include confidence thresholds, an “other/unknown” category, human review for low‑confidence predictions, and continuous feedback loops where agents correct misclassifications and provide new training data.
3. **Accuracy strategies:** Better preprocessing (lowercasing, lemmatization, handling negations), balanced training sets or class weighting, character n‑grams to catch typos, using `ComplementNB` to address imbalance, and hyperparameter tuning can all improve accuracy.  When simple models plateau, try linear SVMs or fine‑tuned transformer models like BERT, which capture context and semantics at the cost of computational resources.

## Reflection

This exploration demonstrates the value of systematic experimentation and documentation.  By starting with a clear plan, measuring baselines, and recording failures, it becomes easier to justify modelling choices and identify next steps.  Even a synthetic project can teach important lessons about handling class imbalance, understanding model assumptions, and anticipating real‑world challenges.