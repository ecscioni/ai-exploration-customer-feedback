# 03 – Main modelling with TF‑IDF and Naive Bayes

After establishing a baseline using Bag‑of‑Words features I moved to a slightly richer representation: TF‑IDF (term frequency–inverse document frequency).  TF‑IDF down‑weights words that appear in many documents and emphasizes those that are distinctive for a given category.  I compared two variants of Naive Bayes and tuned the smoothing hyperparameter.

## Naive Bayes in plain English

Naive Bayes classifiers estimate the probability that a document belongs to a class by multiplying the probabilities of its individual words given that class.  The “naive” assumption is that words are *conditionally independent* of each other once you know the class.  In other words, the model pretends that seeing the word “refund” in a message tells you something about the class (e.g. **refund_request**) but provides no additional information when combined with “charged”; it simply multiplies the separate probabilities.  This simplification makes the model very fast and works surprisingly well for text because high‑frequency keywords often dominate the decision.

Mathematically the model computes

\[
P(C \mid \mathbf{w}) \propto P(C) \prod_{i=1}^n P(w_i \mid C),
\]

where \(P(C)\) is the prior probability of class \(C\), \(w_i\) are the words in the message, and \(P(w_i \mid C)\) are estimated from the training data.  The classifier picks the class with the highest posterior probability.  A smoothing parameter \(\alpha\) is used to avoid zero probabilities when a word was never seen in a class.

### Why Naive Bayes is fast and good for sparse text

* **Speed:** Training simply counts how often each word occurs per class; there is no gradient descent.  Prediction multiplies (or adds log‑probabilities of) a handful of word frequencies, so it is extremely efficient.
* **Works on sparse text:** TF‑IDF vectors are high‑dimensional and mostly zeros.  Naive Bayes handles this because it only needs to look up the few non‑zero features for each document.  Linear classifiers like NB or SVMs are well‑suited to such sparse representations.

### When Naive Bayes struggles

* **Sarcasm and negation:** NB cannot capture that “I didn’t get a refund” is the opposite of “I got a refund” because it treats words independently.
* **Long‑range context:** Phrases where meaning depends on the combination of words (e.g. “app crashes after payment”) may be oversimplified.
* **Rare phrases:** If a rare but highly indicative phrase never appears in training, NB will assign a low probability to that class.  Smoothing mitigates this but cannot invent unseen evidence.

### Toy example

Suppose a message contains the words “refund”, “charged”, and “late”.  In a Naive Bayes model the probability for the class **refund_request** might be computed as follows (using hypothetical conditional probabilities):

* \(P(\text{refund}\mid \text{refund_request}) = 0.4\)
* \(P(\text{charged}\mid \text{refund_request}) = 0.2\)
* \(P(\text{late}\mid \text{refund_request}) = 0.05\)
* Prior \(P(\text{refund_request}) = 0.3\)

Multiplying these gives a proportional score of \(0.3 \times 0.4 \times 0.2 \times 0.05 = 0.0012\).  The same words might have lower probabilities for other classes (e.g. only \(P(\text{late}\mid \text{delivery_issue})\) might be high), so the classifier will likely choose **refund_request**.

## Parameter tuning and model selection

I used a `TfidfVectorizer` with 1‑ and 2‑gram features and a minimum document frequency of 2 to avoid extremely rare bigrams.  Two Naive Bayes variants were evaluated:

* **MultinomialNB:** The standard algorithm for multinomial event counts.
* **ComplementNB:** Designed to better handle class imbalance by estimating statistics on all other classes.

I explored smoothing values \(\alpha \in \{0.1, 0.3, 1.0, 3.0\}\).  Surprisingly, all combinations performed identically on this dataset (macro F1 ≈ 0.82).  The best model according to macro F1 was `MultinomialNB` with \(\alpha=0.1\), but the difference to other settings was negligible.

The classification report for the chosen model is similar to the Bag‑of‑Words baseline: overall accuracy **83 %** and macro F1 **0.82**.  The confusion matrix below shows the distribution of errors:

![Confusion matrix for TF‑IDF + NB]({{file:file-8qADizpV1YAKisRw87LHN2}})

## Observations and failures

* **No improvement over baseline:** TF‑IDF features did not improve performance over simple counts.  This is likely because the dataset is small and the informative words (e.g. “refund”, “crashed”, “charged”, “delayed”) are already distinctive without re‑weighting.
* **ComplementNB not helpful:** Despite being designed for class imbalance, `ComplementNB` offered no benefit here.  The class imbalance is modest and the vocabulary is simple.
* **Failure to capture ambiguous messages:** The misclassifications primarily involved messages that mention multiple topics.  For example, “I was billed incorrectly and need a refund” mentions both billing and refund; Naive Bayes can only pick one class, leading to errors when the context is ambiguous.  This limitation could be mitigated by multi‑label classification or by allowing an “other/unknown” bucket in production.

Overall the TF‑IDF + Naive Bayes model serves as a solid, fast baseline for routing customer feedback.  However, to handle more nuanced messages or sarcasm a more expressive model (e.g. linear SVM or fine‑tuned transformer) might be necessary.