# 04 – Error analysis and improvements

The baseline and main models achieved good performance on the synthetic dataset, but real‑world systems must handle ambiguous or unusual messages gracefully.  To study model errors I deliberately reduced the training data to 20 % of the deduplicated dataset and evaluated on the remaining 80 %.  This created a larger set of misclassifications to analyse.

## Misclassified examples

The under‑trained TF‑IDF + MultinomialNB model misclassified 31 out of 90 test samples (≈34 %).  The first 20 misclassified messages are listed below along with their true and predicted labels:

1. **other → delivery_issue:** “How do I change my account password? Just curious”
2. **other → app_bug:** “Is there a way to report bugs in the app?”
3. **billing_problem → app_bug:** “I want to update my payment method but your app keeps glitching”
4. **other → delivery_issue:** “How do I change my account password?”
5. **app_bug → refund_request:** “The checkout screen is glitching and charged me twice”
6. **other → app_bug:** “What should I do if the app crashes during checkout?”
7. **billing_problem → delivery_issue:** “Why was I charged extra for shipping when it was supposed to be free?”
8. **refund_request → delivery_issue:** “The app crashed and I had to place my order twice, please refund the duplicate charge”
9. **app_bug → billing_problem:** “The app shows an error when I request a refund for a delayed package”
10. **billing_problem → delivery_issue:** “The delivery never arrived yet my card was charged”
11. **billing_problem → refund_request:** “I was charged for a product that never shipped”
12. **other → delivery_issue:** “How do I change my account password? No urgent issue”
13. **other → refund_request:** “I'd like to know why my package is late”
14. **other → refund_request:** “Why was I charged tax separately?”
15. **delivery_issue → refund_request:** “My package arrived late and I was charged extra for shipping”
16. **app_bug → billing_problem:** “I'm getting an error message when I log in through the mobile app. Please fix this glitch”
17. **billing_problem → delivery_issue:** “Why was I charged extra for shipping when it was supposed to be free? I expected a discount”
18. **billing_problem → delivery_issue:** “Why was I charged extra for shipping when it was supposed to be free? Can someone explain this charge?”
19. **app_bug → delivery_issue:** “When I track my delivery in the app it crashes”
20. **billing_problem → refund_request:** “I updated my payment method but I'm still being billed on the old card. Can someone explain this charge?”

These errors fall into a few patterns:

* **Lack of strong keywords:** Generic questions like changing a password or asking about tax don’t contain distinctive words, so the model defaults to common classes like `delivery_issue`.
* **Multi‑issue messages:** Sentences that mention more than one category (e.g. delivery *and* refund) confuse the single‑label classifier.  The model must pick one class and often chooses the majority class present in training.
* **Keyword ambiguity:** Words like “charged” or “glitch” appear in both billing and app bug contexts.  The model may latch onto the wrong context when the message is short.

## Hypotheses for selected errors

For five of the misclassifications above I hypothesised why the model failed:

1. **`other` → `delivery_issue`** – Short generic questions (e.g. about password changes) lack category‑specific words.  With no strong signals the classifier falls back to the most common class.
2. **`billing_problem` → `app_bug`** – Messages mentioning “update” or “glitching” trigger the app bug class even though the primary issue is billing.
3. **`refund_request` → `delivery_issue`** – Sentences containing both refund and delivery words (e.g. “refund” and “arrived late”) confuse the model; it chooses delivery because that class has more training examples.
4. **`app_bug` → `billing_problem`** – Error messages about the app may include “payment” or “charged” which are strongly associated with billing.
5. **`other` → `refund_request`** – Questions like “Why was I charged tax separately?” include the word “charged,” pushing the model toward refund‑related classes even though the tone is informational.

## Informative features

By comparing the log probabilities of words across classes I extracted the most indicative features for each category.  For example:

* **app_bug:** words like *update*, *caused*, *search*, *working* were strong signals.
* **billing_problem:** phrases such as *subscription renewal*, *payment method*, *renewal*, *unexpected fee* drove predictions.
* **delivery_issue:** words like *arrived*, *delivery*, *damaged*, *order* dominated.
* **refund_request:** expressions like *need refund*, *was defective*, *money back* were highly indicative.
* **other:** generic verbs such as *provide*, *information*, *about* had slightly higher weights but overall this class lacked strong features.

## Targeted improvement attempt

Given that class imbalance and ambiguous keywords contributed to errors, I tried switching from `MultinomialNB` to `ComplementNB`, which estimates feature statistics on all other classes and often handles imbalance better.  Using the same TF‑IDF features and the 20/80 split, the baseline model achieved **macro F1 ≈ 0.63** and **accuracy ≈ 0.66**.  `ComplementNB` improved macro F1 to **≈ 0.68** and accuracy to **≈ 0.69**, a modest but consistent gain.  Character 3–5‑gram TF‑IDF features were also tested but did not improve macro F1.  The confusion matrix for the improved model is shown below:

![Confusion matrix – ComplementNB improvement]({{file:file-WyTWfns3Z6cwG5zFY6Ear8}})

The improvements mainly reduced confusion between `billing_problem` and other classes.  However, many ambiguous messages remained challenging.  Since the dataset is small and synthetic it is hard to push performance much further without more data or more expressive models.

## Business and UX implications

In a real support routing system misclassifications have tangible consequences.  Routing a billing problem to the delivery team delays resolution and frustrates customers; sending a refund request to technical support may postpone a reimbursement.  To mitigate these risks:

* **Confidence thresholds:** Only auto‑route messages when the predicted probability exceeds a threshold; otherwise send to a general queue for human triage.
* **“Other/unknown” bucket:** When the model is uncertain or no class exceeds a confidence threshold, assign the message to an “other” category for manual review.
* **Human‑in‑the‑loop:** Continuously collect misrouted tickets and use them to refine the model.  Agents could reassign categories, providing labelled data for retraining.
* **Multi‑label classification:** Some messages genuinely span multiple issues.  A multi‑label model could assign several categories or flags, allowing parallel routing.

These strategies, combined with periodic model updates, help ensure that automated classification aids rather than hinders customer support.