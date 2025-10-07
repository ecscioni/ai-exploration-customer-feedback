# 00 – Project plan

## Goal

The aim of this mini‑project is to build a simple yet complete NLP pipeline that automatically classifies short customer feedback messages into a handful of high‑level complaint categories (for example *delivery issue*, *refund request*, *billing problem*, *app bug*, or *other*).  The end product should include an exploratory analysis of the data, a couple of baseline models, a tuned Naive Bayes classifier with TF‑IDF features, an error analysis with a small improvement attempt, and a command‑line script for making predictions.

## Approach

1. **Data acquisition:**  Search for an existing dataset of customer feedback labelled with complaint categories.  The ideal dataset would be small (≲10k rows), multi‑class, permissively licensed, and contain categories relevant to deliveries, refunds, billing, or app issues.  Public data sources such as the U.S. Consumer Financial Protection Bureau’s complaint database were examined, but they cover financial products rather than the desired categories【56518301983924†L85-L103】.  Many Kaggle or GitHub datasets were inaccessible or required accounts.  Given these constraints a synthetic dataset will be generated with at least 500 samples and five categories to demonstrate the pipeline.

2. **Data processing:**  Load the raw dataset from `data/raw/customer_feedback.csv`, clean it if necessary (e.g. remove duplicates, drop empty strings, check for non‑English characters), and split it into training and validation sets.  Save processed splits under `data/processed/` for reproducibility.  Implement any reusable cleaning functions in `src/preprocess.py`.

3. **Exploratory data analysis (EDA):**  In `notebooks/01_eda.ipynb` inspect the dataset shape, per‑class counts, and basic text statistics such as message lengths.  Visualise the class imbalance and distribution of text lengths using bar charts/histograms.  Note any issues such as duplicates or missing values and decide how to handle them.  Document findings in `notes/01_data.md`.

4. **Baselines and feature experiments:**  In `notebooks/02_baselines_and_features.ipynb` establish simple baselines:
   - Majority‑class predictor: always predict the most common class.  This sets a low bar for accuracy and F1 scores.
   - Bag‑of‑Words features (via `CountVectorizer`) + Multinomial Naive Bayes.  Use basic preprocessing like lowercasing and punctuation removal.  Compare the effect of stopword removal and different n‑gram ranges (word 1‑2 grams and character 3–5 grams).  Record macro and micro F1, accuracy, and confusion matrices.  Save a classification report and confusion matrix figure to `reports/`.
   - Document the impact of each feature setting in `notes/02_baseline.md`.

5. **Main model (TF‑IDF + Naive Bayes):**  In `notebooks/03_model_nb_tfidf.ipynb` build the main classifier using `TfidfVectorizer`.  Use a simple grid search over smoothing hyperparameter `alpha` for both `MultinomialNB` and `ComplementNB`.  Perform a stratified train/validation split (e.g. 80/20) or cross‑validation.  Save the best vectorizer and classifier to `models/vectorizer.joblib` and `models/classifier.joblib`.  Produce a confusion matrix and per‑class performance metrics.  Write plain‑language explanations of Naive Bayes assumptions and limitations in `notes/03_modeling.md`, including a small worked example.

6. **Error analysis and improvements:**  In `notebooks/04_error_analysis_and_improvements.ipynb` examine at least 20 misclassified examples, noting the true label, predicted label and message.  For five of these, hypothesise in one sentence why the model failed (e.g. ambiguous phrasing, sarcasm, multi‑issue text).  Compute and display the most informative features for each class using log probability ratios.  Attempt one targeted improvement (such as adding character n‑grams or using `ComplementNB` to mitigate class imbalance) and report whether macro F1 improves.  Discuss why the improvement did or did not help in `notes/04_error_analysis.md`.

7. **Narrative documentation:**  For each stage write a Markdown note summarising what was done, why, and what was learned.  Be honest about failures: if a certain feature did not help, record that.  Use simple language suitable for a beginner.  Include at least two screenshots of key plots by saving them into `notes/assets` and referencing them in the notes.

8. **Command‑line prediction:**  Implement `src/predict.py` so that it loads the saved model and vectorizer and prints the predicted category and class probabilities for a given text.  Document usage in the README.

9. **Reproducibility and packaging:**  Record the exact package versions in `requirements.txt` and set random seeds for splits.  Provide clear setup and run instructions in the README.  Optionally zip the entire directory for download when everything is complete.

## Success criteria

Given the synthetic nature of the dataset the absolute performance thresholds are arbitrary.  However the following will guide evaluation:

* The majority‑class baseline should achieve a macro F1 of roughly 0.2–0.3.  A Bag‑of‑Words + Naive Bayes model should improve over this baseline significantly.
* The TF‑IDF + Naive Bayes model should achieve a macro F1 of at least 0.6 on the validation set.  A small improvement (e.g. ComplementNB or character n‑grams) should be explored and retained only if it demonstrably improves macro F1.
* All notebooks should run end‑to‑end without errors, saving required artefacts (classification reports, confusion matrices, models).  The CLI should accept a text string and return a sensible prediction.

## Constraints

* **Dataset availability:**  Public datasets with appropriate categories were difficult to access.  A synthetic dataset will be used instead; its limitations (lack of real‑world nuance, limited vocabulary) should be acknowledged.
* **Environment:**  Only common open source Python libraries (pandas, numpy, scikit‑learn, matplotlib) may be used.  Deep learning models or external APIs are out of scope.  Everything must run within this environment without internet access once the dataset is created.

With this plan in place, the next step is to document the data creation and exploratory analysis.