# AI Exploration — Customer Feedback Classification

This repository contains a small end‑to‑end natural language processing (NLP) project.  The goal is to build a simple but complete pipeline that classifies short customer feedback messages into high‑level complaint categories.  It demonstrates exploratory data analysis (EDA), baseline modelling, feature engineering, model training, error analysis, and a minimal command‑line prediction script.

## Project structure

The project uses a conventional layout to keep data, code, models and notes separate:

```
ai‑exploration‑customer‑feedback/
├── data/          # datasets used by the project
│   ├── raw/       # original unmodified data files
│   │   └── customer_feedback.csv
│   └── processed/ # cleaned and split data (train/validation/test)
├── notebooks/     # Jupyter notebooks used for exploratory analysis and modelling
├── src/           # reusable Python modules and scripts (preprocess, training, evaluation, prediction)
├── models/        # persisted models and vectorizers (*.joblib)
├── reports/
│   └── figures/   # plots generated during the analysis
├── notes/         # human‑readable narrative logs (planning, data exploration, modelling, error analysis, takeaways)
│   ├── assets/    # screenshots or small video clips referenced in the notes
├── requirements.txt
└── README.md      # this file
```

Each subfolder and script is described in detail in the accompanying notes.  Consult the Markdown files in `notes/` for plain‑language explanations of the decisions made at each stage.

## Getting started

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-exploration-customer-feedback
```

### 2. Set up a virtual environment

Creating a dedicated Python environment avoids conflicts with other projects.  Any environment manager (such as `venv`, `conda` or `pipenv`) works; here is an example using the built‑in `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the notebooks

The notebooks under `notebooks/` walk through the entire process:

1. **01_eda.ipynb** – inspect the dataset, explore class distribution and text characteristics.
2. **02_baselines_and_features.ipynb** – implement a majority‑class baseline and a simple Bag‑of‑Words + Naive Bayes baseline; compare basic preprocessing options.
3. **03_model_nb_tfidf.ipynb** – build and tune a TF‑IDF + Naive Bayes model, perform validation, and save the best vectorizer and classifier to the `models/` directory.
4. **04_error_analysis_and_improvements.ipynb** – review misclassified examples, look at the most informative features per class, and try one small improvement (e.g. character n‑grams or ComplementNB).

Open a Jupyter environment (for example with `jupyter notebook` or `jupyter lab`), navigate to the `notebooks/` folder, and execute each notebook sequentially.  Plots and reports are saved automatically to `reports/figures/`.

### 4. Command‑line prediction

After training the model, you can classify new feedback messages via the CLI script:

```bash
python src/predict.py --text "App keeps crashing when I try to pay"
```

The script loads the saved vectorizer and classifier from the `models/` directory, transforms the input text, and prints the predicted category along with class probabilities.

## Data

Finding a publicly available dataset of customer feedback annotated with high‑level complaint categories (e.g. *delivery issue*, *refund request*, *billing problem*, *app bug*) proved difficult.  Public sources such as the U.S. Consumer Financial Protection Bureau’s complaint database describe financial product complaints, but their topics differ from delivery/billing/app issues, and the licence was unspecified【56518301983924†L85-L103】.  Other datasets on Kaggle or GitHub required user accounts or were inaccessible in this environment.  To proceed with the project we therefore generated a synthetic dataset.

The synthetic dataset contains 500 messages distributed across five categories with intentional class imbalance.  Each sample combines typical phrases customers use when complaining about deliveries, requesting refunds, questioning billing errors, reporting application bugs, or asking miscellaneous questions.  See `notes/01_data.md` for details about how the dataset was generated.

## Licence

This project’s code is released under the MIT licence.  The synthetic dataset was created solely for demonstration purposes and may be reused freely.

## Contributing

Contributions are welcome!  To report issues or suggest improvements, please open an issue or submit a pull request.