"""
CLI for predicting the category of a customer feedback message.

Usage:

```sh
python src/predict.py --text "App keeps crashing when I try to pay"
```

It will load the vectorizer and classifier saved in the `models/` directory and
print the predicted label along with class probabilities.
"""

from __future__ import annotations

import argparse
import joblib
import numpy as np


def predict(text: str, vectorizer_path: str, model_path: str) -> None:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    X_vec = vectorizer.transform([text])
    proba = model.predict_proba(X_vec)[0]
    classes = model.classes_
    predicted = classes[np.argmax(proba)]
    print(f"Predicted category: {predicted}")
    # Print probabilities
    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        print(f"{cls}: {p:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Predict the category of a customer feedback message.')
    parser.add_argument('--text', required=True, help='Text of the customer feedback message')
    parser.add_argument('--vectorizer', default='models/vectorizer.joblib', help='Path to saved vectorizer')
    parser.add_argument('--model', default='models/classifier.joblib', help='Path to saved classifier')
    args = parser.parse_args()
    predict(args.text, args.vectorizer, args.model)


if __name__ == '__main__':
    main()