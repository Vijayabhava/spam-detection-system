
"""Classifier definitions and training helpers."""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def build_models():

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True
            )),
            ("classifier", MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True
            )),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True
            )),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(
                    random_state=42,
                    dual="auto"
                ),
                cv=3
            ))
        ])
    }

    return models
