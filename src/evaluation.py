"""Evaluation metrics, model rankings, and visualization helpers will be added here."""

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)


def evaluate_model(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Minority-Class Recall": recall_score(
            y_true, y_pred,
            pos_label="spam",
            zero_division=0
        ),

        "Macro-F1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "MCC": matthews_corrcoef(y_true, y_pred)
    }
