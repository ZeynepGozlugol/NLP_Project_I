
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from config import LABELS


def final_sentiment_metrics(results):
    """
    Evaluate sentiment classification on all samples.

    Invalid predictions count as classification errors
    and are included in F1 calculations.
    """

    if results.empty:
        raise ValueError("No samples to evaluate.")

    y_true = results["true_label"].tolist()

    # Missing predictions get a special error label.
    # This label is not a valid sentiment class.
    y_pred = (
        results["pred_label"]
        .fillna("__INVALID__")
        .tolist()
    )

    if not set(y_true).issubset(set(LABELS)):
        raise ValueError("Unexpected ground-truth label")

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy_%": round(accuracy * 100, 2),
        "macro_precision_%": round(precision * 100, 2),
        "macro_recall_%": round(recall * 100, 2),
        "macro_f1_%": round(macro_f1 * 100, 2)
    }


def main():
    # Test: one missing prediction among six samples
    example = pd.DataFrame({
        "true_label": [
            "positive", "negative", "neutral",
            "positive", "negative", "neutral"
        ],
        "pred_label": [
            "positive", "negative", "neutral",
            "positive", None, "neutral"
        ]
    })

    metrics = final_sentiment_metrics(example)

    print("=== FINAL CLASSIFICATION METRICS TEST ===")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report:")

    print(
        classification_report(
            example["true_label"],
            example["pred_label"].fillna("__INVALID__"),
            labels=LABELS,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
