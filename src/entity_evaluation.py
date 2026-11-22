
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from config import ENTITY_TYPES


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "results"
    / "final_json_comparison"
    / "final"
)

GOLD_PATH = DATA_DIR / "gold_entities.csv"

MODES = ["none", "schema"]


# ============================================================
# ENTITY NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Exact match after whitespace normalization.

    Matching remains case-sensitive.
    Entity type must also match.
    """
    return " ".join(text.strip().split())


def entity_counter(entities):
    """
    Represent entities as (text, type) pairs.

    Counter preserves repeated mentions if annotated.
    """
    return Counter(
        (normalize_text(entity["text"]), entity["type"])
        for entity in entities
    )


# ============================================================
# ENTITY-LEVEL EVALUATION
# ============================================================

def evaluate_mode(gold, predictions):

    merged = gold.merge(
        predictions,
        on="sample_id",
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if (
        len(merged) != len(gold)
        or (merged["_merge"] != "both").any()
    ):
        raise ValueError(
            "Gold and prediction sample IDs do not match"
        )

    if (merged["text_x"] != merged["text_y"]).any():
        raise ValueError(
            "Gold and prediction tweet texts do not match"
        )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    details = []

    for _, row in merged.iterrows():

        gold_entities = json.loads(row["gold_entities"])
        pred_entities = json.loads(row["pred_entities"])

        gold_counts = entity_counter(gold_entities)
        pred_counts = entity_counter(pred_entities)

        # Correctly predicted entities
        tp = sum(
            (gold_counts & pred_counts).values()
        )

        # Incorrectly predicted entities
        fp = sum(
            (pred_counts - gold_counts).values()
        )

        # Missed gold entities
        fn = sum(
            (gold_counts - pred_counts).values()
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        details.append({
            "sample_id": row["sample_id"],
            "text": row["text_x"],
            "gold_entities": row["gold_entities"],
            "pred_entities": row["pred_entities"],
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    metrics = {
        "samples": len(merged),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "entity_precision_%": round(precision * 100, 2),
        "entity_recall_%": round(recall * 100, 2),
        "entity_f1_%": round(f1 * 100, 2)
    }

    return metrics, pd.DataFrame(details)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== ENTITY EXTRACTION EVALUATION ===")

    gold = pd.read_csv(
        GOLD_PATH,
        keep_default_na=False
    )

    if len(gold) != 120:
        raise ValueError("Expected exactly 120 gold samples")

    if gold["gold_entities"].eq("").any():
        raise ValueError("Some gold samples are not annotated")

    summaries = []

    for mode in MODES:

        prediction_path = (
            DATA_DIR / f"qwen_few_shot_v2_{mode}.csv"
        )

        predictions = pd.read_csv(
            prediction_path,
            keep_default_na=False
        )

        # Keep only columns required for entity evaluation
        predictions = predictions[
            ["sample_id", "text", "pred_entities"]
        ]

        metrics, details = evaluate_mode(
            gold,
            predictions
        )

        metrics["mode"] = mode
        summaries.append(metrics)

        details.to_csv(
            DATA_DIR / f"entity_details_{mode}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nMode: {mode}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    summary = pd.DataFrame(summaries)

    summary.to_csv(
        DATA_DIR / "entity_f1_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== FINAL ENTITY F1 COMPARISON ===")
    print(summary.to_string(index=False))

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
