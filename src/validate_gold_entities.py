
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

GOLD_PATH = DATA_DIR / "gold_entities_review_draft.csv"
SAMPLES_PATH = DATA_DIR / "evaluation_samples.csv"
REVIEW_PATH = DATA_DIR / "entity_review_needed.csv"


# ============================================================
# VALIDATE ANNOTATIONS
# ============================================================

def main():

    print("\n=== GOLD ENTITY VALIDATION ===")

    gold = pd.read_csv(
        GOLD_PATH,
        keep_default_na=False
    )

    samples = pd.read_csv(
        SAMPLES_PATH,
        keep_default_na=False
    )

    if len(gold) != 120:
        raise ValueError(
            f"Expected 120 rows, found {len(gold)}"
        )

    if gold["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found")

    required = {
        "sample_id", "text", "gold_entities"
    }

    if not required.issubset(gold.columns):
        raise ValueError("Missing required columns")

    if not {"sample_id", "text"}.issubset(samples.columns):
        raise ValueError("Invalid evaluation samples file")

    merged = gold.merge(
        samples[["sample_id", "text"]],
        on="sample_id",
        how="outer",
        suffixes=("_gold", "_original"),
        indicator=True,
        validate="one_to_one"
    )

    if (
        len(merged) != len(gold)
        or (merged["_merge"] != "both").any()
        or (merged["text_gold"] != merged["text_original"]).any()
    ):
        raise ValueError(
            "Gold samples do not match evaluation samples"
        )

    counts = Counter()
    errors = []
    review_rows = []

    for _, row in gold.iterrows():

        sample_id = row["sample_id"]
        tweet = row["text"]
        raw = row["gold_entities"]

        try:
            entities = json.loads(raw)
        except json.JSONDecodeError:
            errors.append((sample_id, "Invalid JSON"))
            continue

        if not isinstance(entities, list):
            errors.append((sample_id, "Not a list"))
            continue

        seen = set()

        for entity in entities:

            if not isinstance(entity, dict):
                errors.append((sample_id, "Invalid entity"))
                continue

            if set(entity.keys()) != {"text", "type"}:
                errors.append((sample_id, "Invalid entity fields"))
                continue

            text = entity["text"]
            entity_type = entity["type"]

            if not isinstance(text, str) or not text:
                errors.append((sample_id, "Invalid entity text"))
                continue

            if entity_type not in ENTITY_TYPES:
                errors.append((sample_id, "Invalid entity type"))
                continue

            if text not in tweet:
                errors.append((
                    sample_id,
                    f"Entity not found: {text}"
                ))

            key = (text, entity_type)

            if key in seen:
                errors.append((
                    sample_id,
                    f"Duplicate entity: {text}"
                ))

            seen.add(key)
            counts[entity_type] += 1

            # Ambiguous annotations to review manually
            reason = None

            if entity_type == "other":
                reason = "Check whether this is a named entity"

            elif (
                entity_type == "flight_number"
                and (
                    text[0].isdigit()
                    or " " in text
                )
            ):
                reason = "Check flight number boundaries"

            if reason:
                review_rows.append({
                    "sample_id": sample_id,
                    "tweet": tweet,
                    "entity_text": text,
                    "entity_type": entity_type,
                    "reason": reason
                })

    print(f"\nSamples: {len(gold)}")
    print(f"Total entities: {sum(counts.values())}")

    print("\nEntity distribution:")

    for entity_type in ENTITY_TYPES:
        print(
            f"{entity_type}: {counts[entity_type]}"
        )

    print(f"\nValidation errors: {len(errors)}")

    for item in errors[:20]:
        print(item)

    review_df = pd.DataFrame(
        review_rows,
        columns=[
            "sample_id",
            "tweet",
            "entity_text",
            "entity_type",
            "reason"
        ]
    )

    review_df.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nAnnotations flagged for manual review: "
        f"{len(review_df)}"
    )

    print(f"Review file: {REVIEW_PATH}")

    if errors:
        print("\nValidation failed. Correct errors first.")
    else:
        print("\nStructural validation passed.")
        print("Manual annotation review is still required.")


if __name__ == "__main__":
    main()
