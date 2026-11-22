
import json
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

INPUT_PATH = DATA_DIR / "gold_entities_review_draft.csv"
OUTPUT_PATH = DATA_DIR / "gold_entities.csv"


# ============================================================
# MANUAL ANNOTATION CORRECTIONS
# ============================================================

# Remove items that do not belong to our entity task.
REMOVE = {
    32: ["LE2V9D"],
    35: ["b1"],
    41: ["#F8NEQM"],
    82: ["Mother Nature"],
}

# Correct the boundaries of flight numbers.
REPLACE = {
    35: {
        "736 flight": "736",
        "917 one": "917",
    }
}


# ============================================================
# FINALIZE GOLD ANNOTATIONS
# ============================================================

def main():

    print("=== FINALIZE GOLD ENTITIES ===")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Review it before replacing."
        )

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    if len(df) != 120:
        raise ValueError("Expected exactly 120 samples")

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs")

    final_annotations = []
    removed_count = 0
    replaced_count = 0

    for _, row in df.iterrows():

        sample_id = int(row["sample_id"])
        tweet = row["text"]

        entities = json.loads(
            row["gold_entities"]
        )

        corrected = []

        for entity in entities:

            entity = entity.copy()
            entity_text = entity["text"]

            # Remove excluded annotations
            if entity_text in REMOVE.get(sample_id, []):
                removed_count += 1
                continue

            # Correct entity boundaries
            replacements = REPLACE.get(sample_id, {})

            if entity_text in replacements:
                entity["text"] = replacements[entity_text]
                replaced_count += 1

            # Validate the final entity
            if set(entity) != {"text", "type"}:
                raise ValueError(
                    f"Invalid fields at sample {sample_id}"
                )

            if entity["type"] not in ENTITY_TYPES:
                raise ValueError(
                    f"Invalid type at sample {sample_id}"
                )

            if entity["text"] not in tweet:
                raise ValueError(
                    f"Entity not in tweet: {entity['text']}"
                )

            corrected.append(entity)

        final_annotations.append(
            json.dumps(
                corrected,
                ensure_ascii=False
            )
        )

    df["gold_entities"] = final_annotations

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    total_entities = sum(
        len(json.loads(value))
        for value in df["gold_entities"]
    )

    print(f"Samples: {len(df)}")
    print(f"Removed entities: {removed_count}")
    print(f"Corrected boundaries: {replaced_count}")
    print(f"Final entity count: {total_entities}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
