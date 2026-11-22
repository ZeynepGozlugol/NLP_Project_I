
from pathlib import Path
import pandas as pd


# ============================================================
# GOLD ENTITY ANNOTATION TEMPLATE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "results"
    / "final_json_comparison"
    / "final"
    / "evaluation_samples.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "results"
    / "final_json_comparison"
    / "final"
    / "gold_entities.csv"
)


def main():

    print("=== GOLD ENTITY ANNOTATION ===")

    # Load the exact evaluation samples
    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id", "text", "true_label"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Missing required columns in evaluation samples."
        )

    if len(df) != 120:
        raise ValueError(
            f"Expected 120 samples, found {len(df)}."
        )

    # Avoid overwriting manually annotated data
    if OUTPUT_PATH.exists():
        print(f"File already exists: {OUTPUT_PATH}")
        print("Existing annotations were not overwritten.")
        return

    # Create an empty annotation column
    # Blank means NOT ANNOTATED.
    # [] means ANNOTATED, no entities found.
    annotation_df = df[
        ["sample_id", "text", "true_label"]
    ].copy()

    annotation_df["gold_entities"] = ""

    # Save the annotation template
    annotation_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Number of samples: {len(annotation_df)}")
    print(f"Annotation file created: {OUTPUT_PATH}")
    print("\nFirst five samples:")

    print(
        annotation_df[
            ["sample_id", "text"]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
