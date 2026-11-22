
import json
from pathlib import Path

import pandas as pd

import config
import llm_client

from prompts import build_prompt
from json_parser import parse_llm_json, validate_output


# ============================================================
# LIVE DEMO CONFIGURATION
# ============================================================

MODEL = "qwen2.5:7b"
VARIANT = "few_shot_v2"

MODES = ["none", "schema"]
TEMPERATURES = [0.0, 1.0]
SEED = 42


# Five new samples created for the live demonstration.
# These examples are not copied from the evaluation dataset.

NEW_SAMPLES = [
    {
        "id": 1,
        "text": (
            "@SkyJetAir Thanks for the smooth flight from Madrid, "
            "boarding took five minutes."
        ),
        "expected_sentiment": "positive"
    },
    {
        "id": 2,
        "text": (
            "@FlyNova Flight FN482 from Paris was cancelled "
            "and nobody told us anything."
        ),
        "expected_sentiment": "negative"
    },
    {
        "id": 3,
        "text": (
            "@FlyNova can you resend my booking code QX7R2P? "
            "Also is gate C14 correct for the Rome flight?"
        ),
        "expected_sentiment": "neutral"
    },
    {
        "id": 4,
        "text": (
            "@SkyJetAir Thanks for letting me practice my patience "
            "for six hours. Very kind of you."
        ),
        "expected_sentiment": "negative"
    },
    {
        "id": 5,
        "text": (
            "@SkyJetAir Boarding was announced for 6:00. "
            "It is 9:40 and our plane has not arrived."
        ),
        "expected_sentiment": "negative"
    }
]


# ============================================================
# EXPERIMENT
# ============================================================

def main():

    print("\n=== NLP PROJECT I: LIVE DEMO ===")

    llm_client.check_model(MODEL)

    output_dir = (
        config.RESULTS_DIR / "live_demo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify that the samples are not in our test set.
    evaluation_path = (
        config.RESULTS_DIR
        / "final_json_comparison"
        / "final"
        / "evaluation_samples.csv"
    )

    evaluation = pd.read_csv(evaluation_path)

    existing_texts = set(
        evaluation["text"].astype(str).str.strip()
    )

    for sample in NEW_SAMPLES:
        if sample["text"].strip() in existing_texts:
            raise ValueError(
                f"Sample {sample['id']} exists in evaluation data"
            )

    rows = []

    # Same samples for both temperature values and modes.
    for sample in NEW_SAMPLES:

        print(f"\n{'=' * 60}")
        print(f"SAMPLE {sample['id']}")
        print(f"Tweet: {sample['text']}")
        print(
            f"Expected sentiment: "
            f"{sample['expected_sentiment']}"
        )

        prompt = build_prompt(
            sample["text"],
            VARIANT
        )

        for mode in MODES:
            for temperature in TEMPERATURES:

                print(
                    f"\nMode: {mode} | "
                    f"Temperature: {temperature}"
                )

                try:
                    raw, latency = llm_client.query_ollama(
                        prompt,
                        mode=mode,
                        temperature=temperature,
                        seed=SEED,
                        model=MODEL
                    )

                    try:
                        json.loads(raw)
                        raw_parse_ok = True
                    except (json.JSONDecodeError, TypeError):
                        raw_parse_ok = False

                    data, stage = parse_llm_json(raw)

                    parse_ok = data is not None

                    if parse_ok:
                        schema_ok, error = validate_output(data)
                    else:
                        schema_ok = False
                        error = "JSON parsing failed"

                    prediction = (
                        data["category"]
                        if schema_ok
                        else None
                    )

                    correct = (
                        prediction == sample["expected_sentiment"]
                    )

                    print(f"Prediction: {prediction}")
                    print(f"Correct: {correct}")
                    print(f"Raw JSON: {raw_parse_ok}")
                    print(f"Parse stage: {stage}")
                    print(f"Schema valid: {schema_ok}")
                    print(f"Latency: {latency:.2f} seconds")

                    print("Model output:")
                    print(raw)

                    rows.append({
                        "sample_id": sample["id"],
                        "text": sample["text"],
                        "expected_sentiment": (
                            sample["expected_sentiment"]
                        ),
                        "mode": mode,
                        "temperature": temperature,
                        "pred_label": prediction,
                        "correct": correct,
                        "raw_parse_ok": raw_parse_ok,
                        "parse_ok": parse_ok,
                        "schema_ok": schema_ok,
                        "parse_stage": stage,
                        "raw_output": raw,
                        "latency_s": latency,
                        "error": error
                    })

                except Exception as exc:

                    print(f"ERROR: {exc}")

                    rows.append({
                        "sample_id": sample["id"],
                        "text": sample["text"],
                        "expected_sentiment": (
                            sample["expected_sentiment"]
                        ),
                        "mode": mode,
                        "temperature": temperature,
                        "pred_label": None,
                        "correct": False,
                        "raw_parse_ok": False,
                        "parse_ok": False,
                        "schema_ok": False,
                        "parse_stage": "request_failed",
                        "raw_output": None,
                        "latency_s": None,
                        "error": str(exc)
                    })

    results = pd.DataFrame(rows)

    results.to_csv(
        output_dir / "live_demo_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        results.groupby(
            ["mode", "temperature"],
            as_index=False
        )
        .agg(
            samples=("sample_id", "count"),
            correct_predictions=("correct", "sum"),
            raw_json_rate=("raw_parse_ok", "mean"),
            parse_rate=("parse_ok", "mean"),
            schema_rate=("schema_ok", "mean")
        )
    )

    for column in [
        "raw_json_rate",
        "parse_rate",
        "schema_rate"
    ]:
        summary[column] = (
            summary[column] * 100
        ).round(2)

    summary["accuracy_%"] = (
        summary["correct_predictions"]
        / summary["samples"]
        * 100
    ).round(2)

    summary.to_csv(
        output_dir / "live_demo_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== LIVE DEMO SUMMARY ===")
    print(summary.to_string(index=False))

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
