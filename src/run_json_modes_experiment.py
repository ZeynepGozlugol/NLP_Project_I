
import json

import pandas as pd
from tqdm.auto import tqdm

import config
import llm_client

from data import load_full, make_splits, balanced_sample
from final_metrics import final_sentiment_metrics
from json_parser import parse_llm_json, validate_output
from prompts import build_prompt


# ============================================================
# FINAL EXPERIMENT CONFIGURATION
# ============================================================

MODEL = "qwen2.5:7b"
VARIANT = "few_shot_v2"

# Two required experimental settings
MODES = ["none", "schema"]

# Small preliminary test: 2 per class = 6 tweets
# Final experiment: 40 per class = 120 tweets
N_PER_CLASS = 40

TEMPERATURE = 0.0
SEED = 42
SAMPLE_SEED = 2026


# ============================================================
# RUN MODEL ON ONE EXPERIMENTAL SETTING
# ============================================================

def run_mode(df, mode):

    rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"mode={mode}"
    ):

        prompt = build_prompt(
            row["text"],
            VARIANT
        )

        try:
            # Query the same model with identical parameters
            raw, latency = llm_client.query_ollama(
                prompt,
                mode=mode,
                temperature=TEMPERATURE,
                seed=SEED,
                model=MODEL
            )

            # Direct parsing of raw response
            try:
                json.loads(raw)
                raw_parse_ok = True
            except (json.JSONDecodeError, TypeError):
                raw_parse_ok = False

            # Parsing after cleaning
            data, stage = parse_llm_json(raw)

            parse_ok = data is not None

            # Schema validation
            if parse_ok:
                schema_ok, error = validate_output(data)
            else:
                schema_ok = False
                error = "JSON parsing failed"

            # Classification prediction
            prediction = (
                data["category"]
                if schema_ok
                else None
            )

            # Save extracted entities for later F1 evaluation
            entities = (
                json.dumps(
                    data["entities"],
                    ensure_ascii=False
                )
                if schema_ok
                else "[]"
            )

            rows.append({
                "sample_id": row["sample_id"],
                "text": row["text"],
                "true_label": row["true_label"],
                "pred_label": prediction,
                "pred_entities": entities,
                "raw_output": raw,
                "parse_stage": stage,
                "raw_parse_ok": raw_parse_ok,
                "parse_ok": parse_ok,
                "schema_ok": schema_ok,
                "latency_s": latency,
                "mode": mode,
                "error": error
            })

        except Exception as exc:

            # Handle model/API errors without stopping the test
            rows.append({
                "sample_id": row["sample_id"],
                "text": row["text"],
                "true_label": row["true_label"],
                "pred_label": None,
                "pred_entities": "[]",
                "raw_output": None,
                "parse_stage": "request_failed",
                "raw_parse_ok": False,
                "parse_ok": False,
                "schema_ok": False,
                "latency_s": None,
                "mode": mode,
                "error": f"{type(exc).__name__}: {exc}"
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n=== NLP PROJECT I: FINAL JSON COMPARISON ===")

    print(f"Model: {MODEL}")
    print(f"Prompt variant: {VARIANT}")
    print(f"Modes: {MODES}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Seed: {SEED}")
    print(f"Context length: {llm_client.CONTEXT_LENGTH}")

    # Verify installed model
    llm_client.check_model(MODEL)

    # Load dataset and preserve the original test split
    _, test = make_splits(load_full())

    # Select exactly the same samples for both settings
    sample = balanced_sample(
        test,
        N_PER_CLASS,
        seed=SAMPLE_SEED
    ).copy()

    sample["sample_id"] = range(len(sample))

    print(f"\nNumber of samples: {len(sample)}")
    print(f"Expected model calls: {len(sample) * len(MODES)}")

    # Separate directory for preliminary and final results
    experiment_name = (
        "pilot"
        if N_PER_CLASS < 34
        else "final"
    )

    output_dir = (
        config.RESULTS_DIR
        / "final_json_comparison"
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save exact evaluation samples
    sample.to_csv(
        output_dir / "evaluation_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Save reproducibility metadata
    settings = {
        "model": MODEL,
        "model_quantization": "Q4_K_M",
        "ollama_version": "0.40.16",
        "prompt_variant": VARIANT,
        "modes": MODES,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "sample_seed": SAMPLE_SEED,
        "context_length": llm_client.CONTEXT_LENGTH,
        "n_samples": len(sample),
        "n_per_class": N_PER_CLASS
    }

    with open(
        output_dir / "experiment_settings.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            settings,
            f,
            indent=2,
            ensure_ascii=False
        )

    summary = []

    # Compare the same samples across both modes
    for mode in MODES:

        print(f"\nRunning mode: {mode}")

        results = run_mode(sample, mode)

        # Save all outputs including failures
        result_path = (
            output_dir / f"qwen_few_shot_v2_{mode}.csv"
        )

        results.to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig"
        )

        # Classification evaluation including invalid outputs
        metrics = final_sentiment_metrics(results)

        # JSON and classification metrics
        summary.append({
            "mode": mode,
            "n": len(results),

            "accuracy_%": metrics["accuracy_%"],
            "macro_precision_%": metrics["macro_precision_%"],
            "macro_recall_%": metrics["macro_recall_%"],
            "macro_f1_%": metrics["macro_f1_%"],

            "raw_json_%": round(
                results["raw_parse_ok"].mean() * 100,
                2
            ),

            "parse_success_%": round(
                results["parse_ok"].mean() * 100,
                2
            ),

            "schema_valid_%": round(
                results["schema_ok"].mean() * 100,
                2
            ),

            "request_failures": int(
                (results["parse_stage"] == "request_failed").sum()
            ),

            "median_latency_s": round(
                results["latency_s"].median(),
                3
            )
        })

        print(f"Detailed results saved: {result_path}")

    # Final comparison table
    summary_df = pd.DataFrame(summary)

    summary_path = (
        output_dir / "json_modes_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== FINAL COMPARISON ===")
    print(summary_df.to_string(index=False))

    print(f"\nSummary saved: {summary_path}")
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
