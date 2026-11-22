
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import config
import llm_client
from data import load_full, make_splits, balanced_sample
from evaluate import sentiment_metrics
from json_parser import parse_llm_json, validate_output
from prompts import build_prompt


MODEL = "qwen2.5:7b"
VARIANT = "few_shot_v2"
MODE = "schema"
TEMPERATURES = [0.0, 0.3, 0.7]
N_PER_CLASS = 15  # Her sınıftan 15 tweet: toplam 45 tweet


def run_temperature_experiment(df, temperature):
    rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"temperature={temperature}"
    ):
        prompt = build_prompt(row["text"], VARIANT)

        try:
            # Her sıcaklıkta yalnızca bir deneme yapıyoruz.
            # Böylece retry sıcaklığı sonucu etkilemiyor.
            raw, latency = llm_client.query_ollama(
                prompt,
                mode=MODE,
                temperature=temperature,
                seed=config.SEED,
                model=MODEL,
            )

            data, parse_stage = parse_llm_json(raw)

            if data is None:
                schema_ok, error = False, "JSON parse failed"
            else:
                schema_ok, error = validate_output(data)

            rows.append({
                "text": row["text"],
                "true_label": row["true_label"],
                "pred_label": (
                    data["category"] if schema_ok else None
                ),
                "sentiment_score": (
                    data["sentiment_score"] if schema_ok else None
                ),
                "temperature": temperature,
                "parse_stage": parse_stage,
                "schema_ok": schema_ok,
                "latency_s": latency,
                "error": error,
            })

        except Exception as exc:
            rows.append({
                "text": row["text"],
                "true_label": row["true_label"],
                "pred_label": None,
                "sentiment_score": None,
                "temperature": temperature,
                "parse_stage": "failed",
                "schema_ok": False,
                "latency_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return pd.DataFrame(rows)


def main():
    llm_client.check_model(MODEL)

    # Ana projede kullandığımız aynı test bölmesini oluştur.
    _, test = make_splits(load_full())

    # Her sıcaklık için aynı, dengeli 45 tweet kullanılacak.
    sample = balanced_sample(
        test,
        n_per_class=N_PER_CLASS,
        seed=2026,
    )

    output_dir = config.RESULTS_DIR / "temperature"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for temperature in TEMPERATURES:
        results = run_temperature_experiment(sample, temperature)

        # Her sıcaklık ayrı dosyaya kaydedilir.
        temp_name = str(temperature).replace(".", "_")
        output_path = output_dir / f"qwen_few_shot_v2_temp_{temp_name}.csv"
        results.to_csv(output_path, index=False, encoding="utf-8-sig")

        metrics = sentiment_metrics(results)
        json_valid = results["schema_ok"].mean() * 100

        summary.append({
            "temperature": temperature,
            "n": len(results),
            "accuracy_%": metrics["accuracy_%"],
            "macro_f1_%": metrics["macro_f1_%"],
            "json_schema_valid_%": round(json_valid, 2),
            "median_latency_s": results["latency_s"].median(),
        })

        print(f"\nKaydedildi: {output_path}")

    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "temperature_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== TEMPERATURE DENEYİ SONUÇLARI ===")
    print(summary_df.to_string(index=False))
    print(f"\nÖzet kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()
