
import pandas as pd
from pathlib import Path

from config import MODELS
from json_parser import parse_llm_json, validate_output
from prompts import build_prompt
from llm_client import query_ollama


tweets = [
    # İlk 3 tweet: Önceki testte kullandığımız normal örnekler
    "@United my bag is lost again",
    "@Delta Thanks for cancelling my flight with no warning!",
    "@Southwest flying tomorrow, hope everything goes smoothly.",

    # Yeni 3 tweet: Daha karmaşık ifadeler
    '@United I was told "your flight is cancelled" but the agent later said it wasn\'t. Great communication... 🙄',
    '@Delta Flight DL204 from JFK to LAX was delayed 5 hours on 12/10. Agent Sarah said "weather", but the app says "technical issue".',
    '@Southwest Not bad at all! I arrived only 20 minutes late, and the crew was actually helpful 😊'
]

modes = ["none", "json", "schema"]
results = []

output_dir = Path("results/parser_robustness")
output_dir.mkdir(parents=True, exist_ok=True)


for model in MODELS:
    for tweet_id, tweet in enumerate(tweets, start=1):
        prompt = build_prompt(tweet, variant="few_shot_v2")

        for mode in modes:
            try:
                # Modelin ham çıktısını ve yanıt süresini al.
                raw, latency = query_ollama(
                    prompt,
                    mode=mode,
                    temperature=0.0,
                    model=model
                )

                # Ham çıktı doğrudan JSON olarak okunabiliyor mu?
                data, stage = parse_llm_json(raw)
                parse_ok = data is not None

                # Okunan JSON beklenen şemaya uyuyor mu?
                if parse_ok:
                    schema_ok, error = validate_output(data)
                else:
                    schema_ok = False
                    error = "JSON okunamadı"

                request_error = ""

            except Exception as exc:
                raw = ""
                latency = None
                stage = "failed"
                parse_ok = False
                schema_ok = False
                error = ""
                request_error = f"{type(exc).__name__}: {exc}"

            results.append({
                "model": model,
                "tweet_id": tweet_id,
                "mode": mode,
                "parse_stage": stage,
                "parse_success": parse_ok,
                "schema_valid": schema_ok,
                "latency_seconds": latency,
                "validation_error": error,
                "request_error": request_error,
                "raw_output": raw
            })

            print(
                f"{model} | tweet={tweet_id} | {mode} | "
                f"stage={stage} | parse={parse_ok} | "
                f"schema={schema_ok}"
            )


# Bütün ham sonuçları CSV'ye kaydet.
df = pd.DataFrame(results)
df.to_csv(output_dir / "model_results.csv", index=False)

# Model ve mod bazında özet çıkar.
summary = (
    df.groupby(["model", "mode"])
    .agg(
        sample_count=("tweet_id", "count"),
        parse_success_rate=("parse_success", "mean"),
        schema_valid_rate=("schema_valid", "mean"),
        median_latency=("latency_seconds", "median")
    )
    .reset_index()
)

summary["parse_success_rate"] *= 100
summary["schema_valid_rate"] *= 100

summary.to_csv(output_dir / "model_summary.csv", index=False)

print("\n=== ÖZET SONUÇLAR ===")
print(summary.to_string(index=False))
print(f"\nDosyalar kaydedildi: {output_dir}")
