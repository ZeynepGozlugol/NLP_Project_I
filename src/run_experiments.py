import argparse
import json
 
import pandas as pd
from tqdm.auto import tqdm
 
import config
import llm_client
from config import run_path
from data import load_full, make_splits
from evaluate import json_report, sentiment_metrics
from json_parser import parse_llm_json, validate_output
from prompts import build_prompt
 
 
def process_tweet(tweet, model, variant, mode, max_retries=1):
    """prompt -> LLM -> parse -> doğrula -> (gerekirse retry). Hiçbir hata programı çökertmez."""
    prompt = build_prompt(tweet, variant)
    rec = {"first_raw_parse": False, "first_parse_after_cleaning": False, "first_schema_ok": False,
           "attempts": 0, "final_ok": False, "error": "", "raw_output": "", "latency": 0.0, "parsed": None}
 
    for attempt in range(max_retries + 1):
        rec["attempts"] = attempt + 1
        try:
            # ilk deneme deterministik (temp=0); retry'da farklı çıktı için sıcaklık artırılır
            raw, lat = llm_client.query_ollama(prompt, mode,
                                               temperature=0.0 if attempt == 0 else 0.5,
                                               seed=config.SEED + attempt, model=model)
            rec["latency"] += lat
            rec["raw_output"] = raw
 
            data, stage = parse_llm_json(raw)
            ok, err = (False, "parse failed") if data is None else validate_output(data)
 
            if attempt == 0:                      # ödevin "ilk denemede başarı" metrikleri
                rec["first_raw_parse"] = (stage == "raw") and isinstance(data, dict)
                rec["first_parse_after_cleaning"] = stage != "failed"
                rec["first_schema_ok"] = ok
            if ok:
                rec.update(final_ok=True, parsed=data, error="")
                break
            rec["error"] = err
        except Exception as e:                    # bağlantı, timeout vb.
            rec["error"] = f"{type(e).__name__}: {e}"
    return rec
 
 
def run_experiment(df, model, variant, mode, split, force=False, save=True):
    path = run_path(model, split, variant, mode)
    if save and path.exists() and not force:
        print(f"[önbellek] {path.name} bulundu, tekrar çalıştırılmadı.")
        return pd.read_csv(path)
 
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model} | {split} | {variant}"):
        rec = process_tweet(row["text"], model, variant, mode)
        p = rec.pop("parsed")
        rec.update({
            "text": row["text"], "true_label": row["true_label"], "airline": row["airline"],
            "negativereason": row["negativereason"],
            "label_conf": row.get("airline_sentiment_confidence"),   # etiketçi güveni (hata analizi için)
            "pred_label": p["category"] if p else None,
            "sentiment_score": p["sentiment_score"] if p else None,
            "entities": json.dumps(p["entities"], ensure_ascii=False) if p else None,
            "issues": json.dumps(p["issues"], ensure_ascii=False) if p else None,
        })
        rows.append(rec)
    res = pd.DataFrame(rows)
    if save:
        res.to_csv(path, index=False, encoding="utf-8-sig")
    return res
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test"], required=True)
    ap.add_argument("--models", nargs="+", default=[config.MODEL_NAME])
    ap.add_argument("--mode", default=config.DEFAULT_MODE, choices=["none", "json", "schema"])
    ap.add_argument("--limit", type=int, default=None, help="deneme modu: ilk N tweet, kayıt yok (yalnızca dev)")
    ap.add_argument("--force", action="store_true", help="önbelleği yoksay")
    args = ap.parse_args()
 
    if args.limit and args.stage == "test":
        raise SystemExit("--limit yalnızca dev aşamasında kullanılabilir (test seti tek seferde ve eksiksiz çalışmalı).")
    for m in args.models:
        llm_client.check_model(m)
 
    dev, test = make_splits(load_full())
    dev.to_csv(config.DATA_DIR / "dev.csv", index=False)       # tekrarlanabilirlik: kullanılan tweetler diskte
    test.to_csv(config.DATA_DIR / "test.csv", index=False)
 
    for model in args.models:
        if args.stage == "dev":
            df = dev.head(args.limit) if args.limit else dev
            rows = []
            for variant in config.VARIANTS:
                res = run_experiment(df, model, variant, args.mode, "dev", args.force, save=not args.limit)
                rows.append({"variant": variant, **sentiment_metrics(res), **json_report(res)})
            print(f"\n=== DEV: {model} (n={len(df)}) ===")
            print(pd.DataFrame(rows)[["variant", "accuracy_%", "ci95_±", "macro_f1_%", "recall_neu",
                                      "final_after_retry_%", "median_latency_s"]].to_string(index=False))
            if args.limit:
                print("(deneme modu: sonuçlar kaydedilmedi)")
        else:
            scores = {}
            for variant in config.VARIANTS:
                p = run_path(model, "dev", variant, args.mode)
                if not p.exists():
                    raise SystemExit(f"Önce dev aşamasını çalıştırın: python src/run_experiments.py "
                                     f"--stage dev --models {model}")
                scores[variant] = sentiment_metrics(pd.read_csv(p))["macro_f1_%"]
            best = max(scores, key=scores.get)
            sel = config.RUNS_DIR / config.model_slug(model) / "selected_variant.json"
            sel.write_text(json.dumps({"model": model, "selected": best, "dev_macro_f1": scores}, indent=2))
            print(f"\n[{model}] dev macro-F1: {scores}\n-> seçilen prompt: {best} (artık değişmez)")
 
            for variant in dict.fromkeys(["zero_shot", best]):     # baseline + seçilen
                res = run_experiment(test, model, variant, args.mode, "test", args.force)
                print(f"TEST {model} | {variant}: {sentiment_metrics(res)} | JSON: {json_report(res)['final_after_retry_%']}%")
 
    print("\nSonraki adım: python src/evaluate.py")
 
 
if __name__ == "__main__":
    main()
 