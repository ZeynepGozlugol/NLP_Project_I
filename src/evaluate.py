import json
import math
import re
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
 
import config
from config import LABELS
 
 
# ------------------------------------------------------------------ ölçümler
def json_report(res):
    """Ödevin zorunlu metriği: JSON parse başarısı."""
    return {
        "n": len(res),
        "raw_parse_%": round(res["first_raw_parse"].mean() * 100, 2),            # ödevin 'ilk deneme' tanımı
        "parse_after_cleaning_%": round(res["first_parse_after_cleaning"].mean() * 100, 2),
        "schema_valid_first_try_%": round(res["first_schema_ok"].mean() * 100, 2),
        "final_after_retry_%": round(res["final_ok"].mean() * 100, 2),
        "retries_used": int((res["attempts"] > 1).sum()),
        "median_latency_s": round(res["latency"].median(), 2),                   # medyan: uç değerlere dayanıklı
    }
 
 
def sentiment_metrics(res):
    n = len(res)
    valid = res.dropna(subset=["pred_label"])
    acc = float((res["pred_label"] == res["true_label"]).mean())   # parse hatası = yanlış sayılır
    ci = float(1.96 * np.sqrt(acc * (1 - acc) / n))                     # %95 güven aralığı (normal yaklaşım)
    rep = classification_report(valid["true_label"], valid["pred_label"], labels=LABELS,
                                output_dict=True, zero_division=0)
    return {
        "accuracy_%": round(acc * 100, 2), "ci95_±": round(ci * 100, 1),
        "macro_f1_%": round(f1_score(valid["true_label"], valid["pred_label"], labels=LABELS,
                                     average="macro", zero_division=0) * 100, 2),
        "recall_neg": round(rep["negative"]["recall"], 2),
        "recall_neu": round(rep["neutral"]["recall"], 2),
        "recall_pos": round(rep["positive"]["recall"], 2),
    }
 
 
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())
 
 
def extraction_eval(res):
    """entities ve issues kalitesi. Dönüş: (özet sözlüğü, issue-türü bazlı tablo)"""
    r = res[res["final_ok"]].copy()
    r["ents"] = r["entities"].apply(json.loads)
    r["iss"] = r["issues"].apply(json.loads)
 
    # 1) Havayolu varlığı (referans: 'airline' sütunu; yalnızca adı tweette geçenler)
    m = r[r.apply(lambda x: _norm(x["airline"]) in _norm(x["text"]), axis=1)]
    found = m.apply(lambda x: any(_norm(x["airline"]) in _norm(e["text"]) for e in x["ents"]), axis=1)
    typed = m.apply(lambda x: any(_norm(x["airline"]) in _norm(e["text"]) and e["type"] == "airline"
                                  for e in x["ents"]), axis=1)
 
    # 2) Halüsinasyon: entity metni tweette birebir geçiyor mu?
    ents = [(x["text"], e["text"]) for _, x in r.iterrows() for e in x["ents"]]
    grounded = np.mean([e.lower() in t.lower() for t, e in ents]) if ents else float("nan")
 
    # 3) Issues (referans: 'negativereason'; yalnızca negatif ve etiketli tweetler)
    neg = r[(r["true_label"] == "negative") & r["negativereason"].notna()
            & (r["negativereason"] != "Can't Tell")].copy()
    neg["hit"] = neg.apply(lambda x: x["negativereason"] in x["iss"], axis=1)
    false_alarm = r[r["true_label"] != "negative"]["iss"].apply(len).gt(0).mean()
 
    summary = {"airline_found_%": round(found.mean() * 100, 1),
               "airline_typed_%": round(typed.mean() * 100, 1),
               "entities_per_tweet": round(len(ents) / max(len(r), 1), 2),
               "entity_grounded_%": round(grounded * 100, 1),
               "issue_gold_hit_%": round(neg["hit"].mean() * 100, 1),
               "issue_false_alarm_%": round(false_alarm * 100, 1)}
    per_reason = neg.groupby("negativereason")["hit"].agg(hit_rate="mean", n="count").round(2)
    return summary, per_reason
 
 
def accuracy_by_label_confidence(res, thr=0.7):
    """Etiketçiler anlaşamayan (güveni düşük) tweetlerde model daha mı çok yanılıyor?"""
    if "label_conf" not in res.columns:
        return {}
    ok = res["pred_label"] == res["true_label"]
    hi, lo = res["label_conf"] >= thr, res["label_conf"] < thr
    return {f"acc_conf>={thr}_%": round(ok[hi].mean() * 100, 1), "n_high": int(hi.sum()),
            f"acc_conf<{thr}_%": round(ok[lo].mean() * 100, 1), "n_low": int(lo.sum())}
 
 
def mcnemar_exact(correct_a, correct_b):
    """Aynı tweetler üzerinde iki sınıflandırıcının farkı gerçek mi? (kesin McNemar)
    Dönüş: (A doğru-B yanlış sayısı, A yanlış-B doğru sayısı, p-değeri)"""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n_a, n_b = int((a & ~b).sum()), int((~a & b).sum())
    n = n_a + n_b
    if n == 0:
        return n_a, n_b, 1.0
    k = min(n_a, n_b)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n_a, n_b, min(1.0, p)
 
 
# ------------------------------------------------------------------ rapor
def _load(model, split, variant, mode):
    p = config.run_path(model, split, variant, mode)
    return pd.read_csv(p) if p.exists() else None
 
 
def _selected(model):
    p = config.RUNS_DIR / config.model_slug(model) / "selected_variant.json"
    return json.loads(p.read_text())["selected"] if p.exists() else None
 
 
def _show(df, name, title):
    if df.empty:
        print(f"\n[{title}] sonuç yok"); return
    df.to_csv(config.TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n===== {title} =====\n{df.to_string(index=False)}")
 
 
def main(mode=config.DEFAULT_MODE):
    dev_rows, test_rows, json_rows, ext_rows, score_rows, conf_rows = [], [], [], [], [], []
    test_selected = {}
 
    for model in config.MODELS:
        chosen = _selected(model)
        for variant in config.VARIANTS:
            for split, bucket in [("dev", dev_rows), ("test", test_rows)]:
                res = _load(model, split, variant, mode)
                if res is None:
                    continue
                bucket.append({"model": model, "variant": variant, "chosen": variant == chosen,
                               **sentiment_metrics(res)})
                json_rows.append({"model": model, "split": split, "variant": variant, **json_report(res)})
 
        res = _load(model, "test", chosen, mode) if chosen else None
        if res is None:
            continue
        test_selected[model] = res
        summary, per_reason = extraction_eval(res)
        ext_rows.append({"model": model, "variant": chosen, **summary})
        per_reason.to_csv(config.TABLES_DIR / f"issues_per_reason_{config.model_slug(model)}.csv")
 
        # sentiment_score gerçekten polarite mi? (neg~0, neu~0.5, pos~1 beklenir)
        sc = res.groupby("true_label")["sentiment_score"].mean().round(3)
        score_rows.append({"model": model, **{f"mean_score_{k}": v for k, v in sc.items()}})
        conf_rows.append({"model": model, "variant": chosen, **accuracy_by_label_confidence(res)})
 
        wrong = res[res["pred_label"] != res["true_label"]]
        cols = [c for c in ["text", "true_label", "pred_label", "sentiment_score", "label_conf"] if c in wrong]
        wrong[cols].to_csv(config.TABLES_DIR / f"errors_{config.model_slug(model)}.csv",
                           index=False, encoding="utf-8-sig")
 
    _show(pd.DataFrame(dev_rows), "dev_comparison", "DEV: model x prompt")
    _show(pd.DataFrame(test_rows), "test_comparison", "TEST: model x prompt (chosen=True: dev'de seçilen)")
    _show(pd.DataFrame(json_rows), "json_robustness", "JSON dayanıklılığı (ödev zorunlu metriği)")
    _show(pd.DataFrame(ext_rows), "extraction_comparison", "Extraction (test, seçilen prompt)")
    _show(pd.DataFrame(score_rows), "sentiment_score_check", "sentiment_score sınıf ortalamaları")
    _show(pd.DataFrame(conf_rows), "accuracy_by_label_confidence", "Etiket güvenine göre accuracy (EDA bölüm 6 bağlantısı)")
 
    for model in config.MODELS:                    # stres testi tabloları (varsa)
        p = config.RUNS_DIR / config.model_slug(model) / "stress_table.csv"
        if p.exists():
            print(f"\n===== STRES TESTİ: {model} =====\n{pd.read_csv(p).to_string(index=False)}")
 
    models = [m for m in config.MODELS if m in test_selected]
    if len(models) >= 2:                           # modeller arası anlamlılık
        a, b = models[0], models[1]
        ra, rb = test_selected[a], test_selected[b]
        assert (ra["text"] == rb["text"]).all(), "iki model farklı tweetler üzerinde çalışmış"
        na, nb, p = mcnemar_exact(ra["pred_label"] == ra["true_label"], rb["pred_label"] == rb["true_label"])
        verdict = "ANLAMLI" if p < 0.05 else "anlamlı DEĞİL (şans eseri olabilir)"
        print(f"\n===== McNEMAR ({a} vs {b}, test) =====\n"
              f"{a} doğru/{b} yanlış: {na} | {a} yanlış/{b} doğru: {nb} | p = {p:.4f} -> fark {verdict}")
 
    test_df = pd.DataFrame(test_rows)
    if not test_df.empty:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a_, metric in zip(ax, ["accuracy_%", "macro_f1_%"]):
            test_df.pivot(index="variant", columns="model", values=metric).plot.bar(ax=a_, rot=15)
            a_.set_title(f"TEST {metric}"); a_.set_xlabel(""); a_.set_ylim(0, 100)
        plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "test_model_comparison.png", dpi=150); plt.close()
 
    if test_selected:
        fig, ax = plt.subplots(1, len(test_selected), figsize=(5 * len(test_selected), 4), squeeze=False)
        for a_, (model, res) in zip(ax[0], test_selected.items()):
            v = res.dropna(subset=["pred_label"])
            cm = confusion_matrix(v["true_label"], v["pred_label"], labels=LABELS)
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=a_, colorbar=False)
            a_.set_title(f"{model}\n({_selected(model)})")
        plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "test_confusion_matrices.png", dpi=150); plt.close()
    print(f"\nTablolar: {config.TABLES_DIR}\nŞekiller: {config.FIGURES_DIR}")
 
 
if __name__ == "__main__":
    main()
