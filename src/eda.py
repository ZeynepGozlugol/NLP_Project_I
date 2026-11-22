import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EDA_DIR
from data import load_raw, load_full, make_splits

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def save_fig(name):
    plt.tight_layout()
    plt.savefig(EDA_DIR / name, dpi=150)
    plt.close()


raw = load_raw()
full = load_full()
dev, test = make_splits(full)

# ---------------------------------------------------------------- 1. Genel
log("=" * 60 + "\n1. GENEL BAKIŞ\n" + "=" * 60)
log(f"Ham satır sayısı        : {len(raw)}")
log(f"Tekrarlar silindikten sonra: {len(full)}  (silinen: {len(raw) - len(full)})")
log("\nEksik değerler (temiz veri):")
log(full.isna().sum().to_string())
log("\nNot: negativereason yalnızca negatif tweetlerde dolu olmalı, diğerlerinde NaN normal.")

# ---------------------------------------------------------------- 2. Sentiment
log("\n" + "=" * 60 + "\n2. SENTIMENT DAĞILIMI (sınıf dengesizliği)\n" + "=" * 60)
sent = full["true_label"].value_counts()
log(pd.DataFrame({"adet": sent, "yüzde": (sent / len(full) * 100).round(1)}).to_string())
sent.plot(kind="bar", color=["#d9534f", "#999999", "#5cb85c"], rot=0)
plt.title("Sentiment dağılımı (tüm veri)")
plt.ylabel("tweet sayısı")
save_fig("01_sentiment_dagilimi.png")

# ---------------------------------------------------------------- 3. Havayolu
log("\n" + "=" * 60 + "\n3. HAVAYOLU DAĞILIMI ve HAVAYOLUNA GÖRE SENTIMENT\n" + "=" * 60)
log(full["airline"].value_counts().to_string())
ct = pd.crosstab(full["airline"], full["true_label"], normalize="index").round(3) * 100
log("\nHavayoluna göre sentiment yüzdeleri:")
log(ct.to_string())
ct[["negative", "neutral", "positive"]].plot(
    kind="barh", stacked=True, color=["#d9534f", "#999999", "#5cb85c"])
plt.title("Havayoluna göre sentiment (%)")
plt.xlabel("%")
save_fig("02_havayolu_sentiment.png")

# ---------------------------------------------------------------- 4. Issue
log("\n" + "=" * 60 + "\n4. NEGATİF SEBEPLERİ (issues için gold etiket)\n" + "=" * 60)
neg = full[full["true_label"] == "negative"]
reasons = neg["negativereason"].value_counts(dropna=False)
log(reasons.to_string())
log(f"\nNegatif tweet sayısı: {len(neg)} | sebebi 'Can't Tell' veya boş: "
    f"{(neg['negativereason'].isna() | (neg['negativereason'] == chr(67) + 'an' + chr(39) + 't Tell')).sum()}")
neg["negativereason"].value_counts().sort_values().plot(kind="barh", color="#d9534f")
plt.title("Negatif tweetlerde şikayet sebepleri")
save_fig("03_negatif_sebepler.png")

# ---------------------------------------------------------------- 5. Metin
log("\n" + "=" * 60 + "\n5. METİN ÖZELLİKLERİ\n" + "=" * 60)
full["n_chars"] = full["text"].str.len()
full["n_words"] = full["text"].str.split().str.len()
full["n_mentions"] = full["text"].str.count(r"@\w+")
full["n_hashtags"] = full["text"].str.count(r"#\w+")
full["has_url"] = full["text"].str.contains(r"http\S+")
full["has_question"] = full["text"].str.contains(r"\?")

log(full[["n_chars", "n_words", "n_mentions", "n_hashtags"]].describe().round(1).to_string())
log("\nSınıfa göre ortalamalar:")
log(full.groupby("true_label")[["n_chars", "n_words", "n_mentions", "n_hashtags",
                                "has_url", "has_question"]].mean().round(2).to_string())

fig, ax = plt.subplots()
for lab, col in zip(["negative", "neutral", "positive"], ["#d9534f", "#999999", "#5cb85c"]):
    ax.hist(full[full["true_label"] == lab]["n_words"], bins=30, alpha=0.5, label=lab, color=col)
ax.set_title("Kelime sayısı dağılımı (sınıfa göre)")
ax.set_xlabel("kelime sayısı")
ax.legend()
save_fig("04_kelime_sayisi.png")

# ---------------------------------------------------------------- 6. Etiket güveni
log("\n" + "=" * 60 + "\n6. ETİKET GÜVENİLİRLİĞİ (airline_sentiment_confidence)\n" + "=" * 60)
log("Veri seti insan etiketleyicilerle hazırlanmış; confidence < 1 ise etiketçiler anlaşamamış demektir.")
conf = full.groupby("true_label")["airline_sentiment_confidence"].agg(["mean", "min"]).round(3)
conf["confidence<0.7 (%)"] = full.groupby("true_label")["airline_sentiment_confidence"].apply(
    lambda s: round((s < 0.7).mean() * 100, 1))
log(conf.to_string())
log("-> Neutral sınıfta güven düşükse modelin neutral'da zorlanması kısmen etiket belirsizliğinden gelir.")

# ---------------------------------------------------------------- 7. Split
log("\n" + "=" * 60 + "\n7. DEV / TEST AYRIMI\n" + "=" * 60)
for name, d in [("DEV", dev), ("TEST", test)]:
    log(f"\n{name}: n={len(d)}")
    log(d["true_label"].value_counts().to_string())
    log("Havayolu oranları:")
    log(d["airline"].value_counts(normalize=True).round(3).to_string())
log(f"\nDev ∩ Test (sızıntı kontrolü, 0 olmalı): {len(set(dev['text']) & set(test['text']))}")

# ---------------------------------------------------------------- 8. Şema eşlemesi
log("\n" + "=" * 60 + "\n8. JSON ŞEMASI <-> VERİ SETİ EŞLEMESİ\n" + "=" * 60)
log("""category         -> gold: airline_sentiment       | ölçüm: accuracy, macro-F1
sentiment_score  -> gold: YOK                     | ölçüm: sınıf ortalamaları (neg~0, neu~0.5, pos~1)
entities         -> gold: kısmen (airline sütunu) | ölçüm: havayolu bulundu mu + halüsinasyon oranı
issues           -> gold: negativereason          | ölçüm: gold etiket listede mi (sadece negatifler)""")

# ---------------------------------------------------------------- 9. Örnekler
samples = full.groupby("true_label").sample(5, random_state=42)[["true_label", "airline", "text"]]
samples.to_csv(EDA_DIR / "ornek_tweetler.csv", index=False, encoding="utf-8-sig")
log("\nÖrnek tweetler kaydedildi: results/eda/ornek_tweetler.csv")

(EDA_DIR / "eda_ozet.txt").write_text("\n".join(lines), encoding="utf-8")
log("\nTüm çıktılar results/eda/ klasörüne kaydedildi.")