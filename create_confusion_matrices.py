import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

# Test sonuçlarının bulunduğu CSV dosyaları
files = {
    "Llama 3.2 3B\nZero-shot":
        "results/runs/llama3.2_3b/test_zero_shot_schema.csv",

    "Llama 3.2 3B\nFew-shot v1":
        "results/runs/llama3.2_3b/test_few_shot_v1_schema.csv",

    "Qwen 2.5 7B\nZero-shot":
        "results/runs/qwen2.5_7b/test_zero_shot_schema.csv",

    "Qwen 2.5 7B\nFew-shot v2":
        "results/runs/qwen2.5_7b/test_few_shot_v2_schema.csv"
}

# Sınıfların sırası bütün matrislerde aynı kalmalı
labels = ["negative", "neutral", "positive"]

# Dört grafiği tek görselde, 2x2 düzeninde oluştur
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, file_path) in zip(axes.flat, files.items()):
    df = pd.read_csv(file_path)  # CSV sonuçlarını oku

    # Gerçek ve tahmin edilen etiketleri standartlaştır
    df["true_label"] = df["true_label"].str.strip().str.lower()
    df["pred_label"] = df["pred_label"].str.strip().str.lower()

    # Eksik veya beklenmeyen etiketleri değerlendirmeye katma
    df = df[
        df["true_label"].isin(labels)
        & df["pred_label"].isin(labels)
    ]

    # Gerçek etiketlerle tahminleri karşılaştır
    cm = confusion_matrix(
        df["true_label"],
        df["pred_label"],
        labels=labels
    )

    # Confusion matrix grafiğini çiz
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    display.plot(ax=ax, values_format="d", colorbar=False)

    # Accuracy ve Macro F1 değerlerini hesapla
    accuracy = accuracy_score(df["true_label"], df["pred_label"])
    macro_f1 = f1_score(
        df["true_label"],
        df["pred_label"],
        labels=labels,
        average="macro",
        zero_division=0
    )

    # Model adı ve performans bilgilerini başlığa ekle
    ax.set_title(
        f"{name}\nAccuracy: {accuracy:.2%} | Macro F1: {macro_f1:.2%}"
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Hesaplanan değerleri terminale yazdır
    print(
        f"{name.replace(chr(10), ' ')} | "
        f"Samples: {len(df)} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )

# Grafikleri sıkıştırmadan düzenle ve PNG olarak kaydet
plt.tight_layout()

output_path = Path("results/figures/confusion_matrices.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nGraph saved to: {output_path}")