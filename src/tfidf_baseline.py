"""Classical baseline for defense question G: TF-IDF + logistic regression.
Evaluated on exactly the same 120 tweets used in the LLM comparison."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from config import DATA_DIR, RESULTS_DIR

FINAL = RESULTS_DIR / "final_json_comparison" / "final" / "qwen_few_shot_v2_schema.csv"

eval_df = pd.read_csv(FINAL)[["text", "true_label"]]           # the same 120 tweets
test_texts = set(pd.read_csv(DATA_DIR / "test.csv")["text"])   # 300-tweet test subset
full = pd.read_csv(DATA_DIR / "tweets_clean.csv")
train_full = full[~full["text"].isin(test_texts)]              # never train on test tweets
dev = pd.read_csv(DATA_DIR / "dev.csv")


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def score(train_df):
    model = make_model().fit(train_df["text"], train_df["true_label"])
    pred = model.predict(eval_df["text"])
    return (accuracy_score(eval_df["true_label"], pred) * 100,
            f1_score(eval_df["true_label"], pred, average="macro") * 100)


rows = []
acc, f1 = score(train_full)
rows.append((f"TF-IDF + LR, {len(train_full)} training tweets", acc, f1))
acc, f1 = score(dev)
rows.append((f"TF-IDF + LR, {len(dev)} training tweets (dev set)", acc, f1))

# Only 10 labeled examples: average over 20 random draws (about 3 per class)
accs, f1s = [], []
for seed in range(20):
    small = pd.concat([train_full[train_full["true_label"] == c].sample(n, random_state=seed)
                       for c, n in [("negative", 3), ("neutral", 3), ("positive", 4)]])
    a, f = score(small)
    accs.append(a); f1s.append(f)
rows.append(("TF-IDF + LR, 10 training tweets (mean of 20 draws)", np.mean(accs), np.mean(f1s)))

rows.append(("Qwen 2.5 7B few-shot, prompt-only (reference)", 75.83, 76.34))
rows.append(("Qwen 2.5 7B few-shot, schema (reference)", 76.67, 76.84))

print(pd.DataFrame(rows, columns=["method", "accuracy_%", "macro_f1_%"]).round(2).to_string(index=False))