import html
import pandas as pd
from config import DATA_DIR, DATA_URL, SEED

RAW_PATH = DATA_DIR / "tweets_raw.csv"
CLEAN_PATH = DATA_DIR / "tweets_clean.csv"


def load_raw():
    """Ham veri. İlk seferde indirir, sonra diskten okur."""
    if not RAW_PATH.exists():
        pd.read_csv(DATA_URL).to_csv(RAW_PATH, index=False)
    return pd.read_csv(RAW_PATH)


def load_full():
    """Temizlenmiş veri: html entity temizliği + tekrar eden tweet'lerin silinmesi."""
    raw = load_raw()
    df = raw[["text", "airline_sentiment", "airline", "negativereason",
              "airline_sentiment_confidence"]].rename(columns={"airline_sentiment": "true_label"})
    df["text"] = df["text"].apply(html.unescape)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df.to_csv(CLEAN_PATH, index=False)
    return df


def balanced_sample(df, n_per_class, seed):
    parts = [df[df["true_label"] == c].sample(n_per_class, random_state=seed)
             for c in ["negative", "neutral", "positive"]]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def make_splits(full):
    dev = balanced_sample(full, 50, SEED)                    # prompt geliştirme: 150
    rest = full[~full["text"].isin(dev["text"])]
    test = balanced_sample(rest, 100, SEED)                  # final değerlendirme: 300
    return dev, test