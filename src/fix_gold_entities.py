import json, shutil
import pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "results/final_json_comparison/final"
P = D / "gold_entities.csv"
BACKUP = D / "gold_entities_before_fix.csv"

# Sadece bir kez çalışır: yedek orijinal dosyayı korur
if BACKUP.exists():
    raise FileExistsError("Backup already exists, fix was probably applied before.")
shutil.copy(P, BACKUP)

ADD = {
    3:  [{"text": "@triflight",   "type": "location"}],      # airport account
    88: [{"text": "@Leopolds_IC", "type": "organization"}],  # ice cream brand
}
REMOVE = {
    33: ["@SaharaSams"],  # looks like a personal account, keep rule consistent
}

df = pd.read_csv(P, keep_default_na=False)
for i, row in df.iterrows():
    sid = int(row.sample_id)
    tweet = row.text
    ents = json.loads(row.gold_entities)
    ents = [e for e in ents if e["text"] not in REMOVE.get(sid, [])]
    for e in ADD.get(sid, []):
        if e["text"] not in tweet:
            raise ValueError(f"Entity not in tweet {sid}: {e['text']}")
        ents.append(e)
    df.at[i, "gold_entities"] = json.dumps(ents, ensure_ascii=False)

df.to_csv(P, index=False, encoding="utf-8-sig")
print("Gold updated. Backup:", BACKUP.name)