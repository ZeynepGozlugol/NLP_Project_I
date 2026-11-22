
import json

from config import CATEGORIES
from json_parser import parse_llm_json, validate_output


# Beklenen doğru JSON örneğimiz.
valid_data = {
    "category": CATEGORIES[0],
    "sentiment_score": 0.9,
    "entities": [],
    "issues": []
}

valid_json = json.dumps(valid_data)


# Her test için beklenen parse ve şema sonucu.
tests = [
    ("Temiz JSON", valid_json, True, True),

    ("Markdown kod bloğu",
     f"```json\n{valid_json}\n```", True, True),

    ("Açıklama metni",
     f"Sonuç:\n{valid_json}\nBitti.", True, True),

    ("Sondaki fazla virgül",
     valid_json[:-1] + ",}", True, True),

    ("Eksik alan",
     json.dumps({
         "sentiment_score": 0.9,
         "entities": [],
         "issues": []
     }), True, False),

    ("Yanlış skor tipi",
     json.dumps({
         **valid_data,
         "sentiment_score": "yüksek"
     }), True, False),

    ("Geçersiz kategori",
     json.dumps({
         **valid_data,
         "category": "__invalid__"
     }), True, False),

    ("Yanlış entity yapısı",
     json.dumps({
         **valid_data,
         "entities": [{"text": 123, "type": "airline"}]
     }), True, False),

    ("Bozuk JSON",
     '{"category": "positive", "sentiment_score": }',
     False, False),

    ("JSON olmayan metin",
     "Bu bir JSON çıktısı değil.",
     False, False),
]


passed = 0

for name, raw, expected_parse, expected_schema in tests:
    data, stage = parse_llm_json(raw)

    parse_ok = data is not None

    if parse_ok:
        schema_ok, error = validate_output(data)
    else:
        schema_ok, error = False, "JSON okunamadı"

    test_ok = (
        parse_ok == expected_parse
        and schema_ok == expected_schema
    )

    passed += test_ok

    print(
        f"{name:25} | "
        f"aşama={stage:8} | "
        f"parse={parse_ok} | "
        f"şema={schema_ok} | "
        f"test={'BAŞARILI' if test_ok else 'BAŞARISIZ'}"
    )

print(f"\nTest başarı oranı: {passed}/{len(tests)} "
      f"(%{passed / len(tests) * 100:.1f})")
