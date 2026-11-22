from json_parser import parse_llm_json, validate_output
from prompts import build_prompt
from llm_client import query_ollama

print("=== 1) PARSER TESTİ (Ollama gerekmez) ===")
tests = {
    "temiz JSON":          '{"category": "positive", "sentiment_score": 0.9, "entities": [], "issues": []}',
    "markdown bloklu":     '```json\n{"category": "negative", "sentiment_score": 0.1, "entities": [], "issues": []}\n```',
    "sohbet metinli":      'Here is your JSON:\n{"category": "neutral", "sentiment_score": 0.5, "entities": [], "issues": []}\nHope this helps!',
    "sondaki virgül":      '{"category": "neutral", "sentiment_score": 0.5, "entities": [], "issues": [],}',
    "tamamen bozuk":       'Sorry, I cannot do that.',
}
for name, raw in tests.items():
    data, stage = parse_llm_json(raw)
    ok = validate_output(data)[0] if data else False
    print(f"{name:18s} -> aşama={stage:9s} şema_geçerli={ok}")

print("\n=== 2) OLLAMA TESTİ ===")
tweet = "@United my bag is lost again"
try:
    for mode in ["none", "json", "schema"]:
        raw, lat = query_ollama(build_prompt(tweet), mode)
        data, stage = parse_llm_json(raw)
        ok = validate_output(data)[0] if data else False
        print(f"mod={mode:6s} aşama={stage:7s} geçerli={ok} {lat:.1f}s | {raw[:90]!r}")
except Exception as e:
    print("Ollama'ya ulaşılamadı:", type(e).__name__, "-", e)
    print("Ollama uygulamasının açık olduğundan emin ol, sonra tekrar dene.")