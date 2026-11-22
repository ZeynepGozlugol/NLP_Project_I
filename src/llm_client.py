
import time
import requests

from config import OLLAMA_URL, MODEL_NAME, SEED
from prompts import JSON_SCHEMA


# ============================================================
# MODEL GENERATION SETTINGS
# ============================================================

CONTEXT_LENGTH = 4096
REQUEST_TIMEOUT = 300


# ============================================================
# OLLAMA MODEL QUERY
# ============================================================

def query_ollama(
    prompt,
    mode="schema",
    temperature=0.0,
    seed=SEED,
    model=MODEL_NAME
):
    """
    Sends a prompt to a local LLM using Ollama.

    Modes:
        none   : No format parameter (prompt-only)
        json   : JSON format enforcement
        schema : JSON schema-constrained decoding

    Returns:
        raw_response (str)
        latency_seconds (float)
    """

    if mode not in ("none", "json", "schema"):
        raise ValueError(f"Unsupported mode: {mode}")

    # Common settings for all experiments
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": CONTEXT_LENGTH
        },
        "keep_alive": "30m"
    }

    # Prompt-only: no format parameter
    if mode == "none":
        pass

    # JSON mode: enforce valid JSON
    elif mode == "json":
        payload["format"] = "json"

    # Schema-constrained: enforce JSON structure
    elif mode == "schema":
        payload["format"] = JSON_SCHEMA

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    response_data = response.json()

    latency = time.perf_counter() - start_time

    raw_response = response_data["response"]

    return raw_response, latency


# ============================================================
# CHECK INSTALLED MODEL
# ============================================================

def check_model(model):
    """
    Checks whether the requested model is installed in Ollama.
    """

    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=10
    )

    response.raise_for_status()

    installed_models = [
        item["name"]
        for item in response.json().get("models", [])
    ]

    if model not in installed_models:
        raise RuntimeError(
            f"Model '{model}' is not installed. "
            f"Available models: {installed_models}"
        )

    print(f"[OK] Model hazır: {model}")
    print(f"[INFO] Context length: {CONTEXT_LENGTH}")
