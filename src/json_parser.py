
import json
import math
import re

from config import CATEGORIES, ENTITY_TYPES, ISSUE_LABELS


# ============================================================
# JSON CLEANING AND PARSING
# ============================================================

def parse_llm_json(raw):
    """
    Parse raw LLM output.

    Stages:
        raw: Valid JSON without cleaning
        cleaned: JSON extracted from surrounding text
        repaired: Trailing commas removed
        failed: Parsing unsuccessful
    """

    # First attempt: parse the raw output directly
    try:
        return json.loads(raw), "raw"
    except (json.JSONDecodeError, TypeError):
        pass

    if not isinstance(raw, str):
        return None, "failed"

    # Remove markdown code fences and surrounding text
    try:
        text = re.sub(
            r"```(?:json)?",
            "",
            raw,
            flags=re.IGNORECASE
        ).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            return None, "failed"

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate), "cleaned"
        except json.JSONDecodeError:
            pass

        # Repair trailing commas
        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            candidate
        )

        return json.loads(repaired), "repaired"

    except (json.JSONDecodeError, TypeError, AttributeError):
        return None, "failed"


# ============================================================
# STRICT SCHEMA VALIDATION
# ============================================================

def validate_output(data):
    """
    Validate JSON against the structure defined in prompts.py.
    """

    required = {
        "category",
        "sentiment_score",
        "entities",
        "issues"
    }

    # Root must be a JSON object
    if not isinstance(data, dict):
        return False, "Output must be a JSON object"

    # Exactly four fields are required
    if set(data.keys()) != required:
        return False, "Missing or unexpected fields"

    # Sentiment category
    if (
        not isinstance(data["category"], str)
        or data["category"] not in CATEGORIES
    ):
        return False, "Invalid sentiment category"

    # Sentiment score: 0.0 to 1.0
    score = data["sentiment_score"]

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        return False, "Invalid sentiment score"

    # Entities must be a list
    entities = data["entities"]

    if not isinstance(entities, list):
        return False, "Entities must be a list"

    for entity in entities:

        if not isinstance(entity, dict):
            return False, "Invalid entity object"

        # Entity objects must contain exactly two fields
        if set(entity.keys()) != {"text", "type"}:
            return False, "Invalid entity fields"

        if not isinstance(entity["text"], str):
            return False, "Entity text must be a string"

        if (
            not isinstance(entity["type"], str)
            or entity["type"] not in ENTITY_TYPES
        ):
            return False, "Invalid entity type"

    # Issues must be a list
    issues = data["issues"]

    if not isinstance(issues, list):
        return False, "Issues must be a list"

    for issue in issues:

        if (
            not isinstance(issue, str)
            or issue not in ISSUE_LABELS
        ):
            return False, "Invalid issue label"

    return True, ""


# ============================================================
# PARSER TESTS
# ============================================================

def main():

    print("=== JSON PARSER TESTS ===")

    samples = [
        (
            "Valid JSON",
            '{"category":"positive",'
            '"sentiment_score":0.9,'
            '"entities":[],'
            '"issues":[]}'
        ),
        (
            "Markdown JSON",
            '```json\n'
            '{"category":"neutral",'
            '"sentiment_score":0.5,'
            '"entities":[],'
            '"issues":[]}'
            '\n```'
        ),
        (
            "Extra text",
            'Here is your JSON: '
            '{"category":"negative",'
            '"sentiment_score":0.1,'
            '"entities":[],'
            '"issues":[]}'
        ),
        (
            "Invalid entity type",
            '{"category":"positive",'
            '"sentiment_score":0.9,'
            '"entities":[{"text":"SkyJet",'
            '"type":"banana"}],'
            '"issues":[]}'
        ),
        (
            "Extra entity field",
            '{"category":"positive",'
            '"sentiment_score":0.9,'
            '"entities":[{"text":"SkyJet",'
            '"type":"airline","extra":1}],'
            '"issues":[]}'
        ),
        (
            "Invalid issue",
            '{"category":"negative",'
            '"sentiment_score":0.1,'
            '"entities":[],'
            '"issues":["Unknown Problem"]}'
        ),
        (
            "Invalid JSON",
            'This is not JSON.'
        )
    ]

    for name, raw in samples:

        data, stage = parse_llm_json(raw)

        if data is not None:
            valid, error = validate_output(data)
        else:
            valid, error = False, "Parsing failed"

        print(f"\nTest: {name}")
        print(f"Parse stage: {stage}")
        print(f"Schema valid: {valid}")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
