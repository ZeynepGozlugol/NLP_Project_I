
import json

from config import CATEGORIES, ENTITY_TYPES, ISSUE_LABELS


# ============================================================
# JSON SCHEMA FOR STRUCTURED OUTPUT
# ============================================================

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": CATEGORIES
        },
        "sentiment_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string",
                        "enum": ENTITY_TYPES
                    }
                },
                "required": ["text", "type"],
                "additionalProperties": False
            }
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ISSUE_LABELS
            }
        }
    },
    "required": [
        "category",
        "sentiment_score",
        "entities",
        "issues"
    ],
    "additionalProperties": False
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an information extraction system for airline customer tweets.

Return ONLY one valid JSON object.
No markdown, no explanations, and no additional text.

Output format:
{{"category": "positive|negative|neutral",
"sentiment_score": 0.0,
"entities": [{{"text": "...", "type": "..."}}],
"issues": ["..."]}}

Field definitions:

- category:
  Overall sentiment expressed by the customer.

  positive = clear satisfaction, praise or gratitude.
  negative = clear complaint, frustration or a reported service problem.
  neutral = factual information, a plain question, or no clear opinion.

- sentiment_score:
  Sentiment polarity between 0.0 and 1.0.
  0.0 = very negative.
  0.5 = neutral.
  1.0 = very positive.

- entities:
  Named entities explicitly appearing in the tweet.
  Copy entity text exactly from the tweet.
  Airline names and airline handles have type airline.
  Entity type must be one of {ENTITY_TYPES}.

- issues:
  Reported service problems.
  Each issue must be one of {ISSUE_LABELS}.
  Use [] when no problem is reported.

Rules:
- Judge the whole tweet, not isolated keywords.
- Mentioning an airline or flight does not automatically mean negative sentiment.
- A question is neutral unless it clearly expresses a complaint.
- Do not invent entities or issues.
- If there are no entities or issues, use empty lists.
- Include exactly the four required JSON fields.
"""


# ============================================================
# FEW-SHOT V1
# ============================================================

FEW_SHOT_V1 = [
    (
        "@SkyJet thank you for the smooth flight and friendly crew!",
        {
            "category": "positive",
            "sentiment_score": 0.95,
            "entities": [
                {"text": "@SkyJet", "type": "airline"}
            ],
            "issues": []
        }
    ),
    (
        "@SkyJet my flight to Boston was delayed 3 hours and nobody told us why",
        {
            "category": "negative",
            "sentiment_score": 0.05,
            "entities": [
                {"text": "@SkyJet", "type": "airline"},
                {"text": "Boston", "type": "location"}
            ],
            "issues": [
                "Late Flight",
                "Customer Service Issue"
            ]
        }
    ),
    (
        "@SkyJet what time does flight 456 leave tomorrow?",
        {
            "category": "neutral",
            "sentiment_score": 0.5,
            "entities": [
                {"text": "@SkyJet", "type": "airline"},
                {"text": "flight 456", "type": "flight_number"}
            ],
            "issues": []
        }
    )
]


# ============================================================
# FEW-SHOT V2
# ============================================================

FEW_SHOT_V2 = [
    (
        "@SkyJet I love the new seats, but the flight was delayed for two hours",
        {
            "category": "negative",
            "sentiment_score": 0.25,
            "entities": [
                {"text": "@SkyJet", "type": "airline"}
            ],
            "issues": ["Late Flight"]
        }
    ),
    (
        "@SkyJet could you tell me whether flight 456 is on time?",
        {
            "category": "neutral",
            "sentiment_score": 0.5,
            "entities": [
                {"text": "@SkyJet", "type": "airline"},
                {"text": "flight 456", "type": "flight_number"}
            ],
            "issues": []
        }
    ),
    (
        "@SkyJet thank you for helping me find my bag!",
        {
            "category": "positive",
            "sentiment_score": 0.9,
            "entities": [
                {"text": "@SkyJet", "type": "airline"}
            ],
            "issues": []
        }
    )
]


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(tweet, variant="zero_shot"):
    """
    Builds a prompt for sentiment classification
    and information extraction.

    Variants:
        zero_shot
        few_shot_v1
        few_shot_v2
    """

    if isinstance(variant, bool):
        variant = (
            "few_shot_v1"
            if variant
            else "zero_shot"
        )

    examples_by_variant = {
        "zero_shot": [],
        "few_shot_v1": FEW_SHOT_V1,
        "few_shot_v2": FEW_SHOT_V2
    }

    if variant not in examples_by_variant:
        raise ValueError(
            f"Unknown prompt variant: {variant}"
        )

    prompt = SYSTEM_PROMPT

    examples = examples_by_variant[variant]

    if examples:
        prompt += "\nExamples:\n"

        for example_tweet, expected_output in examples:
            prompt += (
                f"\nTweet: "
                f"{json.dumps(example_tweet, ensure_ascii=False)}\n"
                f"JSON: "
                f"{json.dumps(expected_output, ensure_ascii=False)}\n"
            )

    prompt += (
        f"\nTweet: {json.dumps(tweet, ensure_ascii=False)}\n"
        "JSON:"
    )

    return prompt


# ============================================================
# CHECK JSON SCHEMA
# ============================================================

if __name__ == "__main__":
    print("=== JSON SCHEMA ===")
    print(json.dumps(JSON_SCHEMA, indent=2, ensure_ascii=False))

    print("\n=== SAMPLE PROMPT ===")
    print(
        build_prompt(
            "@united My flight to Chicago was delayed.",
            "few_shot_v2"
        )
    )
