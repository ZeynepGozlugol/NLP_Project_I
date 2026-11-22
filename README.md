# Structured Text Classification and Information Extraction with Local LLMs

**NLP Project I — Çukurova University**  
Python · Ollama · Qwen 2.5 7B · Structured JSON output

This project classifies airline customer tweets as **negative**, **neutral**, or **positive** and extracts named entities and service-related issues into a machine-readable JSON object. It investigates whether **schema-constrained decoding** improves JSON reliability and downstream task performance compared with requesting JSON through a prompt alone.

The project uses **local, open-source models through Ollama**; no cloud LLM API or model fine-tuning is used.

## What the system returns

For a tweet such as `@united Flight UA123 to London was delayed.`, the intended output structure is:

```json
{
  "category": "negative",
  "sentiment_score": 0.1,
  "entities": [
    {"text": "@united", "type": "airline"},
    {"text": "UA123", "type": "flight_number"},
    {"text": "London", "type": "location"}
  ],
  "issues": ["Late Flight"]
}
```

The six entity types are `airline`, `person`, `location`, `flight_number`, `organization`, and `other`. The JSON schema also constrains valid sentiment categories, score range, required fields, and issue labels.

## Dataset and model selection

The [Twitter Airline Sentiment dataset](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment) contains 14,640 records. After HTML unescaping and duplicate-text removal, the original project obtained 14,427 unique records. A 150-tweet development subset supported preliminary prompt selection; a separate 300-tweet test subset was created. The final JSON experiment uses **120 balanced tweets (40 per class)** drawn from that test subset, with the **same samples in both modes**.

Preliminary experiments compared Llama 3.2 3B and Qwen 2.5 7B with different prompts:

| Preliminary setting | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Llama 3.2 3B — zero-shot | 77.26% | 76.82% |
| Llama 3.2 3B — few-shot v1 | 74.67% | 72.60% |
| Qwen 2.5 7B — zero-shot | 80.00% | 79.79% |
| Qwen 2.5 7B — few-shot v2 | **80.67%** | **80.43%** |

These are **earlier selection experiments**, not the final controlled two-mode comparison. As the prompts differ, they should not be interpreted as an isolated comparison of model sizes. Qwen with `few_shot_v2` was selected for the final comparison.

## Final experiment: two decoding settings

The experiment holds the model, data, prompt, and sampling parameters fixed and varies only Ollama's `format` argument:

- **Prompt-only (`mode="none"`)**: The prompt requests JSON, but the Ollama `format` parameter is **not provided**.
- **Schema-constrained (`mode="schema"`)**: The Ollama `format` parameter contains the JSON schema defined in `src/prompts.py`.

| Setting | Value |
| --- | --- |
| Model tag | `qwen2.5:7b` |
| Model size / quantization | 7.6B parameters / `Q4_K_M` (model file ~4.7 GB) |
| Ollama version during evaluation | `0.40.16` |
| Temperature / seed | `0.0` / `42` |
| Context allocation (`num_ctx`) | `4096` |
| Prompt variant | `few_shot_v2`, three examples |
| Test samples | 120, balanced; selection seed `2026` |
| Calls | 120 samples × 2 modes = **240** |
| Other sampling options | Ollama/model defaults |

The model reports a maximum supported context length of 32,768 tokens; **4,096** was configured for these runs.

### Final results

| Metric | Prompt-only | Schema-constrained |
| --- | ---: | ---: |
| Raw JSON parsing success | 100.00% | 100.00% |
| JSON parsing success after cleaning | 100.00% | 100.00% |
| Schema validity | 99.17% | **100.00%** |
| Sentiment accuracy | 75.83% | 76.67% |
| Sentiment macro F1 | 76.34% | 76.84% |
| Entity precision | 88.41% | 87.95% |
| Entity recall | 65.91% | 66.36% |
| Entity micro F1 | 75.52% | 75.65% |
| Median response time | 1.716 s | 1.700 s |

The entity metrics above use the **latest reviewed annotation set of 220 gold entity mentions**. Earlier annotation drafts produce different entity F1 values; do not mix them with the current results. The main insight is that schema constraints improved **structural reliability**, while semantic classification and extraction changed only slightly.

**Example limitation:** A schema-valid result can still omit locations or assign the wrong entity type. Exact `entity text + entity type` matching is case-sensitive (with whitespace normalization), and duplicate mentions are counted. This is *not* a character-offset span-level evaluation.

## Parsing, validation, and gold entities

1. Attempt `json.loads()` directly on the raw model string.
2. If needed, remove Markdown fences / surrounding prose and repair some trailing-comma errors.
3. Validate the required keys, field types, allowed sentiment/entity/issue values, and score bounds.
4. Count invalid outputs as failed predictions rather than silently removing them from the evaluation.

The controlled parser robustness suite was rerun against the updated parser and **all 10 tests passed**. In the final 120-tweet comparison, all raw strings parsed without cleaning, although one prompt-only response violated the schema.

Since the dataset does not include complete entity gold labels, an initial annotation draft was generated with AI assistance. Python checks validated structure and whether entity strings occur in their tweet. Most annotations were reviewed; ambiguous social-media accounts (for example `@triflight`) were researched on Twitter, and misclassified, missing, or out-of-scope mentions were corrected. The resulting set contains **220 entity mentions** across 120 tweets. There was no independent second annotator or inter-annotator agreement measurement.

## Temperature and live demonstration

An earlier supplementary test compared temperatures **0.0, 0.3, and 0.7** on 45 tweets; accuracy was **80.00%** and macro F1 was **79.72%** in the tested conditions.

The updated live demo uses **five newly written tweets**, each evaluated with both decoding modes at temperatures **0.0 and 1.0** (**20 calls**). In the recorded development run, all sentiment predictions were correct and all outputs were valid JSON, but some entities were omitted: `Paris` and `Rome` were missed at both temperatures, while `Madrid` appeared at temperature 1.0 but not 0.0. These five cases illustrate behavior; they do not establish population-level effects.

## Repository structure

```text
nlp_proje1/
├── src/
│   ├── config.py                    # Project paths, model and label constants
│   ├── data.py                      # Loading, cleaning and data splits
│   ├── eda.py                       # Dataset exploration
│   ├── prompts.py                   # Prompt variants and JSON schema
│   ├── llm_client.py                # Local Ollama requests
│   ├── json_parser.py               # Cleaning, parsing and validation
│   ├── run_experiments.py           # Earlier model / prompt experiments
│   ├── run_json_modes_experiment.py # Final 120-sample two-mode experiment
│   ├── final_metrics.py             # Final classification metrics
│   ├── prepare_gold_entities.py     # Empty annotation template
│   ├── validate_gold_entities.py    # Structural annotation checks
│   ├── finalize_gold_entities.py    # Initial draft corrections
│   ├── entity_evaluation.py         # Entity-level P/R/F1
│   ├── run_temperature_experiment.py
│   ├── live_demo.py                 # Five new tweets, two modes, two temperatures
│   └── test_parser.py               # Parser robustness tests
├── data/                             # Local data files (may be excluded from Git)
├── results/
│   ├── final_json_comparison/final/ # Evaluation outputs and reviewed annotations
│   └── live_demo/                   # Live demonstration outputs
├── requirements.txt
├── README.md
└── NLP_Project_I_Technical_Report.pdf
```

*Some files shown here were added after the original project archive; the README describes the latest developed project, not the initial ZIP snapshot. The report and final evaluation results should be included in the submitted repository.*

## Setup and usage

1. Install [Ollama](https://ollama.com/) and download the model:

   ```bash
   ollama pull qwen2.5:7b
   ollama list
   ollama show qwen2.5:7b
   ```

2. Create a Python virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   python -m pip install -r requirements.txt
   ```

3. Ensure your data files are available in the paths expected by `src/config.py` and `src/data.py`. The raw dataset can be obtained from the source linked above.

4. Run checks and experiments from the project root:

   ```bash
   python src/test_parser.py
   python src/run_json_modes_experiment.py
   python src/entity_evaluation.py
   python src/live_demo.py
   ```

   **Caution:** `run_json_modes_experiment.py` makes 240 local inference calls when `N_PER_CLASS = 40`, and re-running can replace outputs in the final results folder. Keep a backup of the submitted results and do not change code/results after the assignment deadline.

The final evaluation uses `results/final_json_comparison/final/gold_entities.csv`, `qwen_few_shot_v2_none.csv`, and `qwen_few_shot_v2_schema.csv`. For reproducibility, include the exact final files (not just earlier annotation drafts) and `experiment_settings.json` in the repository.

## Limitations and acknowledgements

- Performance was measured on one balanced 120-tweet subset; one run per setting does not capture run-to-run variation.
- Gold annotations were created with AI assistance and targeted human review, not independently double-annotated.
- Exact matching is strict; abbreviated tweet text, hashtags and noisy user handles complicate entity extraction.
- Seeds improve reproducibility but may not guarantee identical model outputs across hardware and Ollama versions.
- AI assistants were used to help draft and revise project code, documentation and the initial gold annotation draft. The reported experiment outputs were obtained by running local models and Python evaluation scripts.

See **`NLP_Project_I_Technical_Report.pdf`** for the methodology, detailed analysis and references.
