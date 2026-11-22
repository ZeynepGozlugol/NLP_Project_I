from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
for d in (DATA_DIR, EDA_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment/resolve/main/Tweets.csv"
SEED = 42

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

LABELS = ["negative", "neutral", "positive"]
CATEGORIES = ["positive", "negative", "neutral"]
ENTITY_TYPES = ["airline", "person", "location", "flight_number", "organization", "other"]
ISSUE_LABELS = [
    "Bad Flight", "Cancelled Flight", "Customer Service Issue", "Damaged Luggage",
    "Flight Attendant Complaints", "Flight Booking Problems", "Late Flight",
    "Lost Luggage", "longlines", "Other",
]
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
for d in (DATA_DIR, EDA_DIR, RUNS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
VARIANTS = ["zero_shot", "few_shot_v1", "few_shot_v2"]
DEFAULT_MODE = "schema"

def model_slug(model):
    return model.replace(":", "_").replace("/", "_")

def run_path(model, split, variant, mode):
    d = RUNS_DIR / model_slug(model)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}_{variant}_{mode}.csv"