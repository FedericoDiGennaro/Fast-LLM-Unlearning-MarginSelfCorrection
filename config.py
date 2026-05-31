from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "cache" / "models"
MODEL_CONFIGS = {
    "gpt2": str(MODELS_DIR / "gpt2"),
    "llama": str(MODELS_DIR / "llama-7b"),
}
