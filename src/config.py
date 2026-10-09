"""
Configuration settings for the agent, models, and paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure required directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Local GGUF models config
LOCAL_MODELS = {
    "qwen": {
        "name": "Qwen 2.5 3B Instruct",
        "path": MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "chat_format": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>", "Observation:", "\nObservation:"],
    },
    "llama": {
        "name": "Llama 3.2 3B Instruct",
        "path": MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "chat_format": "llama-3",
        "stop": ["<|eot_id|>", "<|end_of_text|>", "Observation:", "\nObservation:"],
    },
    "maple": {
        "name": "Maple-Preview 20B-A1B",
        "path": MODELS_DIR / "maple-preview-TQ1_0-head-Q4_K.gguf",
        "chat_format": None,
        "stop": ["Observation:", "\nObservation:"],
    }
}

# Default runtime options
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "llama")
DEFAULT_LOCAL_MODEL = os.getenv("DEFAULT_LOCAL_MODEL", "qwen")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# llama.cpp inference parameters
DEFAULT_CTX_SIZE = 4096
DEFAULT_N_THREADS = 8  # Matched to physical cores on Ryzen 7 5700U
DEFAULT_TEMPERATURE = 0.2  # Lower temperature for reliable reasoning and tool use
