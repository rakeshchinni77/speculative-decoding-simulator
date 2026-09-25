"""
Configuration module for speculative decoding simulation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Model identifiers
DRAFT_MODEL_ID = os.getenv("DRAFT_MODEL_ID", "gpt2")
TARGET_MODEL_ID = os.getenv("TARGET_MODEL_ID", "gpt2-large")

# Device configuration
DEVICE = os.getenv("DEVICE", "auto")

# Hyperparameters
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "15"))
DEFAULT_N_DRAFT = int(os.getenv("DEFAULT_N_DRAFT", "4"))

# Directory paths
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RESULTS_DIR = BASE_DIR / os.getenv("RESULTS_DIR", "results")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
