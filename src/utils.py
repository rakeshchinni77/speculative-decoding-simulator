"""
Utility functions for model loading, device management, and logging.
"""

import os
import sys
import logging
from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"


def get_device(preference: str = "auto") -> torch.device:
    """
    Determines and returns the torch device based on preference and availability.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preference == "cpu":
        return torch.device("cpu")
    elif preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def setup_logger(name: str = "speculative_decoding", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs to stderr and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if already added
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[torch.device] = None,
    hf_token: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads a causal language model and tokenizer from local cache if present,
    or downloads from Hugging Face hub.
    """
    if device is None:
        device = get_device()

    token = hf_token or os.getenv("HF_TOKEN") or None

    # Attempt fast offline load first to avoid network latency if cached
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=token,
            local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=token
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=token,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=token,
            low_cpu_mem_usage=True
        )

    model.to(device)
    model.eval()

    return model, tokenizer
