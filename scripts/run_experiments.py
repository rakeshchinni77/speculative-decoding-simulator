"""
CLI experiment runner for speculative decoding benchmarks.

Executes performance sweeps across prompt datasets and draft configurations,
recording latency, throughput, and acceptance rates to CSV.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import argparse
import json
import csv
from typing import List, Dict, Any
import torch

from src.utils import load_model_and_tokenizer, get_device, setup_logger
from src.generators import baseline_generate, speculative_generate
from src.config import (
    DRAFT_MODEL_ID,
    TARGET_MODEL_ID,
    MAX_NEW_TOKENS,
    RESULTS_DIR,
    LOG_DIR
)

logger = setup_logger("run_experiments", str(LOG_DIR / "experiments.log"))


def run_sweeps(
    prompts_path: str,
    output_csv: str = "results/sweep_metrics.csv",
    draft_model_id: str = DRAFT_MODEL_ID,
    target_model_id: str = TARGET_MODEL_ID,
    n_drafts: List[int] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    include_baseline: bool = True
) -> None:
    """
    Runs the benchmark sweep over all prompts in the provided JSON dataset.
    """
    if n_drafts is None:
        n_drafts = [2, 4]

    prompts_file = Path(prompts_path)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts dataset not found at {prompts_path}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data: List[Dict[str, Any]] = json.load(f)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Loading draft model: {draft_model_id} and target model: {target_model_id}")

    draft_model, tokenizer = load_model_and_tokenizer(draft_model_id, device=device)
    if draft_model_id == target_model_id:
        target_model = draft_model
    else:
        target_model, _ = load_model_and_tokenizer(target_model_id, device=device)

    results_rows: List[Dict[str, Any]] = []

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(prompts_data) * (len(n_drafts) + (1 if include_baseline else 0))
    current_run = 0

    logger.info(f"Starting sweeps: {len(prompts_data)} prompts, n_drafts={n_drafts}, include_baseline={include_baseline}")

    for item in prompts_data:
        p_id = str(item["id"])
        domain = str(item.get("domain", "default"))
        prompt_text = str(item["prompt"])

        # 1. Run Baseline (n_draft = 0)
        if include_baseline:
            current_run += 1
            logger.info(f"[{current_run}/{total_runs}] Baseline: prompt_id={p_id}, domain={domain}")
            base_res = baseline_generate(
                model=target_model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                max_new_tokens=max_new_tokens
            )
            results_rows.append({
                "prompt_id": p_id,
                "domain": domain,
                "n_draft": 0,
                "acceptance_rate": 0.0,
                "tokens_per_sec": round(base_res["tokens_per_sec"], 4),
                "latency": round(base_res["latency"], 4)
            })

        # 2. Run Speculative Sweeps
        for n_draft in n_drafts:
            current_run += 1
            logger.info(f"[{current_run}/{total_runs}] Speculative (n_draft={n_draft}): prompt_id={p_id}, domain={domain}")
            spec_res = speculative_generate(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                n_draft=n_draft,
                max_new_tokens=max_new_tokens
            )
            results_rows.append({
                "prompt_id": p_id,
                "domain": domain,
                "n_draft": int(n_draft),
                "acceptance_rate": round(spec_res["acceptance_rate"], 4),
                "tokens_per_sec": round(spec_res["tokens_per_sec"], 4),
                "latency": round(spec_res["latency"], 4)
            })

    # Write out exact required CSV
    fieldnames = ["prompt_id", "domain", "n_draft", "acceptance_rate", "tokens_per_sec", "latency"]
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    logger.info(f"Successfully saved {len(results_rows)} metric rows to {out_path}")
    print(f"Sweep results successfully written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding performance sweeps.")
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/test_prompts.json",
        help="Path to JSON prompt dataset."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sweep_metrics.csv",
        help="Path for output CSV file."
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default=DRAFT_MODEL_ID,
        help="Hugging Face model ID for draft model."
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default=TARGET_MODEL_ID,
        help="Hugging Face model ID for target model."
    )
    parser.add_argument(
        "--n_drafts",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Draft token counts to evaluate (e.g., 2 4 or 2 4 8)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum new tokens to generate per prompt."
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Exclude baseline (n_draft=0) runs."
    )

    args = parser.parse_args()
    run_sweeps(
        prompts_path=args.prompts,
        output_csv=args.output,
        draft_model_id=args.draft_model,
        target_model_id=args.target_model,
        n_drafts=args.n_drafts,
        max_new_tokens=args.max_new_tokens,
        include_baseline=not args.no_baseline
    )


if __name__ == "__main__":
    main()
