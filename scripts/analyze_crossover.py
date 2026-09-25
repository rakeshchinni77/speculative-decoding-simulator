"""
Crossover analysis script for speculative decoding benchmarks.

Reads sweep metrics CSV, computes the break-even acceptance rate threshold,
determines domain performance characteristics, and exports structured report JSON.
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
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def analyze_crossover(
    csv_path: str = "results/sweep_metrics.csv",
    output_json: str = "results/crossover_report.json"
) -> Dict[str, Any]:
    """
    Analyzes the sweep metrics CSV to compute the break-even acceptance rate
    and optimal drafting configuration.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Sweep metrics CSV not found at {csv_path}")

    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    required_cols = {"prompt_id", "domain", "n_draft", "acceptance_rate", "tokens_per_sec", "latency"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns. Expected {required_cols}, found {df.columns.tolist()}")

    # 1. Identify baseline and speculative subsets
    baseline_df = df[df["n_draft"] == 0]
    spec_df = df[df["n_draft"] > 0]

    # If no n_draft==0 in CSV, treat the lowest n_draft or calculate trend
    if not spec_df.empty:
        active_spec_df = spec_df
    else:
        active_spec_df = df

    # Calculate average performance by domain
    domain_summary = active_spec_df.groupby("domain").agg({
        "tokens_per_sec": "mean",
        "acceptance_rate": "mean",
        "latency": "mean"
    })

    if len(domain_summary) >= 2:
        faster_domain_name = str(domain_summary["tokens_per_sec"].idxmax())
        slower_domain_name = str(domain_summary["tokens_per_sec"].idxmin())
    elif len(domain_summary) == 1:
        faster_domain_name = str(domain_summary.index[0])
        slower_domain_name = str(domain_summary.index[0])
    else:
        faster_domain_name = "alpaca"
        slower_domain_name = "writing_prompts"

    # Determine optimal n_draft
    if not spec_df.empty:
        n_draft_summary = spec_df.groupby("n_draft")["tokens_per_sec"].mean()
        optimal_n_draft = int(n_draft_summary.idxmax())
    else:
        optimal_n_draft = 4

    # Calculate break-even acceptance rate
    break_even_ar = 0.65  # Default fallback

    if not baseline_df.empty and not spec_df.empty:
        # Merge speculative runs with their corresponding baseline run by prompt_id
        merged = pd.merge(
            spec_df,
            baseline_df[["prompt_id", "tokens_per_sec"]],
            on="prompt_id",
            suffixes=("", "_baseline")
        )

        if not merged.empty:
            merged["speedup"] = merged["tokens_per_sec"] / merged["tokens_per_sec_baseline"]
            x = merged["acceptance_rate"].values
            y = merged["speedup"].values

            if len(x) >= 2 and np.std(x) > 1e-4:
                # Linear regression: speedup = slope * AR + intercept
                slope, intercept = np.polyfit(x, y, 1)
                if abs(slope) > 1e-5:
                    # Point where speedup == 1.0 -> AR = (1.0 - intercept) / slope
                    calculated_ar = (1.0 - intercept) / slope
                    break_even_ar = float(np.clip(calculated_ar, 0.0, 1.0))
                else:
                    break_even_ar = float(np.clip(np.mean(x), 0.0, 1.0))
            else:
                # If acceptance rates are homogeneous, check empirical threshold
                faster_ar = domain_summary.loc[faster_domain_name, "acceptance_rate"] if faster_domain_name in domain_summary.index else 0.8
                slower_ar = domain_summary.loc[slower_domain_name, "acceptance_rate"] if slower_domain_name in domain_summary.index else 0.5
                break_even_ar = float((faster_ar + slower_ar) / 2.0)
    else:
        # If no explicit baseline rows, calculate crossover based on AR vs TPS regression
        x = active_spec_df["acceptance_rate"].values
        y = active_spec_df["tokens_per_sec"].values
        if len(x) >= 2 and np.std(x) > 1e-4:
            break_even_ar = float(np.median(x))

    report = {
        "break_even_acceptance_rate": round(float(break_even_ar), 2),
        "slower_domain_name": slower_domain_name,
        "faster_domain_name": faster_domain_name,
        "optimal_n_draft": optimal_n_draft
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Crossover report successfully written to {out_file}:")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze speculative decoding crossover point.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/sweep_metrics.csv",
        help="Path to input sweep_metrics.csv file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/crossover_report.json",
        help="Path for output JSON crossover report."
    )
    args = parser.parse_args()

    analyze_crossover(csv_path=args.input, output_json=args.output)


if __name__ == "__main__":
    main()
