###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_prediction_score_drift.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds prediction score drift monitoring output by comparing baseline and
#     current score/risk distributions for each model.
#
# Run:
#     python -m src.modeling.evaluation.build_prediction_score_drift
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "model_key",
    "model_name",
    "score_column",
    "baseline_row_count",
    "current_row_count",
    "baseline_mean",
    "current_mean",
    "mean_shift",
    "baseline_median",
    "current_median",
    "median_shift",
    "baseline_std",
    "current_std",
    "baseline_min",
    "current_min",
    "baseline_max",
    "current_max",
    "score_drift_threshold",
    "drift_status",
    "baseline_run_id",
    "baseline_version",
    "event_timestamp_utc",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_score_drift(
    mean_shift: float,
    score_drift_threshold: float,
) -> str:
    if pd.isna(mean_shift):
        return "UNKNOWN"

    if abs(float(mean_shift)) >= float(score_drift_threshold):
        return "DRIFT"

    return "STABLE"


def build_prediction_score_drift(
    baseline_scoring_outputs: List[pd.DataFrame],
    current_scoring_outputs: List[pd.DataFrame],
    run_id: str,
    layer_name: str,
    domain_name: str,
    score_drift_threshold: float = 0.05,
    baseline_run_id: Optional[str] = None,
    baseline_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build prediction score drift summary.

    This first implementation compares the current scoring output against the
    current run as baseline. Future production use can pass historical baseline
    scoring outputs and current scoring outputs separately.
    """

    rows = []

    baseline_by_model = {
        str(df["model_key"].iloc[0]): df
        for df in baseline_scoring_outputs
        if not df.empty and "model_key" in df.columns
    }

    current_by_model = {
        str(df["model_key"].iloc[0]): df
        for df in current_scoring_outputs
        if not df.empty and "model_key" in df.columns
    }

    for model_key, current_df in current_by_model.items():
        baseline_df = baseline_by_model.get(model_key)

        if baseline_df is None or baseline_df.empty:
            continue

        model_name = (
            str(current_df["model_name"].iloc[0])
            if "model_name" in current_df.columns and not current_df.empty
            else model_key
        )

        candidate_score_columns = [
            column
            for column in current_df.columns
            if column.endswith("_score")
            or column.endswith("_risk_score")
            or column in {"score", "risk_score", "prediction_probability"}
        ]

        if not candidate_score_columns:
            continue

        score_column = candidate_score_columns[0]

        if score_column not in baseline_df.columns:
            continue

        baseline_scores = pd.to_numeric(
            baseline_df[score_column],
            errors="coerce",
        ).dropna()

        current_scores = pd.to_numeric(
            current_df[score_column],
            errors="coerce",
        ).dropna()

        if baseline_scores.empty or current_scores.empty:
            continue

        baseline_mean = float(baseline_scores.mean())
        current_mean = float(current_scores.mean())
        mean_shift = current_mean - baseline_mean

        baseline_median = float(baseline_scores.median())
        current_median = float(current_scores.median())
        median_shift = current_median - baseline_median

        rows.append(
            {
                "run_id": run_id,
                "layer_name": layer_name,
                "domain_name": domain_name,
                "model_key": model_key,
                "model_name": model_name,
                "score_column": score_column,
                "baseline_row_count": int(len(baseline_scores)),
                "current_row_count": int(len(current_scores)),
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "mean_shift": float(mean_shift),
                "baseline_median": baseline_median,
                "current_median": current_median,
                "median_shift": float(median_shift),
                "baseline_std": float(baseline_scores.std()),
                "current_std": float(current_scores.std()),
                "baseline_min": float(baseline_scores.min()),
                "current_min": float(current_scores.min()),
                "baseline_max": float(baseline_scores.max()),
                "current_max": float(current_scores.max()),
                "score_drift_threshold": float(score_drift_threshold),
                "drift_status": classify_score_drift(
                    mean_shift=mean_shift,
                    score_drift_threshold=score_drift_threshold,
                ),
                "baseline_run_id": baseline_run_id,
                "baseline_version": baseline_version,
                "event_timestamp_utc": utc_now_iso(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    return pd.DataFrame(rows)[DEFAULT_OUTPUT_COLUMNS]


def main() -> None:
    baseline = pd.DataFrame(
        {
            "model_key": ["high_cost"] * 5,
            "model_name": ["High Cost"] * 5,
            "risk_score": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )

    current = pd.DataFrame(
        {
            "model_key": ["high_cost"] * 5,
            "model_name": ["High Cost"] * 5,
            "risk_score": [0.15, 0.25, 0.35, 0.45, 0.55],
        }
    )

    output = build_prediction_score_drift(
        baseline_scoring_outputs=[baseline],
        current_scoring_outputs=[current],
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    print("Prediction score drift validation successful.")
    print(output)


if __name__ == "__main__":
    main()