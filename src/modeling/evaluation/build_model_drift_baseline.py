###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_model_drift_baseline.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Release:
#     Release 2D - Modeling Framework Completion
#
# Purpose:
#     Builds a model drift baseline from the current modeling feature matrix.
#
#     This file prepares the reference baseline that future drift monitoring
#     jobs will compare against.
#
# Outputs:
#     data/modeling/model_drift_baseline.parquet
#
# Run:
#     python -m src.modeling.evaluation.build_model_drift_baseline
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.modeling.evaluation.build_feature_baseline_statistics import (
    build_feature_baseline_statistics,
)


LAYER_NAME = "Layer 2D - Enterprise Modeling Framework"
DOMAIN_NAME = "Modeling"

DEFAULT_INPUT_PATH = Path("data/modeling/modeling_feature_matrix.parquet")
DEFAULT_OUTPUT_PATH = Path("data/modeling/model_drift_baseline.parquet")

DEFAULT_BASELINE_TYPE = "model_drift_baseline"
DEFAULT_BASELINE_VERSION = "1.0"

DEFAULT_EXCLUDED_COLUMNS = [
    "member_id",
    "target",
    "target_column",
    "target_name",
    "label",
    "prediction",
    "prediction_probability",
    "risk_score",
    "risk_tier",
    "train_test_split",
    "fold",
]


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def resolve_run_id() -> str:
    """
    Build a simple run identifier for local execution.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_model_drift_baseline")


def identify_feature_columns(
    dataframe: pd.DataFrame,
    excluded_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify candidate feature columns for drift baseline creation.

    The drift baseline should describe model input features only.
    Operational columns, target columns, predictions, and scoring outputs
    are excluded.
    """

    excluded = set(excluded_columns or DEFAULT_EXCLUDED_COLUMNS)

    feature_columns: List[str] = []

    for column_name in dataframe.columns:
        column_lower = column_name.lower()

        if column_name in excluded:
            continue

        if column_lower in excluded:
            continue

        if column_lower.startswith("target_"):
            continue

        if column_lower.endswith("_target"):
            continue

        if column_lower.startswith("prediction"):
            continue

        if column_lower.startswith("score_"):
            continue

        feature_columns.append(column_name)

    return feature_columns


def build_model_drift_baseline(
    dataframe: pd.DataFrame,
    run_id: str,
    layer_name: str = LAYER_NAME,
    domain_name: str = DOMAIN_NAME,
    excluded_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the model drift baseline from a modeling feature matrix.

    This function intentionally reuses build_feature_baseline_statistics()
    so feature profiling, feature-kind classification, null handling,
    numeric summaries, binary summaries, datetime summaries, and categorical
    summaries remain consistent across MedFabric evaluation assets.
    """

    feature_columns = identify_feature_columns(
        dataframe=dataframe,
        excluded_columns=excluded_columns,
    )

    baseline = build_feature_baseline_statistics(
        dataframe=dataframe,
        feature_columns=feature_columns,
        run_id=run_id,
        layer_name=layer_name,
        domain_name=domain_name,
        excluded_columns=excluded_columns,
    )

    if baseline.empty:
        return baseline

    baseline["baseline_type"] = DEFAULT_BASELINE_TYPE
    baseline["baseline_version"] = DEFAULT_BASELINE_VERSION
    baseline["drift_baseline_created_utc"] = utc_now_iso()
    baseline["drift_reference_dataset"] = str(DEFAULT_INPUT_PATH)

    return baseline


def write_model_drift_baseline(
    baseline: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """
    Write the model drift baseline to parquet.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(output_path, index=False)


def run_model_drift_baseline_build(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Execute the complete model drift baseline build.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Modeling feature matrix not found: {input_path}. "
            "Run the modeling layer first."
        )

    run_id = resolve_run_id()

    dataframe = pd.read_parquet(input_path)

    baseline = build_model_drift_baseline(
        dataframe=dataframe,
        run_id=run_id,
        layer_name=LAYER_NAME,
        domain_name=DOMAIN_NAME,
        excluded_columns=DEFAULT_EXCLUDED_COLUMNS,
    )

    write_model_drift_baseline(
        baseline=baseline,
        output_path=output_path,
    )

    return baseline


def main() -> None:
    """
    Main entry point for command-line execution.
    """

    baseline = run_model_drift_baseline_build()

    print("Model drift baseline build completed successfully.")
    print(f"Output: {DEFAULT_OUTPUT_PATH}")
    print(f"Rows: {len(baseline)}")

    if not baseline.empty:
        print(
            baseline[
                [
                    "feature_name",
                    "feature_kind",
                    "row_count",
                    "null_pct",
                    "unique_count",
                    "baseline_type",
                    "baseline_version",
                ]
            ].head(20)
        )


if __name__ == "__main__":
    main()