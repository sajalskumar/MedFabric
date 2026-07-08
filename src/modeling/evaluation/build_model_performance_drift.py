###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_model_performance_drift.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds model performance drift monitoring output by comparing baseline
#     champion model metrics against current champion model metrics.
#
# Run:
#     python -m src.modeling.evaluation.build_model_performance_drift
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_METRIC_COLUMNS = [
    "metric_accuracy",
    "metric_precision",
    "metric_recall",
    "metric_f1",
    "metric_balanced_accuracy",
    "metric_roc_auc",
    "metric_brier_score",
    "metric_log_loss",
]


DEFAULT_OUTPUT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "model_key",
    "model_name",
    "target_column",
    "champion_algorithm_key",
    "champion_algorithm_name",
    "metric_name",
    "baseline_metric_value",
    "current_metric_value",
    "absolute_change",
    "relative_change_pct",
    "performance_drift_threshold",
    "drift_status",
    "baseline_run_id",
    "baseline_version",
    "event_timestamp_utc",
]


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def classify_performance_drift(
    absolute_change: float,
    performance_drift_threshold: float,
) -> str:
    """
    Classify model performance drift using absolute metric change.

    A model is considered drifted when the absolute change in a monitored
    metric is greater than or equal to the configured threshold.
    """

    if pd.isna(absolute_change):
        return "UNKNOWN"

    if abs(float(absolute_change)) >= float(performance_drift_threshold):
        return "DRIFT"

    return "STABLE"


def calculate_relative_change_pct(
    baseline_value: Optional[float],
    current_value: Optional[float],
) -> Optional[float]:
    """
    Calculate relative percentage change between baseline and current values.
    """

    if baseline_value is None or current_value is None:
        return None

    if pd.isna(baseline_value) or pd.isna(current_value):
        return None

    if float(baseline_value) == 0.0:
        return None

    return float((current_value - baseline_value) / baseline_value * 100.0)


def get_metric_columns(
    dataframe: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Return metric columns available in the supplied dataframe.
    """

    requested_metrics = metric_columns or DEFAULT_METRIC_COLUMNS

    return [
        column
        for column in requested_metrics
        if column in dataframe.columns
    ]


def build_model_performance_drift(
    baseline_champion_summary: pd.DataFrame,
    current_champion_summary: pd.DataFrame,
    run_id: str,
    layer_name: str,
    domain_name: str,
    performance_drift_threshold: float = 0.05,
    metric_columns: Optional[List[str]] = None,
    baseline_run_id: Optional[str] = None,
    baseline_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build model performance drift summary.

    Parameters
    ----------
    baseline_champion_summary:
        Baseline champion model summary dataframe.

    current_champion_summary:
        Current champion model summary dataframe.

    run_id:
        Current Modeling run identifier.

    layer_name:
        MedFabric layer name.

    domain_name:
        Modeling domain name.

    performance_drift_threshold:
        Absolute metric change threshold used to classify drift.

    metric_columns:
        Optional list of metric columns to monitor. If omitted, the standard
        MedFabric champion metric columns are used.

    baseline_run_id:
        Optional baseline run identifier.

    baseline_version:
        Optional baseline version label.

    Returns
    -------
    pd.DataFrame
        Model performance drift summary.
    """

    if baseline_champion_summary.empty or current_champion_summary.empty:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    metric_columns_to_check = get_metric_columns(
        dataframe=current_champion_summary,
        metric_columns=metric_columns,
    )

    if not metric_columns_to_check:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    baseline_lookup: Dict[str, pd.Series] = {}

    for _, row in baseline_champion_summary.iterrows():
        model_key = str(row.get("model_key"))
        baseline_lookup[model_key] = row

    rows = []

    for _, current_row in current_champion_summary.iterrows():
        model_key = str(current_row.get("model_key"))

        if model_key not in baseline_lookup:
            continue

        baseline_row = baseline_lookup[model_key]

        for metric_name in metric_columns_to_check:
            baseline_metric_value = pd.to_numeric(
                baseline_row.get(metric_name),
                errors="coerce",
            )

            current_metric_value = pd.to_numeric(
                current_row.get(metric_name),
                errors="coerce",
            )

            if pd.isna(baseline_metric_value) or pd.isna(current_metric_value):
                absolute_change = None
                relative_change_pct = None
            else:
                baseline_metric_value = float(baseline_metric_value)
                current_metric_value = float(current_metric_value)
                absolute_change = current_metric_value - baseline_metric_value
                relative_change_pct = calculate_relative_change_pct(
                    baseline_value=baseline_metric_value,
                    current_value=current_metric_value,
                )

            rows.append(
                {
                    "run_id": run_id,
                    "layer_name": layer_name,
                    "domain_name": domain_name,
                    "model_key": model_key,
                    "model_name": current_row.get("model_name"),
                    "target_column": current_row.get("target_column"),
                    "champion_algorithm_key": current_row.get(
                        "champion_algorithm_key"
                    ),
                    "champion_algorithm_name": current_row.get(
                        "champion_algorithm_name"
                    ),
                    "metric_name": metric_name,
                    "baseline_metric_value": baseline_metric_value,
                    "current_metric_value": current_metric_value,
                    "absolute_change": absolute_change,
                    "relative_change_pct": relative_change_pct,
                    "performance_drift_threshold": float(
                        performance_drift_threshold
                    ),
                    "drift_status": classify_performance_drift(
                        absolute_change=absolute_change,
                        performance_drift_threshold=performance_drift_threshold,
                    ),
                    "baseline_run_id": baseline_run_id,
                    "baseline_version": baseline_version,
                    "event_timestamp_utc": utc_now_iso(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    output_df = pd.DataFrame(rows)

    return output_df[DEFAULT_OUTPUT_COLUMNS]


def main() -> None:
    """
    Lightweight local validation.
    """

    baseline = pd.DataFrame(
        [
            {
                "model_key": "high_cost",
                "model_name": "High Cost",
                "target_column": "high_cost_target",
                "champion_algorithm_key": "logistic_regression",
                "champion_algorithm_name": "Logistic Regression",
                "metric_roc_auc": 0.80,
                "metric_f1": 0.50,
            }
        ]
    )

    current = pd.DataFrame(
        [
            {
                "model_key": "high_cost",
                "model_name": "High Cost",
                "target_column": "high_cost_target",
                "champion_algorithm_key": "logistic_regression",
                "champion_algorithm_name": "Logistic Regression",
                "metric_roc_auc": 0.72,
                "metric_f1": 0.45,
            }
        ]
    )

    output = build_model_performance_drift(
        baseline_champion_summary=baseline,
        current_champion_summary=current,
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        performance_drift_threshold=0.05,
    )

    print("Model performance drift validation successful.")
    print(output)


if __name__ == "__main__":
    main()