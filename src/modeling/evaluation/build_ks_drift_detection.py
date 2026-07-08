###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_ks_drift_detection.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds Kolmogorov-Smirnov (KS) drift detection output for numeric model
#     features.
#
# Notes:
#     KS drift is most appropriate for continuous numeric features. Categorical
#     and boolean features are skipped and can be monitored through PSI.
#
# Run:
#     python -m src.modeling.evaluation.build_ks_drift_detection
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
    "feature_name",
    "feature_kind",
    "baseline_row_count",
    "current_row_count",
    "ks_statistic",
    "ks_threshold",
    "drift_status",
    "baseline_mean",
    "current_mean",
    "baseline_median",
    "current_median",
    "baseline_std",
    "current_std",
    "baseline_min",
    "current_min",
    "baseline_max",
    "current_max",
    "baseline_run_id",
    "baseline_version",
    "event_timestamp_utc",
]


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def classify_ks_status(
    ks_statistic: float,
    ks_threshold: float,
) -> str:
    """
    Classify KS drift status using configured threshold.
    """

    if pd.isna(ks_statistic):
        return "UNKNOWN"

    if ks_statistic >= ks_threshold:
        return "DRIFT"

    return "STABLE"


def infer_feature_kind(
    dataframe: pd.DataFrame,
    feature_name: str,
) -> str:
    """
    Infer feature kind for KS drift detection.
    """

    if feature_name not in dataframe.columns:
        return "unknown"

    series = dataframe[feature_name]

    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def calculate_ks_statistic(
    baseline_values: pd.Series,
    current_values: pd.Series,
) -> float:
    """
    Calculate the two-sample Kolmogorov-Smirnov statistic without requiring
    scipy.

    The KS statistic is the maximum absolute difference between the empirical
    cumulative distribution functions of the baseline and current samples.
    """

    baseline_array = np.sort(
        pd.to_numeric(baseline_values, errors="coerce").dropna().to_numpy()
    )

    current_array = np.sort(
        pd.to_numeric(current_values, errors="coerce").dropna().to_numpy()
    )

    if len(baseline_array) == 0 or len(current_array) == 0:
        return float("nan")

    combined_values = np.sort(np.unique(np.concatenate([baseline_array, current_array])))

    baseline_cdf = np.searchsorted(
        baseline_array,
        combined_values,
        side="right",
    ) / len(baseline_array)

    current_cdf = np.searchsorted(
        current_array,
        combined_values,
        side="right",
    ) / len(current_array)

    return float(np.max(np.abs(baseline_cdf - current_cdf)))


def build_ks_drift_detection(
    baseline_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    feature_columns: List[str],
    run_id: str,
    layer_name: str,
    domain_name: str,
    ks_threshold: float = 0.10,
    baseline_run_id: Optional[str] = None,
    baseline_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build KS drift detection output for numeric model features.
    """

    rows = []

    for feature_name in feature_columns:
        if feature_name not in baseline_dataframe.columns:
            continue

        if feature_name not in current_dataframe.columns:
            continue

        feature_kind = infer_feature_kind(
            dataframe=baseline_dataframe,
            feature_name=feature_name,
        )

        if feature_kind != "numeric":
            continue

        baseline_series = pd.to_numeric(
            baseline_dataframe[feature_name],
            errors="coerce",
        ).dropna()

        current_series = pd.to_numeric(
            current_dataframe[feature_name],
            errors="coerce",
        ).dropna()

        ks_statistic = calculate_ks_statistic(
            baseline_values=baseline_series,
            current_values=current_series,
        )

        rows.append(
            {
                "run_id": run_id,
                "layer_name": layer_name,
                "domain_name": domain_name,
                "feature_name": feature_name,
                "feature_kind": feature_kind,
                "baseline_row_count": int(len(baseline_series)),
                "current_row_count": int(len(current_series)),
                "ks_statistic": ks_statistic,
                "ks_threshold": float(ks_threshold),
                "drift_status": classify_ks_status(
                    ks_statistic=ks_statistic,
                    ks_threshold=ks_threshold,
                ),
                "baseline_mean": float(baseline_series.mean()) if len(baseline_series) else np.nan,
                "current_mean": float(current_series.mean()) if len(current_series) else np.nan,
                "baseline_median": float(baseline_series.median()) if len(baseline_series) else np.nan,
                "current_median": float(current_series.median()) if len(current_series) else np.nan,
                "baseline_std": float(baseline_series.std()) if len(baseline_series) else np.nan,
                "current_std": float(current_series.std()) if len(current_series) else np.nan,
                "baseline_min": float(baseline_series.min()) if len(baseline_series) else np.nan,
                "current_min": float(current_series.min()) if len(current_series) else np.nan,
                "baseline_max": float(baseline_series.max()) if len(baseline_series) else np.nan,
                "current_max": float(current_series.max()) if len(current_series) else np.nan,
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
        {
            "age": [20, 30, 40, 50, 60, 70],
            "city": ["Phoenix", "Phoenix", "Mesa", "Mesa", "Tempe", "Tucson"],
        }
    )

    current = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 75, 85],
            "city": ["Phoenix", "Mesa", "Mesa", "Tucson", "Tucson", "Flagstaff"],
        }
    )

    output = build_ks_drift_detection(
        baseline_dataframe=baseline,
        current_dataframe=current,
        feature_columns=["age", "city"],
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    print("KS drift detection validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()