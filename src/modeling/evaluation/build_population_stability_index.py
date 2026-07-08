###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_population_stability_index.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds Population Stability Index (PSI) drift monitoring output.
#
# Run:
#     python -m src.modeling.evaluation.build_population_stability_index
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "feature_name",
    "feature_kind",
    "bin_number",
    "bin_label",
    "baseline_count",
    "current_count",
    "baseline_pct",
    "current_pct",
    "psi_value",
    "feature_psi",
    "psi_threshold",
    "drift_status",
    "baseline_row_count",
    "current_row_count",
    "baseline_run_id",
    "baseline_version",
    "event_timestamp_utc",
]


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def classify_psi_status(
    psi_value: float,
    psi_threshold: float,
) -> str:
    """
    Classify PSI status using configured threshold.
    """

    if pd.isna(psi_value):
        return "UNKNOWN"

    if psi_value >= psi_threshold:
        return "DRIFT"

    return "STABLE"


def build_numeric_bins(
    series: pd.Series,
    bin_count: int,
) -> List[float]:
    """
    Build numeric quantile bin edges.
    """

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    if numeric_series.empty:
        return []

    quantiles = np.linspace(0, 1, bin_count + 1)
    edges = numeric_series.quantile(quantiles).drop_duplicates().tolist()

    if len(edges) < 2:
        minimum = float(numeric_series.min())
        maximum = float(numeric_series.max())

        if minimum == maximum:
            return [minimum - 0.5, maximum + 0.5]

        return [minimum, maximum]

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def calculate_psi_component(
    baseline_pct: float,
    current_pct: float,
    epsilon: float,
) -> float:
    """
    Calculate PSI contribution for one bin.
    """

    baseline_safe = max(float(baseline_pct), epsilon)
    current_safe = max(float(current_pct), epsilon)

    return float((current_safe - baseline_safe) * np.log(current_safe / baseline_safe))


def build_numeric_feature_psi(
    feature_name: str,
    feature_kind: str,
    baseline_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    bin_count: int,
    epsilon: float,
) -> pd.DataFrame:
    """
    Build PSI rows for one numeric feature.
    """

    baseline_series = pd.to_numeric(
        baseline_dataframe[feature_name],
        errors="coerce",
    )

    current_series = pd.to_numeric(
        current_dataframe[feature_name],
        errors="coerce",
    )

    edges = build_numeric_bins(
        series=baseline_series,
        bin_count=bin_count,
    )

    if not edges:
        return pd.DataFrame()

    baseline_bins = pd.cut(
        baseline_series,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    current_bins = pd.cut(
        current_series,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    bin_labels = [str(category) for category in baseline_bins.cat.categories]

    baseline_counts = baseline_bins.value_counts(sort=False)
    current_counts = current_bins.value_counts(sort=False)

    baseline_row_count = int(baseline_series.notna().sum())
    current_row_count = int(current_series.notna().sum())

    rows = []

    for index, category in enumerate(baseline_bins.cat.categories, start=1):
        baseline_count = int(baseline_counts.get(category, 0))
        current_count = int(current_counts.get(category, 0))

        baseline_pct = baseline_count / baseline_row_count if baseline_row_count else 0.0
        current_pct = current_count / current_row_count if current_row_count else 0.0

        rows.append(
            {
                "feature_name": feature_name,
                "feature_kind": feature_kind,
                "bin_number": index,
                "bin_label": bin_labels[index - 1],
                "baseline_count": baseline_count,
                "current_count": current_count,
                "baseline_pct": baseline_pct,
                "current_pct": current_pct,
                "psi_value": calculate_psi_component(
                    baseline_pct=baseline_pct,
                    current_pct=current_pct,
                    epsilon=epsilon,
                ),
                "baseline_row_count": baseline_row_count,
                "current_row_count": current_row_count,
            }
        )

    return pd.DataFrame(rows)


def build_categorical_feature_psi(
    feature_name: str,
    feature_kind: str,
    baseline_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    max_categories: int,
    epsilon: float,
) -> pd.DataFrame:
    """
    Build PSI rows for one categorical feature.
    """

    baseline_series = baseline_dataframe[feature_name].fillna("__NULL__").astype(str)
    current_series = current_dataframe[feature_name].fillna("__NULL__").astype(str)

    top_categories = (
        baseline_series
        .value_counts(dropna=False)
        .head(max_categories)
        .index
        .tolist()
    )

    baseline_bucketed = baseline_series.where(
        baseline_series.isin(top_categories),
        "__OTHER__",
    )

    current_bucketed = current_series.where(
        current_series.isin(top_categories),
        "__OTHER__",
    )

    categories = sorted(
        set(baseline_bucketed.unique().tolist())
        | set(current_bucketed.unique().tolist())
    )

    baseline_counts = baseline_bucketed.value_counts(dropna=False)
    current_counts = current_bucketed.value_counts(dropna=False)

    baseline_row_count = int(len(baseline_bucketed))
    current_row_count = int(len(current_bucketed))

    rows = []

    for index, category in enumerate(categories, start=1):
        baseline_count = int(baseline_counts.get(category, 0))
        current_count = int(current_counts.get(category, 0))

        baseline_pct = baseline_count / baseline_row_count if baseline_row_count else 0.0
        current_pct = current_count / current_row_count if current_row_count else 0.0

        rows.append(
            {
                "feature_name": feature_name,
                "feature_kind": feature_kind,
                "bin_number": index,
                "bin_label": str(category),
                "baseline_count": baseline_count,
                "current_count": current_count,
                "baseline_pct": baseline_pct,
                "current_pct": current_pct,
                "psi_value": calculate_psi_component(
                    baseline_pct=baseline_pct,
                    current_pct=current_pct,
                    epsilon=epsilon,
                ),
                "baseline_row_count": baseline_row_count,
                "current_row_count": current_row_count,
            }
        )

    return pd.DataFrame(rows)


def infer_feature_kind(
    dataframe: pd.DataFrame,
    feature_name: str,
) -> str:
    """
    Infer feature kind for PSI binning.

    Boolean columns must be detected before numeric columns because pandas may
    treat boolean values as numeric-like. PSI should handle binary features as
    categorical buckets rather than numeric quantile bins.
    """

    if feature_name not in dataframe.columns:
        return "unknown"

    series = dataframe[feature_name]

    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def build_population_stability_index(
    baseline_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    feature_columns: List[str],
    run_id: str,
    layer_name: str,
    domain_name: str,
    psi_threshold: float = 0.20,
    numeric_bin_count: int = 10,
    categorical_max_categories: int = 20,
    epsilon: float = 0.0001,
    baseline_run_id: Optional[str] = None,
    baseline_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build Population Stability Index output for configured features.
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

        if feature_kind in {"numeric"}:
            feature_psi_df = build_numeric_feature_psi(
                feature_name=feature_name,
                feature_kind=feature_kind,
                baseline_dataframe=baseline_dataframe,
                current_dataframe=current_dataframe,
                bin_count=numeric_bin_count,
                epsilon=epsilon,
            )
        else:
            feature_psi_df = build_categorical_feature_psi(
                feature_name=feature_name,
                feature_kind=feature_kind,
                baseline_dataframe=baseline_dataframe,
                current_dataframe=current_dataframe,
                max_categories=categorical_max_categories,
                epsilon=epsilon,
            )

        if feature_psi_df.empty:
            continue

        feature_psi = float(feature_psi_df["psi_value"].sum())

        feature_psi_df["run_id"] = run_id
        feature_psi_df["layer_name"] = layer_name
        feature_psi_df["domain_name"] = domain_name
        feature_psi_df["feature_psi"] = feature_psi
        feature_psi_df["psi_threshold"] = float(psi_threshold)
        feature_psi_df["drift_status"] = classify_psi_status(
            psi_value=feature_psi,
            psi_threshold=psi_threshold,
        )
        feature_psi_df["baseline_run_id"] = baseline_run_id
        feature_psi_df["baseline_version"] = baseline_version
        feature_psi_df["event_timestamp_utc"] = utc_now_iso()

        rows.append(feature_psi_df)

    if not rows:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    output_df = pd.concat(rows, ignore_index=True)

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

    output = build_population_stability_index(
        baseline_dataframe=baseline,
        current_dataframe=current,
        feature_columns=["age", "city"],
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    print("Population Stability Index validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()