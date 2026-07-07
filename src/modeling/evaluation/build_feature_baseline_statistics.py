###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_feature_baseline_statistics.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Release:
#     Release 2D - Modeling Framework Completion
#
# Purpose:
#     Builds baseline statistics for modeling features.
#
#     Enhanced feature kind detection supports:
#       - numeric
#       - binary
#       - categorical
#       - datetime
#       - identifier
#       - constant
#
# Run:
#     python -m src.modeling.evaluation.build_feature_baseline_statistics
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


NUMERIC_KIND = "numeric"
BINARY_KIND = "binary"
CATEGORICAL_KIND = "categorical"
DATETIME_KIND = "datetime"
IDENTIFIER_KIND = "identifier"
CONSTANT_KIND = "constant"

DEFAULT_BASELINE_TYPE = "training_feature_baseline"
DEFAULT_BASELINE_VERSION = "1.1"

IDENTIFIER_NAME_HINTS = (
    "id",
    "_id",
    "member_id",
    "patient_id",
    "claim_id",
    "provider_id",
    "encounter_id",
    "record_id",
    "uuid",
    "guid",
    "key",
)

DATETIME_NAME_HINTS = (
    "date",
    "_dt",
    "datetime",
    "timestamp",
    "created_at",
    "updated_at",
)


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def looks_like_identifier(feature_name: str, series: pd.Series) -> bool:
    """
    Identify high-cardinality ID-like fields.

    Identifier fields should usually be excluded from drift calculations because
    their distribution is not analytically meaningful.
    """

    name = feature_name.lower()
    row_count = len(series)
    non_null_count = int(series.notna().sum())

    if row_count == 0 or non_null_count == 0:
        return False

    unique_count = int(series.nunique(dropna=True))
    unique_pct = unique_count / non_null_count if non_null_count else 0.0

    has_identifier_name = (
        name == "id"
        or name.endswith("_id")
        or name.endswith("_key")
        or name in IDENTIFIER_NAME_HINTS
    )

    return bool(has_identifier_name and unique_pct >= 0.90)


def looks_like_datetime(feature_name: str, series: pd.Series) -> bool:
    """
    Identify datetime-like fields using dtype first, then safe parsing.

    This helps prevent fields such as birth_date from being treated as
    high-cardinality categorical variables.
    """

    name = feature_name.lower()

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    has_datetime_name = any(hint in name for hint in DATETIME_NAME_HINTS)

    if not has_datetime_name:
        return False

    non_null_series = series.dropna()

    if non_null_series.empty:
        return False

    parsed = pd.to_datetime(non_null_series, errors="coerce")
    parse_success_pct = float(parsed.notna().mean())

    return parse_success_pct >= 0.80


def is_binary_feature(series: pd.Series) -> bool:
    """
    Identify binary features.

    Supports:
      - boolean columns
      - numeric 0/1 columns
      - text yes/no, true/false, y/n columns
    """

    non_null_series = series.dropna()

    if non_null_series.empty:
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    normalized_values = set(
        non_null_series.astype(str).str.strip().str.lower().unique().tolist()
    )

    binary_value_sets = [
        {"0", "1"},
        {"0.0", "1.0"},
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
    ]

    return any(normalized_values.issubset(value_set) for value_set in binary_value_sets)


def classify_feature_kind(feature_name: str, series: pd.Series) -> str:
    """
    Classify a feature into an enterprise modeling feature kind.

    Classification order matters:
      1. constant
      2. identifier
      3. datetime
      4. binary
      5. numeric
      6. categorical
    """

    non_null_count = int(series.notna().sum())
    unique_count = int(series.nunique(dropna=True))

    if non_null_count > 0 and unique_count <= 1:
        return CONSTANT_KIND

    if looks_like_identifier(feature_name=feature_name, series=series):
        return IDENTIFIER_KIND

    if looks_like_datetime(feature_name=feature_name, series=series):
        return DATETIME_KIND

    if is_binary_feature(series):
        return BINARY_KIND

    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC_KIND

    return CATEGORICAL_KIND


def build_numeric_statistics(series: pd.Series) -> Dict[str, Optional[float]]:
    """
    Build numeric baseline statistics for one feature.

    Important:
        Boolean values must be explicitly converted to float before quantile
        calculations. Otherwise NumPy may fail with:
        "numpy boolean subtract, the `-` operator, is not supported"
    """

    if pd.api.types.is_bool_dtype(series):
        numeric_series = series.astype("float64")
    else:
        numeric_series = pd.to_numeric(series, errors="coerce").astype("float64")

    non_null_series = numeric_series.dropna()
    row_count = len(series)

    if non_null_series.empty:
        return {
            "numeric_non_null_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "zero_count": 0,
            "zero_pct": 0.0,
            "negative_count": 0,
            "negative_pct": 0.0,
        }

    zero_count = int((non_null_series == 0).sum())
    negative_count = int((non_null_series < 0).sum())

    return {
        "numeric_non_null_count": int(len(non_null_series)),
        "mean": float(non_null_series.mean()),
        "std": float(non_null_series.std()) if len(non_null_series) > 1 else 0.0,
        "min": float(non_null_series.min()),
        "p01": float(non_null_series.quantile(0.01)),
        "p05": float(non_null_series.quantile(0.05)),
        "median": float(non_null_series.quantile(0.50)),
        "p95": float(non_null_series.quantile(0.95)),
        "p99": float(non_null_series.quantile(0.99)),
        "max": float(non_null_series.max()),
        "zero_count": zero_count,
        "zero_pct": zero_count / row_count if row_count else 0.0,
        "negative_count": negative_count,
        "negative_pct": negative_count / row_count if row_count else 0.0,
    }
def build_binary_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Build binary baseline statistics for one feature.
    """

    non_null_series = series.dropna()
    row_count = len(series)

    normalized = non_null_series.astype(str).str.strip().str.lower()

    positive_values = {"1", "1.0", "true", "yes", "y"}
    negative_values = {"0", "0.0", "false", "no", "n"}

    positive_count = int(normalized.isin(positive_values).sum())
    negative_count = int(normalized.isin(negative_values).sum())

    return {
        "binary_non_null_count": int(len(non_null_series)),
        "positive_count": positive_count,
        "positive_pct": positive_count / row_count if row_count else 0.0,
        "negative_binary_count": negative_count,
        "negative_binary_pct": negative_count / row_count if row_count else 0.0,
    }


def build_datetime_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Build datetime baseline statistics for one feature.
    """

    parsed = pd.to_datetime(series, errors="coerce")
    non_null_series = parsed.dropna()

    if non_null_series.empty:
        return {
            "datetime_non_null_count": 0,
            "min_datetime": None,
            "max_datetime": None,
        }

    return {
        "datetime_non_null_count": int(len(non_null_series)),
        "min_datetime": non_null_series.min().isoformat(),
        "max_datetime": non_null_series.max().isoformat(),
    }


def build_categorical_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Build categorical baseline statistics for one feature.
    """

    non_null_series = series.dropna()
    row_count = len(series)

    if non_null_series.empty:
        return {
            "categorical_non_null_count": 0,
            "top_value": None,
            "top_value_count": 0,
            "top_value_pct": 0.0,
            "categorical_cardinality": 0,
            "categorical_unique_pct": 0.0,
        }

    value_counts = non_null_series.astype(str).value_counts(dropna=True)

    top_value = str(value_counts.index[0])
    top_value_count = int(value_counts.iloc[0])
    cardinality = int(non_null_series.astype(str).nunique(dropna=True))

    return {
        "categorical_non_null_count": int(len(non_null_series)),
        "top_value": top_value,
        "top_value_count": top_value_count,
        "top_value_pct": top_value_count / row_count if row_count else 0.0,
        "categorical_cardinality": cardinality,
        "categorical_unique_pct": cardinality / row_count if row_count else 0.0,
    }


def build_feature_statistics_record(
    dataframe: pd.DataFrame,
    feature_name: str,
    run_id: str,
    layer_name: str,
    domain_name: str,
    feature_role: str = "feature",
    source_dataset: Optional[str] = None,
    baseline_created_utc: Optional[str] = None,
    baseline_type: str = DEFAULT_BASELINE_TYPE,
    baseline_version: str = DEFAULT_BASELINE_VERSION,
) -> Dict[str, Any]:
    """
    Build one baseline statistics record for one feature.
    """

    if feature_name not in dataframe.columns:
        raise ValueError(f"Feature not found in dataframe: {feature_name}")

    series = dataframe[feature_name]

    row_count = int(len(series))
    null_count = int(series.isna().sum())
    non_null_count = int(row_count - null_count)
    unique_count = int(series.nunique(dropna=True))
    feature_kind = classify_feature_kind(feature_name=feature_name, series=series)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "layer_name": layer_name,
        "domain_name": domain_name,
        "baseline_type": baseline_type,
        "baseline_version": baseline_version,
        "feature_name": feature_name,
        "feature_role": feature_role,
        "feature_kind": feature_kind,
        "source_dataset": source_dataset,
        "dtype": str(series.dtype),
        "row_count": row_count,
        "non_null_count": non_null_count,
        "null_count": null_count,
        "null_pct": null_count / row_count if row_count else 0.0,
        "missing_pct": null_count / row_count if row_count else 0.0,
        "unique_count": unique_count,
        "distinct_count": unique_count,
        "unique_pct": unique_count / row_count if row_count else 0.0,

        # Numeric statistics
        "numeric_non_null_count": None,
        "mean": None,
        "std": None,
        "min": None,
        "p01": None,
        "p05": None,
        "median": None,
        "p95": None,
        "p99": None,
        "max": None,
        "zero_count": None,
        "zero_pct": None,
        "negative_count": None,
        "negative_pct": None,

        # Binary statistics
        "binary_non_null_count": None,
        "positive_count": None,
        "positive_pct": None,
        "negative_binary_count": None,
        "negative_binary_pct": None,

        # Datetime statistics
        "datetime_non_null_count": None,
        "min_datetime": None,
        "max_datetime": None,

        # Categorical statistics
        "categorical_non_null_count": None,
        "top_value": None,
        "top_value_count": None,
        "top_value_pct": None,
        "categorical_cardinality": None,
        "categorical_unique_pct": None,

        "baseline_created_utc": baseline_created_utc or utc_now_iso(),
    }

    if feature_kind == NUMERIC_KIND:
        record.update(build_numeric_statistics(series))

    elif feature_kind == BINARY_KIND:
        record.update(build_binary_statistics(series))
        record.update(build_numeric_statistics(series))

    elif feature_kind == DATETIME_KIND:
        record.update(build_datetime_statistics(series))

    elif feature_kind in {CATEGORICAL_KIND, CONSTANT_KIND, IDENTIFIER_KIND}:
        record.update(build_categorical_statistics(series))

    return record


def infer_source_dataset(feature_name: str) -> Optional[str]:
    """
    Infer source dataset from dataset-prefixed feature names.
    """

    if "__" in feature_name:
        return feature_name.split("__", 1)[0]

    return None


def build_feature_baseline_statistics(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    run_id: str,
    layer_name: str,
    domain_name: str,
    excluded_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build baseline statistics for selected modeling features.
    """

    excluded = set(excluded_columns or [])
    records: List[Dict[str, Any]] = []
    baseline_created_utc = utc_now_iso()

    for feature_name in feature_columns:
        if feature_name in excluded:
            continue

        if feature_name not in dataframe.columns:
            continue

        records.append(
            build_feature_statistics_record(
                dataframe=dataframe,
                feature_name=feature_name,
                run_id=run_id,
                layer_name=layer_name,
                domain_name=domain_name,
                feature_role="feature",
                source_dataset=infer_source_dataset(feature_name),
                baseline_created_utc=baseline_created_utc,
            )
        )

    return pd.DataFrame(records)


def main() -> None:
    """
    Lightweight local validation.
    """

    sample = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4, 5],
            "birth_date": [
                "1980-01-01",
                "1990-05-10",
                "1975-08-21",
                "2000-02-14",
                None,
            ],
            "age": [20, 30, 40, 50, None],
            "has_claim_activity": [1, 0, 1, 1, 0],
            "total_paid_amount": [0.0, 100.0, 200.0, 300.0, -10.0],
            "gender": ["M", "F", "F", None, "M"],
            "line_of_business": ["Medicare", "Commercial", "Medicare", "Medicaid", None],
            "constant_flag": [1, 1, 1, 1, 1],
        }
    )

    output = build_feature_baseline_statistics(
        dataframe=sample,
        feature_columns=[
            "member_id",
            "birth_date",
            "age",
            "has_claim_activity",
            "total_paid_amount",
            "gender",
            "line_of_business",
            "constant_flag",
        ],
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    print("Feature baseline statistics validation successful.")
    print(
        output[
            [
                "feature_name",
                "feature_kind",
                "row_count",
                "null_pct",
                "unique_count",
                "positive_pct",
                "min_datetime",
                "max_datetime",
            ]
        ]
    )
    print(output["feature_kind"].value_counts())


if __name__ == "__main__":
    main()