###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/targets/target_strategies.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Implements reusable target-generation strategies for MedFabric models.
#
#     This module is responsible only for converting configured target
#     definitions into binary target labels.
#
# Supported Strategies:
#     - threshold
#     - rule_based
#     - quantile
#     - existing_column
#
# Run:
#     python -m src.modeling.targets.target_strategies
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


SUPPORTED_TARGET_STRATEGIES = {
    "threshold",
    "rule_based",
    "quantile",
    "existing_column",
}


@dataclass
class TargetStrategyResult:
    """
    Standard output returned by every target strategy.
    """

    target_column: str
    target_series: pd.Series
    strategy: str
    source_column: Optional[str]
    resolved_source_column: Optional[str]
    threshold_value: Optional[float]
    positive_count: int
    negative_count: int
    positive_rate: float


def apply_operator(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Apply a configured comparison operator to a pandas Series.
    """

    if operator == "equals":
        return series == value

    if operator == "not_equals":
        return series != value

    if operator == "greater_than":
        return series > value

    if operator == "greater_than_or_equal":
        return series >= value

    if operator == "less_than":
        return series < value

    if operator == "less_than_or_equal":
        return series <= value

    if operator == "in":
        return series.isin(value)

    if operator == "not_in":
        return ~series.isin(value)

    if operator == "is_null":
        return series.isna()

    if operator == "is_not_null":
        return series.notna()

    raise ValueError(f"Unsupported target operator: {operator}")


def resolve_source_column(
    dataframe: pd.DataFrame,
    source_dataset: Optional[str],
    source_column: str,
) -> str:
    """
    Resolve source column from the modeling feature matrix.

    The feature matrix may contain either:
        - original column name
        - dataset-prefixed column name, such as cost_features__total_paid_amount
    """

    if source_column in dataframe.columns:
        return source_column

    if source_dataset:
        prefixed_column = f"{source_dataset}__{source_column}"

        if prefixed_column in dataframe.columns:
            return prefixed_column

    raise ValueError(
        "Unable to resolve target source column. "
        f"source_dataset={source_dataset}, source_column={source_column}"
    )


def build_result(
    target_column: str,
    target_series: pd.Series,
    strategy: str,
    source_column: Optional[str],
    resolved_source_column: Optional[str],
    threshold_value: Optional[float],
) -> TargetStrategyResult:
    """
    Build standardized target strategy result.
    """

    clean_target = target_series.astype(int)

    total_count = int(len(clean_target))
    positive_count = int((clean_target == 1).sum())
    negative_count = int((clean_target == 0).sum())
    positive_rate = positive_count / total_count if total_count else 0.0

    return TargetStrategyResult(
        target_column=target_column,
        target_series=clean_target,
        strategy=strategy,
        source_column=source_column,
        resolved_source_column=resolved_source_column,
        threshold_value=threshold_value,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=positive_rate,
    )


def build_existing_column_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> TargetStrategyResult:
    """
    Use an existing binary target column from the modeling feature matrix.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Existing target column not found: {target_column}")

    return build_result(
        target_column=target_column,
        target_series=dataframe[target_column],
        strategy="existing_column",
        source_column=target_column,
        resolved_source_column=target_column,
        threshold_value=None,
    )


def build_threshold_target(
    dataframe: pd.DataFrame,
    target_column: str,
    source_dataset: Optional[str],
    source_column: str,
    operator: str,
    value: Any,
) -> TargetStrategyResult:
    """
    Build binary target using a configured threshold or comparison rule.
    """

    resolved_column = resolve_source_column(
        dataframe=dataframe,
        source_dataset=source_dataset,
        source_column=source_column,
    )

    mask = apply_operator(
        series=dataframe[resolved_column],
        operator=operator,
        value=value,
    )

    return build_result(
        target_column=target_column,
        target_series=mask.astype(int),
        strategy="threshold",
        source_column=source_column,
        resolved_source_column=resolved_column,
        threshold_value=float(value) if isinstance(value, (int, float)) else None,
    )


def build_quantile_target(
    dataframe: pd.DataFrame,
    target_column: str,
    source_dataset: Optional[str],
    source_column: str,
    quantile: float,
    positive_direction: str = "greater_than_or_equal",
) -> TargetStrategyResult:
    """
    Build binary target using a dynamic quantile threshold.

    Example:
        quantile = 0.90 means top 10 percent becomes target = 1.
    """

    resolved_column = resolve_source_column(
        dataframe=dataframe,
        source_dataset=source_dataset,
        source_column=source_column,
    )

    threshold_value = float(dataframe[resolved_column].quantile(float(quantile)))

    mask = apply_operator(
        series=dataframe[resolved_column],
        operator=positive_direction,
        value=threshold_value,
    )

    return build_result(
        target_column=target_column,
        target_series=mask.astype(int),
        strategy="quantile",
        source_column=source_column,
        resolved_source_column=resolved_column,
        threshold_value=threshold_value,
    )


def build_target_from_config(
    dataframe: pd.DataFrame,
    model_key: str,
    target_config: Dict[str, Any],
) -> TargetStrategyResult:
    """
    Build target for one model using normalized target configuration.
    """

    strategy = target_config.get("strategy") or target_config.get("mode")
    target_column = target_config.get("output_column") or target_config.get("column")

    if not strategy:
        raise ValueError(f"Target strategy missing for model: {model_key}")

    if not target_column:
        raise ValueError(f"Target output column missing for model: {model_key}")

    if strategy not in SUPPORTED_TARGET_STRATEGIES:
        raise ValueError(
            f"Unsupported target strategy for model {model_key}: {strategy}. "
            f"Supported: {sorted(SUPPORTED_TARGET_STRATEGIES)}"
        )

    if strategy == "existing_column":
        return build_existing_column_target(
            dataframe=dataframe,
            target_column=target_column,
        )

    source = target_config.get("source", {})
    parameters = target_config.get("parameters", {})

    source_dataset = (
        source.get("dataset")
        or target_config.get("source_dataset")
        or target_config.get("dataset")
    )
    source_column = (
        source.get("column")
        or target_config.get("source_column")
    )

    if not source_column:
        raise ValueError(f"Target source column missing for model: {model_key}")

    if strategy in {"threshold", "rule_based"}:
        operator = parameters.get("operator") or target_config.get("operator")
        value = parameters.get("value", target_config.get("value"))

        if not operator:
            raise ValueError(f"Target operator missing for model: {model_key}")

        return build_threshold_target(
            dataframe=dataframe,
            target_column=target_column,
            source_dataset=source_dataset,
            source_column=source_column,
            operator=operator,
            value=value,
        )

    if strategy == "quantile":
        quantile = parameters.get("quantile", target_config.get("quantile"))
        positive_direction = parameters.get(
            "positive_direction",
            target_config.get("positive_direction", "greater_than_or_equal"),
        )

        if quantile is None:
            raise ValueError(f"Target quantile missing for model: {model_key}")

        return build_quantile_target(
            dataframe=dataframe,
            target_column=target_column,
            source_dataset=source_dataset,
            source_column=source_column,
            quantile=float(quantile),
            positive_direction=positive_direction,
        )

    raise ValueError(f"Unhandled target strategy: {strategy}")


def main() -> None:
    """
    Lightweight module validation.
    """

    sample = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4, 5],
            "total_paid_amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    config = {
        "strategy": "quantile",
        "output_column": "high_cost_target",
        "source": {
            "dataset": "cost_features",
            "column": "total_paid_amount",
        },
        "parameters": {
            "quantile": 0.80,
            "positive_direction": "greater_than_or_equal",
        },
    }

    result = build_target_from_config(
        dataframe=sample,
        model_key="high_cost",
        target_config=config,
    )

    print("Target strategy validation successful.")
    print(f"Target column: {result.target_column}")
    print(f"Strategy: {result.strategy}")
    print(f"Threshold: {result.threshold_value}")
    print(f"Positive count: {result.positive_count}")
    print(f"Negative count: {result.negative_count}")
    print(f"Positive rate: {result.positive_rate:.4f}")


if __name__ == "__main__":
    main()