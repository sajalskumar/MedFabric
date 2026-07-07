###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/targets/leakage_detection.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Release:
#     Release 2D.2 - Target Leakage Hardening
#
# Purpose:
#     Identifies target leakage columns that must be excluded from model
#     training features.
#
# Run:
#     python -m src.modeling.targets.leakage_detection
#
###############################################################################

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional, Set

import pandas as pd


def normalize_target_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize legacy and framework-style target configuration.
    """

    if "target" in model_config and isinstance(model_config["target"], dict):
        return model_config["target"]

    return {
        "strategy": "rule_based",
        "output_column": model_config.get("target_column"),
        "source": {
            "dataset": model_config.get("target_rule", {}).get("source_dataset"),
            "column": model_config.get("target_rule", {}).get("source_column"),
        },
        "parameters": {
            "operator": model_config.get("target_rule", {}).get("operator"),
            "value": model_config.get("target_rule", {}).get("value"),
        },
    }


def get_target_output_column(model_config: Dict[str, Any]) -> str:
    """
    Return target output column for a model.
    """

    target_config = normalize_target_config(model_config)
    target_column = target_config.get("output_column") or target_config.get("column")

    if not target_column:
        raise ValueError("Target output column is missing.")

    return str(target_column)


def get_target_source_dataset(target_config: Dict[str, Any]) -> Optional[str]:
    """
    Return configured target source dataset.
    """

    source = target_config.get("source", {}) or {}

    source_dataset = (
        source.get("dataset")
        or target_config.get("source_dataset")
        or target_config.get("dataset")
    )

    return str(source_dataset) if source_dataset else None


def get_target_source_column(target_config: Dict[str, Any]) -> Optional[str]:
    """
    Return configured target source column.
    """

    strategy = target_config.get("strategy") or target_config.get("mode")

    if strategy == "existing_column":
        column = target_config.get("output_column") or target_config.get("column")
        return str(column) if column else None

    source = target_config.get("source", {}) or {}

    source_column = (
        source.get("column")
        or target_config.get("source_column")
    )

    return str(source_column) if source_column else None


def get_target_source_column_variants(target_config: Dict[str, Any]) -> List[str]:
    """
    Return unprefixed and dataset-prefixed target source column names.
    """

    source_dataset = get_target_source_dataset(target_config)
    source_column = get_target_source_column(target_config)

    columns: List[str] = []

    if source_column:
        columns.append(source_column)

    if source_dataset and source_column:
        columns.append(f"{source_dataset}__{source_column}")

    return columns

def get_weighted_score_component_variants(target_config: Dict[str, Any]) -> List[str]:
    """
    Return leakage column variants for all weighted_score components.
    """

    if (target_config.get("strategy") or target_config.get("mode")) != "weighted_score":
        return []

    columns: List[str] = []

    for component in target_config.get("components", []) or []:
        dataset = component.get("dataset")
        column = component.get("column")

        if column:
            columns.append(str(column))

        if dataset and column:
            columns.append(f"{dataset}__{column}")

    return columns


def get_target_source_datasets(target_config: Dict[str, Any]) -> List[str]:
    """
    Return all source datasets used by a target.
    """

    if (target_config.get("strategy") or target_config.get("mode")) == "weighted_score":
        return sorted(
            set(
                str(component.get("dataset"))
                for component in target_config.get("components", []) or []
                if component.get("dataset")
            )
        )

    source_dataset = get_target_source_dataset(target_config)
    return [source_dataset] if source_dataset else []

def get_leakage_columns_for_model(model_config: Dict[str, Any]) -> List[str]:
    """
    Return direct leakage columns for one model.
    """

    target_config = normalize_target_config(model_config)

    columns: List[str] = []
    columns.extend(get_target_source_column_variants(target_config))
    columns.extend(get_weighted_score_component_variants(target_config))

    return sorted(set(column for column in columns if column))


def get_leakage_columns_for_models(models_config: Dict[str, Any]) -> List[str]:
    """
    Collect direct target source columns for all enabled models.

    This function is retained for backward compatibility.
    Release 2D.2 should prefer get_model_specific_safe_feature_columns().
    """

    leakage_columns: List[str] = []

    for model_config in models_config.values():
        if not bool(model_config.get("enabled", True)):
            continue

        leakage_columns.extend(get_leakage_columns_for_model(model_config))

    return sorted(set(column for column in leakage_columns if column))


def get_dataset_prefixed_columns(
    dataframe: pd.DataFrame,
    dataset_name: Optional[str],
) -> List[str]:
    """
    Return columns that came from a specific source dataset prefix.
    """

    if not dataset_name:
        return []

    prefix = f"{dataset_name}__"

    return [
        column
        for column in dataframe.columns
        if str(column).startswith(prefix)
    ]


def get_name_pattern_leakage_columns(
    dataframe: pd.DataFrame,
    leakage_config: Dict[str, Any],
) -> List[str]:
    """
    Return YAML pattern-based leakage columns.
    """

    if not bool(leakage_config.get("enabled", False)):
        return []

    exclude_patterns = leakage_config.get("exclude_patterns", []) or []

    matched: Set[str] = set()

    for column in dataframe.columns:
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(str(column).lower(), str(pattern).lower()):
                matched.add(column)

    return sorted(matched)


def get_additional_leakage_columns(
    leakage_config: Dict[str, Any],
) -> List[str]:
    """
    Return YAML-configured additional leakage columns.
    """

    if not bool(leakage_config.get("enabled", False)):
        return []

    return sorted(set(leakage_config.get("additional_columns", []) or []))

def get_source_domain_proxy_columns(
    dataframe: pd.DataFrame,
    source_dataset: Optional[str],
    leakage_config: Dict[str, Any],
) -> List[str]:
    """
    Return source-domain proxy columns for a model target source dataset.
    """

    if not source_dataset:
        return []

    proxy_rules = leakage_config.get("source_domain_proxy_patterns", {}) or {}
    patterns = proxy_rules.get(source_dataset, []) or []

    matched: Set[str] = set()

    for column in dataframe.columns:
        for pattern in patterns:
            if fnmatch.fnmatch(str(column).lower(), str(pattern).lower()):
                matched.add(column)

    return sorted(matched)

def get_high_correlation_leakage_columns(
    dataframe: pd.DataFrame,
    candidate_feature_columns: List[str],
    target_column: str,
    threshold: float = 0.995,
) -> List[str]:
    """
    Detect numeric features that are almost perfectly correlated with the target.
    """

    if target_column not in dataframe.columns:
        return []

    target = pd.to_numeric(dataframe[target_column], errors="coerce")

    if target.nunique(dropna=True) < 2:
        return []

    leakage_columns: List[str] = []

    for column in candidate_feature_columns:
        if column not in dataframe.columns:
            continue

        series = pd.to_numeric(dataframe[column], errors="coerce")

        if series.nunique(dropna=True) < 2:
            continue

        try:
            corr = series.corr(target)
        except Exception:
            corr = None

        if corr is not None and pd.notna(corr) and abs(float(corr)) >= threshold:
            leakage_columns.append(column)

    return sorted(set(leakage_columns))


def build_model_leakage_report(
    dataframe: pd.DataFrame,
    model_key: str,
    model_config: Dict[str, Any],
    member_key: str,
    target_columns: List[str],
    leakage_config: Dict[str, Any],
    correlation_threshold: float = 0.995,
) -> pd.DataFrame:
    """
    Build model-specific leakage report.
    """

    target_config = normalize_target_config(model_config)
    target_column = get_target_output_column(model_config)
    source_datasets = get_target_source_datasets(target_config)

    excluded: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    def add(column: str, reason: str) -> None:
        if column and column in dataframe.columns and column not in excluded:
            excluded.add(column)
            rows.append(
                {
                    "model_key": model_key,
                    "target_column": target_column,
                    "excluded_column": column,
                    "exclusion_reason": reason,
                }
            )

    add(member_key, "member_key")
    add("modeling_layer_run_id", "runtime_metadata")
    add("modeling_layer_built_at_utc", "runtime_metadata")

    for column in target_columns:
        add(column, "generated_target_column")

    for column in get_leakage_columns_for_model(model_config):
        add(column, "direct_target_source_column")

    for column in get_additional_leakage_columns(leakage_config):
        add(column, "yaml_additional_leakage_column")

    for column in get_name_pattern_leakage_columns(dataframe, leakage_config):
        add(column, "yaml_pattern_leakage_column")

    for source_dataset in source_datasets:
        if bool(leakage_config.get("exclude_target_source_dataset", True)):
            for column in get_dataset_prefixed_columns(dataframe, source_dataset):
                add(column, "target_source_dataset_proxy_column")

        for column in get_source_domain_proxy_columns(
            dataframe=dataframe,
            source_dataset=source_dataset,
            leakage_config=leakage_config,
        ):
            add(column, "source_domain_proxy_pattern")
    current_candidate_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    if bool(leakage_config.get("correlation_check_enabled", True)):
        threshold = float(
            leakage_config.get("correlation_threshold", correlation_threshold)
        )

        for column in get_high_correlation_leakage_columns(
            dataframe=dataframe,
            candidate_feature_columns=current_candidate_columns,
            target_column=target_column,
            threshold=threshold,
        ):
            add(column, "high_correlation_with_target")

    return pd.DataFrame(rows)


def get_model_specific_safe_feature_columns(
    dataframe: pd.DataFrame,
    model_key: str,
    model_config: Dict[str, Any],
    member_key: str,
    target_columns: List[str],
    leakage_config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return safe training feature columns for one model.
    """

    leakage_config = leakage_config or {}

    report = build_model_leakage_report(
        dataframe=dataframe,
        model_key=model_key,
        model_config=model_config,
        member_key=member_key,
        target_columns=target_columns,
        leakage_config=leakage_config,
    )

    excluded = set(report["excluded_column"].tolist()) if not report.empty else set()

    return [
        column
        for column in dataframe.columns
        if column not in excluded
    ]


def build_target_leakage_report_for_models(
    dataframe: pd.DataFrame,
    models_config: Dict[str, Any],
    member_key: str,
    target_columns: List[str],
    leakage_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build leakage report for all enabled models.
    """

    leakage_config = leakage_config or {}

    reports: List[pd.DataFrame] = []

    for model_key, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            continue

        report = build_model_leakage_report(
            dataframe=dataframe,
            model_key=model_key,
            model_config=model_config,
            member_key=member_key,
            target_columns=target_columns,
            leakage_config=leakage_config,
        )

        reports.append(report)

    if not reports:
        return pd.DataFrame(
            columns=[
                "model_key",
                "target_column",
                "excluded_column",
                "exclusion_reason",
            ]
        )

    return pd.concat(reports, ignore_index=True)


def main() -> None:
    """
    Lightweight module validation.
    """

    sample_frame = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4, 5],
            "total_paid_amount": [100, 200, 300, 400, 500],
            "cost_features__total_paid_amount": [100, 200, 300, 400, 500],
            "cost_features__allowed_amount": [120, 220, 320, 420, 520],
            "age": [30, 40, 50, 60, 70],
            "high_cost_target": [0, 0, 0, 1, 1],
        }
    )

    sample_model = {
        "enabled": True,
        "target": {
            "strategy": "quantile",
            "output_column": "high_cost_target",
            "source": {
                "dataset": "cost_features",
                "column": "total_paid_amount",
            },
            "parameters": {
                "quantile": 0.80,
            },
        },
    }

    report = build_model_leakage_report(
        dataframe=sample_frame,
        model_key="high_cost",
        model_config=sample_model,
        member_key="member_id",
        target_columns=["high_cost_target"],
        leakage_config={
            "enabled": True,
            "exclude_target_source_dataset": True,
            "correlation_check_enabled": True,
            "correlation_threshold": 0.995,
            "exclude_patterns": ["*paid_amount*"],
            "additional_columns": [],
        },
    )

    safe_features = get_model_specific_safe_feature_columns(
        dataframe=sample_frame,
        model_key="high_cost",
        model_config=sample_model,
        member_key="member_id",
        target_columns=["high_cost_target"],
        leakage_config={
            "enabled": True,
            "exclude_target_source_dataset": True,
            "correlation_check_enabled": True,
            "correlation_threshold": 0.995,
            "exclude_patterns": ["*paid_amount*"],
            "additional_columns": [],
        },
    )

    print("Leakage detection validation successful.")
    print(report)
    print(f"Safe features: {safe_features}")


if __name__ == "__main__":
    main()