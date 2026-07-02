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
# Purpose:
#     Identifies target leakage columns that must be excluded from model
#     training features.
#
# Run:
#     python -m src.modeling.targets.leakage_detection
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Optional


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

    return target_column


def get_target_source_column_variants(target_config: Dict[str, Any]) -> List[str]:
    """
    Return unprefixed and dataset-prefixed target source column names.
    """

    strategy = target_config.get("strategy") or target_config.get("mode")

    if strategy == "existing_column":
        column = target_config.get("output_column") or target_config.get("column")
        return [column] if column else []

    source = target_config.get("source", {}) or {}

    source_dataset: Optional[str] = (
        source.get("dataset")
        or target_config.get("source_dataset")
        or target_config.get("dataset")
    )

    source_column: Optional[str] = (
        source.get("column")
        or target_config.get("source_column")
    )

    columns: List[str] = []

    if source_column:
        columns.append(source_column)

    if source_dataset and source_column:
        columns.append(f"{source_dataset}__{source_column}")

    return columns


def get_leakage_columns_for_models(models_config: Dict[str, Any]) -> List[str]:
    """
    Collect all target source columns that should be excluded from training.
    """

    leakage_columns: List[str] = []

    for model_config in models_config.values():
        if not bool(model_config.get("enabled", True)):
            continue

        target_config = normalize_target_config(model_config)
        leakage_columns.extend(get_target_source_column_variants(target_config))

    return sorted(set(column for column in leakage_columns if column))


def main() -> None:
    """
    Lightweight module validation.
    """

    sample_models = {
        "high_cost": {
            "enabled": True,
            "target": {
                "strategy": "quantile",
                "output_column": "high_cost_target",
                "source": {
                    "dataset": "cost_features",
                    "column": "total_paid_amount",
                },
                "parameters": {
                    "quantile": 0.90,
                },
            },
        },
        "legacy_readmission": {
            "enabled": True,
            "target_column": "readmission_target",
            "target_rule": {
                "source_dataset": "utilization_features",
                "source_column": "total_claims",
                "operator": "greater_than_or_equal",
                "value": 13,
            },
        },
    }

    leakage_columns = get_leakage_columns_for_models(sample_models)

    print("Leakage detection validation successful.")
    print(f"Leakage columns: {leakage_columns}")


if __name__ == "__main__":
    main()