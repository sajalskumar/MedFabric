###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/targets/target_builder.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Orchestrates target generation for MedFabric predictive models.
#
#     This module does not implement target strategies directly. It delegates
#     strategy execution to:
#
#         src/modeling/targets/target_strategies.py
#
#     It also remains backward compatible with the current modeling.yaml style:
#
#         target_column:
#         target_rule:
#
#     while supporting the new framework style:
#
#         target:
#           strategy:
#           output_column:
#           source:
#           parameters:
#
# Run:
#     python -m src.modeling.targets.target_builder
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from src.modeling.targets.leakage_detection import (
    get_leakage_columns_for_models,
    get_target_output_column,
    normalize_target_config,
)
from src.modeling.targets.target_strategies import (
    TargetStrategyResult,
    build_target_from_config,
)


STATUS_SUCCESS = "SUCCESS"
STATUS_WARNING = "WARNING"
STATUS_FAILED = "FAILED"


@dataclass
class TargetBuildResult:
    """
    Standard Target Builder output.
    """

    modeling_frame: pd.DataFrame
    target_columns: List[str]
    target_summary: pd.DataFrame
    leakage_columns: List[str]


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as ISO string.
    """

    return datetime.now(timezone.utc).isoformat()


def build_target_summary_row(
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    result: TargetStrategyResult,
) -> Dict[str, Any]:
    """
    Build one target summary row from a strategy result.
    """

    status = (
        STATUS_SUCCESS
        if result.positive_count > 0 and result.negative_count > 0
        else STATUS_WARNING
    )

    return {
        "run_id": run_id,
        "layer_name": layer_name,
        "domain_name": domain_name,
        "model_key": model_key,
        "model_name": model_name,
        "target_column": result.target_column,
        "target_strategy": result.strategy,
        "source_column": result.source_column,
        "resolved_source_column": result.resolved_source_column,
        "threshold_value": result.threshold_value,
        "positive_count": result.positive_count,
        "negative_count": result.negative_count,
        "positive_rate": result.positive_rate,
        "status": status,
        "event_timestamp_utc": utc_now_iso(),
    }


def build_targets_for_enabled_models(
    feature_matrix: pd.DataFrame,
    models_config: Dict[str, Any],
    run_id: str,
    layer_name: str,
    domain_name: str,
) -> TargetBuildResult:
    """
    Generate targets for all enabled models.

    Parameters
    ----------
    feature_matrix:
        Unified modeling feature matrix.

    models_config:
        The `models` section from config/modeling/modeling.yaml.

    run_id:
        Current modeling run identifier.

    layer_name:
        Modeling layer name.

    domain_name:
        Modeling domain name.

    Returns
    -------
    TargetBuildResult
        Modeling frame with target columns, target column list, target summary,
        and leakage columns to exclude from training.
    """

    modeling_frame = feature_matrix.copy()
    target_columns: List[str] = []
    summary_rows: List[Dict[str, Any]] = []

    for model_key, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            continue

        target_config = normalize_target_config(model_config)

        result = build_target_from_config(
            dataframe=modeling_frame,
            model_key=model_key,
            target_config=target_config,
        )

        modeling_frame[result.target_column] = result.target_series
        target_columns.append(result.target_column)

        summary_rows.append(
            build_target_summary_row(
                run_id=run_id,
                layer_name=layer_name,
                domain_name=domain_name,
                model_key=model_key,
                model_name=model_config.get("model_name", model_key),
                result=result,
            )
        )

    leakage_columns = get_leakage_columns_for_models(models_config)

    target_summary = pd.DataFrame(summary_rows)

    return TargetBuildResult(
        modeling_frame=modeling_frame,
        target_columns=target_columns,
        target_summary=target_summary,
        leakage_columns=leakage_columns,
    )


def get_enabled_model_target_columns(models_config: Dict[str, Any]) -> List[str]:
    """
    Return target output columns for enabled models.
    """

    target_columns: List[str] = []

    for _, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            continue

        target_columns.append(get_target_output_column(model_config))

    return target_columns


def main() -> None:
    """
    Lightweight module validation.
    """

    sample_feature_matrix = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4, 5],
            "total_paid_amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            "total_claims": [1, 3, 5, 7, 9],
        }
    )

    sample_models_config = {
        "high_cost": {
            "enabled": True,
            "model_name": "High Cost Model",
            "target": {
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
            },
        },
        "readmission": {
            "enabled": True,
            "model_name": "Readmission Model",
            "target_column": "readmission_target",
            "target_rule": {
                "source_dataset": "utilization_features",
                "source_column": "total_claims",
                "operator": "greater_than_or_equal",
                "value": 7,
            },
        },
    }

    result = build_targets_for_enabled_models(
        feature_matrix=sample_feature_matrix,
        models_config=sample_models_config,
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    print("Target Builder validation successful.")
    print(f"Target columns: {result.target_columns}")
    print(f"Leakage columns: {result.leakage_columns}")
    print(result.target_summary)


if __name__ == "__main__":
    main()