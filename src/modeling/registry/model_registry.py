###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/registry/model_registry.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds standardized model registry records for MedFabric champion models.
#
# Run:
#     python -m src.modeling.registry.model_registry
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as ISO string.
    """

    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelRegistryRecord:
    """
    Standard champion model registry record.
    """

    run_id: str
    layer_name: str
    domain_name: str
    model_key: str
    model_name: str
    model_type: str
    target_column: str
    champion_algorithm_key: str
    champion_algorithm_name: str
    selection_metric: str
    selection_metric_value: Optional[float]
    model_path: str
    scoring_path: str
    metrics_path: str
    feature_importance_path: str
    scored_row_count: int
    status: str
    metric_summary_json: str
    event_timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry record to dictionary.
        """

        return {
            "run_id": self.run_id,
            "layer_name": self.layer_name,
            "domain_name": self.domain_name,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "target_column": self.target_column,
            "champion_algorithm_key": self.champion_algorithm_key,
            "champion_algorithm_name": self.champion_algorithm_name,
            "selection_metric": self.selection_metric,
            "selection_metric_value": self.selection_metric_value,
            "model_path": self.model_path,
            "scoring_path": self.scoring_path,
            "metrics_path": self.metrics_path,
            "feature_importance_path": self.feature_importance_path,
            "scored_row_count": self.scored_row_count,
            "status": self.status,
            "metric_summary_json": self.metric_summary_json,
            "event_timestamp_utc": self.event_timestamp_utc,
        }


def build_model_registry_record(
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    model_type: str,
    target_column: str,
    champion_algorithm_key: str,
    champion_algorithm_name: str,
    selection_metric: str,
    champion_metrics: Dict[str, Any],
    model_path: str,
    scoring_path: str,
    metrics_path: str,
    feature_importance_path: str,
    scored_row_count: int,
    status: str = STATUS_SUCCESS,
) -> Dict[str, Any]:
    """
    Build one standardized model registry record.
    """

    selection_metric_value = champion_metrics.get(selection_metric)

    record = ModelRegistryRecord(
        run_id=run_id,
        layer_name=layer_name,
        domain_name=domain_name,
        model_key=model_key,
        model_name=model_name,
        model_type=model_type,
        target_column=target_column,
        champion_algorithm_key=champion_algorithm_key,
        champion_algorithm_name=champion_algorithm_name,
        selection_metric=selection_metric,
        selection_metric_value=selection_metric_value,
        model_path=model_path,
        scoring_path=scoring_path,
        metrics_path=metrics_path,
        feature_importance_path=feature_importance_path,
        scored_row_count=scored_row_count,
        status=status,
        metric_summary_json=str(champion_metrics),
        event_timestamp_utc=utc_now_iso(),
    )

    return record.to_dict()


def build_model_registry_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build model registry dataframe from records.
    """

    return pd.DataFrame(records)


def main() -> None:
    """
    Lightweight module validation.
    """

    record = build_model_registry_record(
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        model_type="classification",
        target_column="high_cost_target",
        champion_algorithm_key="random_forest",
        champion_algorithm_name="Random Forest",
        selection_metric="roc_auc",
        champion_metrics={
            "accuracy": 0.91,
            "precision": 0.84,
            "recall": 0.77,
            "f1": 0.80,
            "roc_auc": 0.93,
        },
        model_path="models/high_cost/high_cost_model.pkl",
        scoring_path="data/scoring/high_cost_scores.parquet",
        metrics_path="models/high_cost/high_cost_model_metrics.parquet",
        feature_importance_path="models/high_cost/high_cost_feature_importance.parquet",
        scored_row_count=100000,
    )

    df = build_model_registry_dataframe([record])

    print("Model registry validation successful.")
    print(df)


if __name__ == "__main__":
    main()