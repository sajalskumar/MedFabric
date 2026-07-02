###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/training/trainer.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Trains multiple enabled algorithms for a model, evaluates each candidate,
#     and selects the best-performing champion model.
#
# Run:
#     python -m src.modeling.training.trainer
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modeling.training.algorithms import (
    AlgorithmDefinition,
    build_algorithm_definitions,
    get_default_algorithms_config,
)
from src.modeling.training.preprocessing import (
    build_preprocessor,
    prepare_features_for_preprocessing,
)


STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"


@dataclass
class CandidateModelResult:
    model_key: str
    algorithm_key: str
    algorithm_name: str
    status: str
    pipeline: Optional[Pipeline]
    metrics: Dict[str, Any]
    error_message: Optional[str]
    trained_at_utc: str


@dataclass
class TrainingResult:
    model_key: str
    target_column: str
    champion_algorithm_key: str
    champion_algorithm_name: str
    champion_pipeline: Pipeline
    champion_metrics: Dict[str, Any]
    candidate_results: List[CandidateModelResult]
    metrics_dataframe: pd.DataFrame


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def get_prediction_scores(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]

    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))

    return pipeline.predict(X)


def train_candidate_algorithm(
    model_key: str,
    algorithm_definition: AlgorithmDefinition,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_config: Dict[str, Any],
) -> CandidateModelResult:
    try:
        preprocessor = build_preprocessor(
            dataframe=X_train,
            preprocessing_config=preprocessing_config,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", algorithm_definition.estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = get_prediction_scores(pipeline, X_test)

        metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
        )

        return CandidateModelResult(
            model_key=model_key,
            algorithm_key=algorithm_definition.algorithm_key,
            algorithm_name=algorithm_definition.algorithm_name,
            status=STATUS_SUCCESS,
            pipeline=pipeline,
            metrics=metrics,
            error_message=None,
            trained_at_utc=utc_now_iso(),
        )

    except Exception as exc:
        return CandidateModelResult(
            model_key=model_key,
            algorithm_key=algorithm_definition.algorithm_key,
            algorithm_name=algorithm_definition.algorithm_name,
            status=STATUS_FAILED,
            pipeline=None,
            metrics={},
            error_message=str(exc),
            trained_at_utc=utc_now_iso(),
        )


def select_champion_model(
    candidate_results: List[CandidateModelResult],
    selection_metric: str,
) -> CandidateModelResult:
    successful = [
        result
        for result in candidate_results
        if result.status == STATUS_SUCCESS
        and result.pipeline is not None
        and result.metrics.get(selection_metric) is not None
    ]

    if not successful:
        raise ValueError(
            f"No successful candidate model available for metric: {selection_metric}"
        )

    return sorted(
        successful,
        key=lambda result: float(result.metrics.get(selection_metric, -1)),
        reverse=True,
    )[0]


def build_candidate_metrics_dataframe(
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    target_column: str,
    champion_algorithm_key: str,
    candidate_results: List[CandidateModelResult],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for result in candidate_results:
        if result.status == STATUS_SUCCESS:
            for metric_name, metric_value in result.metrics.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "layer_name": layer_name,
                        "domain_name": domain_name,
                        "model_key": model_key,
                        "model_name": model_name,
                        "target_column": target_column,
                        "algorithm_key": result.algorithm_key,
                        "algorithm_name": result.algorithm_name,
                        "is_champion": result.algorithm_key == champion_algorithm_key,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "status": result.status,
                        "error_message": result.error_message,
                        "event_timestamp_utc": result.trained_at_utc,
                    }
                )
        else:
            rows.append(
                {
                    "run_id": run_id,
                    "layer_name": layer_name,
                    "domain_name": domain_name,
                    "model_key": model_key,
                    "model_name": model_name,
                    "target_column": target_column,
                    "algorithm_key": result.algorithm_key,
                    "algorithm_name": result.algorithm_name,
                    "is_champion": False,
                    "metric_name": None,
                    "metric_value": None,
                    "status": result.status,
                    "error_message": result.error_message,
                    "event_timestamp_utc": result.trained_at_utc,
                }
            )

    return pd.DataFrame(rows)


def train_model_candidates(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    model_key: str,
    model_name: str,
    modeling_defaults: Dict[str, Any],
    algorithms_config: Optional[Dict[str, Any]],
    run_id: str = "UNKNOWN_RUN",
    layer_name: str = "Layer 2D - Enterprise Modeling Framework",
    domain_name: str = "Modeling",
) -> TrainingResult:
    X = dataframe[feature_columns].copy()
    y = dataframe[target_column].astype(int)

    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"Target has fewer than two classes. model={model_key}, target={target_column}"
        )

    X = prepare_features_for_preprocessing(X)

    random_state = int(modeling_defaults.get("random_state", 42))
    test_size = float(modeling_defaults.get("test_size", 0.20))
    selection_metric = modeling_defaults.get("selection_metric", "roc_auc")
    preprocessing_config = modeling_defaults.get("preprocessing", {})

    active_algorithms_config = algorithms_config or get_default_algorithms_config()

    algorithm_definitions = build_algorithm_definitions(
        algorithms_config=active_algorithms_config,
        random_state=random_state,
    )

    if not algorithm_definitions:
        raise ValueError("No enabled algorithms found for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    candidate_results: List[CandidateModelResult] = []

    for algorithm_definition in algorithm_definitions.values():
        candidate_results.append(
            train_candidate_algorithm(
                model_key=model_key,
                algorithm_definition=algorithm_definition,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                preprocessing_config=preprocessing_config,
            )
        )

    champion = select_champion_model(
        candidate_results=candidate_results,
        selection_metric=selection_metric,
    )

    metrics_dataframe = build_candidate_metrics_dataframe(
        run_id=run_id,
        layer_name=layer_name,
        domain_name=domain_name,
        model_key=model_key,
        model_name=model_name,
        target_column=target_column,
        champion_algorithm_key=champion.algorithm_key,
        candidate_results=candidate_results,
    )

    return TrainingResult(
        model_key=model_key,
        target_column=target_column,
        champion_algorithm_key=champion.algorithm_key,
        champion_algorithm_name=champion.algorithm_name,
        champion_pipeline=champion.pipeline,
        champion_metrics=champion.metrics,
        candidate_results=candidate_results,
        metrics_dataframe=metrics_dataframe,
    )


def main() -> None:
    dataframe = pd.DataFrame(
        {
            "member_id": range(1, 101),
            "age": list(range(20, 120)),
            "cost": list(range(100, 10100, 100)),
            "gender": ["M", "F"] * 50,
        }
    )

    dataframe["target"] = (dataframe["cost"] >= dataframe["cost"].quantile(0.80)).astype(int)

    result = train_model_candidates(
        dataframe=dataframe,
        feature_columns=["age", "gender"],
        target_column="target",
        model_key="high_cost",
        model_name="High Cost Model",
        modeling_defaults={
            "random_state": 42,
            "test_size": 0.20,
            "selection_metric": "roc_auc",
            "preprocessing": {
                "numeric_imputation_strategy": "median",
                "categorical_imputation_strategy": "most_frequent",
                "scale_numeric_features": False,
                "one_hot_encode_categorical_features": True,
            },
        },
        algorithms_config=get_default_algorithms_config(),
        run_id="TEST_RUN",
    )

    print("Trainer validation successful.")
    print(f"Champion algorithm: {result.champion_algorithm_key}")
    print(f"Champion metrics: {result.champion_metrics}")
    print(result.metrics_dataframe.head())


if __name__ == "__main__":
    main()