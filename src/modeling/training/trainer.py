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
#     Trains enabled candidate algorithms for one prediction objective, evaluates
#     each candidate, and selects the champion model.
#
# Enhancements:
#     - Supports YAML-driven class imbalance handling.
#     - Supports YAML-driven threshold optimization.
#     - Stores optimized threshold in candidate/champion metrics.
#     - Attaches selected threshold to champion pipeline for scoring.
#
# Run:
#     python -m src.modeling.training.trainer
#
###############################################################################

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.common.parallel_utils import (
    raise_if_any_task_failed,
    resolve_parallelism_config,
    run_tasks,
)
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
    """
    Result for one candidate algorithm.
    """

    model_key: str
    algorithm_key: str
    algorithm_name: str
    status: str
    pipeline: Optional[Pipeline]
    metrics: Dict[str, Any]
    error_message: Optional[str]
    trained_at_utc: str
    training_seconds: float
    train_row_count: int
    test_row_count: int


@dataclass
class TrainingResult:
    """
    Final training result for one prediction objective.
    """

    model_key: str
    target_column: str
    champion_algorithm_key: str
    champion_algorithm_name: str
    champion_pipeline: Pipeline
    champion_metrics: Dict[str, Any]
    candidate_results: List[CandidateModelResult]
    metrics_dataframe: pd.DataFrame
    champion_summary_dataframe: pd.DataFrame
    training_row_count: int
    full_row_count: int
    sampling_applied: bool


###############################################################################
# Configuration Helpers
###############################################################################

def get_training_performance_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return training.performance from modeling.yaml.
    """

    return training_config.get("performance", {}) or {}


def get_training_metrics_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return training.metrics from modeling.yaml.
    """

    return training_config.get("metrics", {}) or {}


def get_imbalance_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return training.imbalance from modeling.yaml.
    """

    return training_config.get("imbalance", {}) or {}


def get_threshold_optimization_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return training.threshold_optimization from modeling.yaml.
    """

    return training_config.get("threshold_optimization", {}) or {}


def get_selection_metric(
    modeling_defaults: Dict[str, Any],
    training_config: Dict[str, Any],
) -> str:
    """
    Resolve champion selection metric.
    """

    metrics_config = get_training_metrics_config(training_config)

    return (
        metrics_config.get("primary_metric")
        or modeling_defaults.get("selection_metric")
        or "roc_auc"
    )


###############################################################################
# Imbalance Handling
###############################################################################

def supports_class_weight(algorithm_key: str) -> bool:
    """
    Return whether an algorithm supports sklearn class_weight.
    """

    return algorithm_key in {
        "logistic_regression",
        "random_forest",
        "extra_trees",
    }


def apply_imbalance_config_to_algorithms(
    algorithms_config: Dict[str, Any],
    imbalance_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply YAML-driven imbalance handling to algorithm configs.

    Current supported production-safe strategy:
        - class_weight

    This mutates a deep copy only. The original loaded YAML dictionary is not
    modified.
    """

    updated_config = copy.deepcopy(algorithms_config)

    if not bool(imbalance_config.get("enabled", False)):
        return updated_config

    strategy = str(imbalance_config.get("strategy", "none")).lower()

    if strategy == "none":
        return updated_config

    if strategy != "class_weight":
        raise ValueError(
            f"Unsupported imbalance strategy currently implemented: {strategy}. "
            "Supported implemented strategies: none, class_weight"
        )

    for algorithm_key, algorithm_config in updated_config.items():
        if not bool(algorithm_config.get("enabled", True)):
            continue

        if not supports_class_weight(algorithm_key):
            continue

        parameters = algorithm_config.setdefault("parameters", {})

        if parameters.get("class_weight") is None:
            parameters["class_weight"] = "balanced"

    return updated_config


###############################################################################
# Training Sampling
###############################################################################

def calculate_sample_size(
    row_count: int,
    performance_config: Dict[str, Any],
) -> int:
    """
    Calculate training sample size using YAML controls.
    """

    train_sample_fraction = float(
        performance_config.get("train_sample_fraction", 1.0)
    )
    min_train_rows = int(performance_config.get("min_train_rows", 0))
    max_train_rows = int(performance_config.get("max_train_rows", row_count))

    requested_rows = int(row_count * train_sample_fraction)

    sample_size = max(requested_rows, min_train_rows)
    sample_size = min(sample_size, max_train_rows)
    sample_size = min(sample_size, row_count)

    return int(sample_size)


def can_stratify(y: pd.Series) -> bool:
    """
    Determine whether stratified sampling is safe.
    """

    class_counts = y.value_counts(dropna=True)

    if len(class_counts) < 2:
        return False

    return bool((class_counts >= 2).all())


def sample_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    performance_config: Dict[str, Any],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Apply optional training-time sampling.
    """

    row_count = len(X)
    sampling_enabled = bool(
        performance_config.get("enable_training_sample", False)
    )

    if not sampling_enabled:
        return X, y, False

    sample_size = calculate_sample_size(
        row_count=row_count,
        performance_config=performance_config,
    )

    if sample_size >= row_count:
        return X, y, False

    sampling_strategy = str(
        performance_config.get("sampling_strategy", "stratified")
    ).lower()

    if sampling_strategy == "stratified" and can_stratify(y):
        sampled_X, _, sampled_y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            random_state=random_state,
            stratify=y,
        )
        return sampled_X, sampled_y, True

    if sampling_strategy in {"random", "stratified"}:
        sampled_index = X.sample(
            n=sample_size,
            random_state=random_state,
        ).index

        return X.loc[sampled_index], y.loc[sampled_index], True

    raise ValueError(
        f"Unsupported sampling_strategy: {sampling_strategy}. "
        "Supported values: stratified, random"
    )


###############################################################################
# Metrics and Threshold Optimization
###############################################################################

def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate standard binary classification metrics.
    """

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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
    """
    Return probability-like scores for metric calculation.
    """

    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]

    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))

    return pipeline.predict(X)


def get_metric_value_for_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    optimization_metric: str,
) -> float:
    """
    Calculate one threshold optimization metric.
    """

    y_pred = (y_score >= threshold).astype(int)

    metric = optimization_metric.lower()

    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))

    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))

    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))

    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))

    return float(f1_score(y_true, y_pred, zero_division=0))


def optimize_prediction_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Select prediction threshold using YAML controls.

    If disabled, returns fixed threshold.
    If enabled, searches thresholds from 0.05 to 0.95.
    """

    fixed_threshold = float(threshold_config.get("fixed_threshold", 0.50))
    enabled = bool(threshold_config.get("enabled", False))
    strategy = str(threshold_config.get("strategy", "fixed")).lower()
    optimization_metric = str(
        threshold_config.get("optimization_metric", "f1")
    ).lower()

    if not enabled or strategy == "fixed":
        return {
            "prediction_threshold": fixed_threshold,
            "threshold_optimization_enabled": False,
            "threshold_optimization_metric": optimization_metric,
            "threshold_optimization_score": None,
        }

    if strategy not in {"optimize", "search"}:
        raise ValueError(
            f"Unsupported threshold optimization strategy: {strategy}. "
            "Supported: fixed, optimize, search"
        )

    best_threshold = fixed_threshold
    best_score = -1.0

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        score = get_metric_value_for_threshold(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
            optimization_metric=optimization_metric,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return {
        "prediction_threshold": best_threshold,
        "threshold_optimization_enabled": True,
        "threshold_optimization_metric": optimization_metric,
        "threshold_optimization_score": float(best_score),
    }


###############################################################################
# Candidate Training
###############################################################################

def train_candidate_algorithm(
    model_key: str,
    algorithm_definition: AlgorithmDefinition,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_config: Dict[str, Any],
    threshold_config: Dict[str, Any],
    event_timestamp_utc: str,
    logger: Optional[logging.Logger] = None,
) -> CandidateModelResult:
    """
    Train and evaluate one candidate algorithm.
    """

    start_time = time.perf_counter()

    if logger is not None:
        logger.info("START Algorithm: %s", algorithm_definition.algorithm_name)

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

        y_score = get_prediction_scores(pipeline, X_test)

        threshold_result = optimize_prediction_threshold(
            y_true=y_test,
            y_score=y_score,
            threshold_config=threshold_config,
        )

        prediction_threshold = float(threshold_result["prediction_threshold"])
        y_pred = (y_score >= prediction_threshold).astype(int)

        metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
        )

        metrics.update(threshold_result)

        setattr(
            pipeline,
            "medfabric_prediction_threshold",
            prediction_threshold,
        )

        training_seconds = float(time.perf_counter() - start_time)

        if logger is not None:
            logger.info(
                "COMPLETE Algorithm: %s | %.2f sec | Threshold: %.2f",
                algorithm_definition.algorithm_name,
                training_seconds,
                prediction_threshold,
            )

        return CandidateModelResult(
            model_key=model_key,
            algorithm_key=algorithm_definition.algorithm_key,
            algorithm_name=algorithm_definition.algorithm_name,
            status=STATUS_SUCCESS,
            pipeline=pipeline,
            metrics=metrics,
            error_message=None,
            trained_at_utc=event_timestamp_utc,
            training_seconds=training_seconds,
            train_row_count=int(len(X_train)),
            test_row_count=int(len(X_test)),
        )

    except Exception as exc:
        training_seconds = float(time.perf_counter() - start_time)

        if logger is not None:
            logger.error(
                "FAILED Algorithm: %s | %.2f sec | Error: %s",
                algorithm_definition.algorithm_name,
                training_seconds,
                exc,
            )

        return CandidateModelResult(
            model_key=model_key,
            algorithm_key=algorithm_definition.algorithm_key,
            algorithm_name=algorithm_definition.algorithm_name,
            status=STATUS_FAILED,
            pipeline=None,
            metrics={},
            error_message=str(exc),
            trained_at_utc=event_timestamp_utc,
            training_seconds=training_seconds,
            train_row_count=int(len(X_train)),
            test_row_count=int(len(X_test)),
        )


def train_candidate_algorithms(
    model_key: str,
    algorithm_definitions: Dict[str, AlgorithmDefinition],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_config: Dict[str, Any],
    threshold_config: Dict[str, Any],
    event_timestamp_utc: str,
    parallelism_config: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> List[CandidateModelResult]:
    """
    Train all enabled candidate algorithms.
    """

    if parallelism_config is None:
        parallelism_config = resolve_parallelism_config()

    if logger is not None:
        logger.info(
            "Candidate training execution mode | Parallel: %s | Workers: %s | Strategy: %s",
            parallelism_config.get("parallel_execution"),
            parallelism_config.get("max_parallel_workers"),
            parallelism_config.get("parallel_strategy"),
        )

    tasks: List[Dict[str, Any]] = []

    for algorithm_definition in algorithm_definitions.values():
        tasks.append(
            {
                "task_name": f"{model_key}:{algorithm_definition.algorithm_key}",
                "callable": lambda alg=algorithm_definition: train_candidate_algorithm(
                    model_key=model_key,
                    algorithm_definition=alg,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    preprocessing_config=preprocessing_config,
                    threshold_config=threshold_config,
                    event_timestamp_utc=event_timestamp_utc,
                    logger=logger,
                ),
            }
        )

    task_results = run_tasks(
        tasks=tasks,
        parallelism_config=parallelism_config,
    )

    raise_if_any_task_failed(task_results)

    return [task_result.result for task_result in task_results]


###############################################################################
# Champion Selection
###############################################################################

def select_champion_model(
    candidate_results: List[CandidateModelResult],
    selection_metric: str,
    logger: Optional[logging.Logger] = None,
) -> CandidateModelResult:
    """
    Select champion algorithm by configured metric.
    """

    start_time = time.perf_counter()

    if logger is not None:
        logger.info("START Champion Selection")

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

    champion = sorted(
        successful,
        key=lambda result: float(result.metrics.get(selection_metric, -1)),
        reverse=True,
    )[0]

    selection_seconds = float(time.perf_counter() - start_time)

    if logger is not None:
        logger.info(
            "COMPLETE Champion Selection | %.2f sec | Champion: %s | %s: %s | Threshold: %s",
            selection_seconds,
            champion.algorithm_key,
            selection_metric,
            champion.metrics.get(selection_metric),
            champion.metrics.get("prediction_threshold"),
        )

    return champion


###############################################################################
# Output DataFrames
###############################################################################

def build_candidate_metrics_dataframe(
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    target_column: str,
    champion_algorithm_key: str,
    candidate_results: List[CandidateModelResult],
    full_row_count: int,
    training_row_count: int,
    sampling_applied: bool,
) -> pd.DataFrame:
    """
    Build one-row-per-metric candidate leaderboard dataframe.
    """

    rows: List[Dict[str, Any]] = []

    for result in candidate_results:
        base_record = {
            "run_id": run_id,
            "layer_name": layer_name,
            "domain_name": domain_name,
            "model_key": model_key,
            "model_name": model_name,
            "target_column": target_column,
            "algorithm_key": result.algorithm_key,
            "algorithm_name": result.algorithm_name,
            "is_champion": result.algorithm_key == champion_algorithm_key,
            "status": result.status,
            "error_message": result.error_message,
            "training_seconds": result.training_seconds,
            "train_row_count": result.train_row_count,
            "test_row_count": result.test_row_count,
            "full_row_count": full_row_count,
            "training_row_count": training_row_count,
            "sampling_applied": sampling_applied,
            "event_timestamp_utc": result.trained_at_utc,
        }

        if result.status == STATUS_SUCCESS:
            for metric_name, metric_value in result.metrics.items():
                row = dict(base_record)
                row["metric_name"] = metric_name
                if isinstance(metric_value, (int, float, np.integer, np.floating, bool)) or metric_value is None:
                    row["metric_value"] = metric_value
                    row["metric_value_text"] = None
                else:
                    row["metric_value"] = None
                    row["metric_value_text"] = str(metric_value)    

                rows.append(row)
        else:
            row = dict(base_record)
            row["metric_name"] = None
            row["metric_value"] = None
            row["metric_value_text"] = None
            rows.append(row)

    return pd.DataFrame(rows)


def build_champion_summary_dataframe(
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    target_column: str,
    champion: CandidateModelResult,
    selection_metric: str,
    full_row_count: int,
    training_row_count: int,
    sampling_applied: bool,
) -> pd.DataFrame:
    """
    Build one-row champion summary dataframe.
    """

    record: Dict[str, Any] = {
        "run_id": run_id,
        "layer_name": layer_name,
        "domain_name": domain_name,
        "model_key": model_key,
        "model_name": model_name,
        "target_column": target_column,
        "champion_algorithm_key": champion.algorithm_key,
        "champion_algorithm_name": champion.algorithm_name,
        "selection_metric": selection_metric,
        "selection_metric_value": champion.metrics.get(selection_metric),
        "prediction_threshold": champion.metrics.get("prediction_threshold"),
        "threshold_optimization_enabled": champion.metrics.get(
            "threshold_optimization_enabled"
        ),
        "threshold_optimization_metric": champion.metrics.get(
            "threshold_optimization_metric"
        ),
        "threshold_optimization_score": champion.metrics.get(
            "threshold_optimization_score"
        ),
        "training_seconds": champion.training_seconds,
        "train_row_count": champion.train_row_count,
        "test_row_count": champion.test_row_count,
        "full_row_count": full_row_count,
        "training_row_count": training_row_count,
        "sampling_applied": sampling_applied,
        "event_timestamp_utc": champion.trained_at_utc,
    }

    for metric_name, metric_value in champion.metrics.items():
        record[f"metric_{metric_name}"] = metric_value

    return pd.DataFrame([record])


###############################################################################
# Main Training Entry Point
###############################################################################

def train_model_candidates(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    model_key: str,
    model_name: str,
    modeling_defaults: Dict[str, Any],
    training_config: Dict[str, Any],
    run_id: str,
    event_timestamp_utc: str,
    layer_name: str = "Layer 2D - Enterprise Modeling Framework",
    domain_name: str = "Modeling",
    parallelism_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> TrainingResult:
    """
    Train all enabled candidate algorithms and select champion model.
    """

    if logger is not None:
        logger.info("START Training Data Preparation | Model: %s", model_key)

    preparation_start = time.perf_counter()

    X_full = dataframe[feature_columns].copy()
    y_full = dataframe[target_column].astype(int)

    if y_full.nunique(dropna=True) < 2:
        raise ValueError(
            f"Target has fewer than two classes. "
            f"model={model_key}, target={target_column}"
        )

    X_full = prepare_features_for_preprocessing(X_full)

    random_state = int(modeling_defaults.get("random_state", 42))
    test_size = float(modeling_defaults.get("test_size", 0.20))
    preprocessing_config = modeling_defaults.get("preprocessing", {})

    selection_metric = get_selection_metric(
        modeling_defaults=modeling_defaults,
        training_config=training_config,
    )

    performance_config = get_training_performance_config(training_config)
    imbalance_config = get_imbalance_config(training_config)
    threshold_config = get_threshold_optimization_config(training_config)

    X_training_base, y_training_base, sampling_applied = sample_training_data(
        X=X_full,
        y=y_full,
        performance_config=performance_config,
        random_state=random_state,
    )

    algorithms_config = (
        training_config.get("algorithms")
        or get_default_algorithms_config()
    )

    algorithms_config = apply_imbalance_config_to_algorithms(
        algorithms_config=algorithms_config,
        imbalance_config=imbalance_config,
    )

    algorithm_definitions = build_algorithm_definitions(
        algorithms_config=algorithms_config,
        random_state=random_state,
    )

    if not algorithm_definitions:
        raise ValueError("No enabled algorithms found for training.")

    stratify_target = y_training_base if can_stratify(y_training_base) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_training_base,
        y_training_base,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    preparation_seconds = float(time.perf_counter() - preparation_start)

    if logger is not None:
        logger.info(
            "COMPLETE Training Data Preparation | Model: %s | %.2f sec | Full Rows: %s | Training Rows: %s | Sampling Applied: %s | Imbalance Enabled: %s | Threshold Optimization Enabled: %s",
            model_key,
            preparation_seconds,
            len(X_full),
            len(X_training_base),
            sampling_applied,
            bool(imbalance_config.get("enabled", False)),
            bool(threshold_config.get("enabled", False)),
        )

    candidate_results = train_candidate_algorithms(
        model_key=model_key,
        algorithm_definitions=algorithm_definitions,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing_config=preprocessing_config,
        threshold_config=threshold_config,
        event_timestamp_utc=event_timestamp_utc,
        parallelism_config=parallelism_config,
        logger=logger,
    )

    champion = select_champion_model(
        candidate_results=candidate_results,
        selection_metric=selection_metric,
        logger=logger,
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
        full_row_count=len(X_full),
        training_row_count=len(X_training_base),
        sampling_applied=sampling_applied,
    )

    champion_summary_dataframe = build_champion_summary_dataframe(
        run_id=run_id,
        layer_name=layer_name,
        domain_name=domain_name,
        model_key=model_key,
        model_name=model_name,
        target_column=target_column,
        champion=champion,
        selection_metric=selection_metric,
        full_row_count=len(X_full),
        training_row_count=len(X_training_base),
        sampling_applied=sampling_applied,
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
        champion_summary_dataframe=champion_summary_dataframe,
        training_row_count=len(X_training_base),
        full_row_count=len(X_full),
        sampling_applied=sampling_applied,
    )


###############################################################################
# Standalone Validation
###############################################################################

def main() -> None:
    """
    Validate trainer module independently.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("medfabric.trainer.validation")

    dataframe = pd.DataFrame(
        {
            "member_id": range(1, 100001),
            "age": list(range(20, 120)) * 1000,
            "cost": list(range(100, 10000100, 100)),
            "gender": ["M", "F"] * 50000,
        }
    )

    dataframe["target"] = (
        dataframe["cost"] >= dataframe["cost"].quantile(0.80)
    ).astype(int)

    parallelism_config = resolve_parallelism_config(
        performance_config={
            "parallel_execution": True,
            "max_parallel_workers": 4,
            "parallel_strategy": "thread",
        }
    )

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
        training_config={
            "performance": {
                "enable_training_sample": True,
                "train_sample_fraction": 0.05,
                "min_train_rows": 2000,
                "max_train_rows": 5000,
                "sampling_strategy": "stratified",
                "log_algorithm_timing": True,
            },
            "imbalance": {
                "enabled": True,
                "strategy": "class_weight",
            },
            "threshold_optimization": {
                "enabled": True,
                "strategy": "optimize",
                "fixed_threshold": 0.50,
                "optimization_metric": "f1",
            },
            "metrics": {
                "primary_metric": "roc_auc",
                "secondary_metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                ],
            },
            "algorithms": {
                "dummy_classifier": {
                    "enabled": True,
                    "name": "Dummy Classifier",
                    "model_type": "classification",
                    "parameters": {
                        "strategy": "most_frequent",
                    },
                },
                "logistic_regression": {
                    "enabled": True,
                    "name": "Logistic Regression",
                    "model_type": "classification",
                    "parameters": {
                        "penalty": "l2",
                        "C": 1.0,
                        "solver": "lbfgs",
                        "max_iter": 1000,
                        "class_weight": None,
                    },
                },
            },
        },
        run_id="TEST_RUN",
        event_timestamp_utc="TEST_TIMESTAMP_UTC",
        parallelism_config=parallelism_config,
        logger=logger,
    )

    print("Trainer validation successful.")
    print(f"Champion algorithm: {result.champion_algorithm_key}")
    print(f"Champion metrics: {result.champion_metrics}")
    print(f"Full rows: {result.full_row_count}")
    print(f"Training rows: {result.training_row_count}")
    print(f"Sampling applied: {result.sampling_applied}")
    print(result.metrics_dataframe.head())
    print(result.champion_summary_dataframe.head())


if __name__ == "__main__":
    main()