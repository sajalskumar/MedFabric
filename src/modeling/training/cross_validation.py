###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/training/cross_validation.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Provides reusable cross-validation utilities for candidate model evaluation.
#
# Run:
#     python -m src.modeling.training.cross_validation
#
###############################################################################

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
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
class CrossValidationResult:
    """
    Cross-validation result returned to the trainer.
    """

    fold_metrics_dataframe: pd.DataFrame
    summary_dataframe: pd.DataFrame


def is_cross_validation_enabled(training_config: Dict[str, Any]) -> bool:
    """
    Return whether cross-validation is enabled.
    """

    return bool(
        training_config
        .get("cross_validation", {})
        .get("enabled", False)
    )


def get_cross_validation_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return training.cross_validation config.
    """

    return training_config.get("cross_validation", {}) or {}


def calculate_fold_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Calculate classification metrics for one fold.
    """

    metrics: Dict[str, Optional[float]] = {
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
    Return model probability-like scores.
    """

    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]

    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))

    return pipeline.predict(X)


def build_cv_pipeline(
    algorithm_definition: AlgorithmDefinition,
    X_train: pd.DataFrame,
    preprocessing_config: Dict[str, Any],
) -> Pipeline:
    """
    Build one fold-specific pipeline.
    """

    preprocessor = build_preprocessor(
        dataframe=X_train,
        preprocessing_config=preprocessing_config,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(algorithm_definition.estimator)),
        ]
    )


def run_cross_validation_for_algorithm(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    model_key: str,
    model_name: str,
    algorithm_definition: AlgorithmDefinition,
    modeling_defaults: Dict[str, Any],
    training_config: Dict[str, Any],
    run_id: str,
    event_timestamp_utc: str,
    layer_name: str,
    domain_name: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Run cross-validation for one candidate algorithm.
    """

    cv_config = get_cross_validation_config(training_config)

    folds = int(cv_config.get("folds", 5))
    shuffle = bool(cv_config.get("shuffle", True))
    random_state = int(cv_config.get("random_state", modeling_defaults.get("random_state", 42)))

    preprocessing_config = modeling_defaults.get("preprocessing", {})

    X = prepare_features_for_preprocessing(dataframe[feature_columns].copy())
    y = dataframe[target_column].astype(int)

    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"Cross-validation requires at least two target classes. "
            f"model={model_key}, target={target_column}"
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    rows: List[Dict[str, Any]] = []

    if logger is not None:
        logger.info(
            "START Cross Validation | Model: %s | Algorithm: %s | Folds: %s",
            model_key,
            algorithm_definition.algorithm_key,
            folds,
        )

    for fold_number, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        fold_start = time.perf_counter()

        X_train = X.iloc[train_index].copy()
        X_test = X.iloc[test_index].copy()
        y_train = y.iloc[train_index].copy()
        y_test = y.iloc[test_index].copy()

        status = STATUS_SUCCESS
        error_message = None
        metrics: Dict[str, Optional[float]] = {}

        try:
            pipeline = build_cv_pipeline(
                algorithm_definition=algorithm_definition,
                X_train=X_train,
                preprocessing_config=preprocessing_config,
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_score = get_prediction_scores(pipeline, X_test)

            metrics = calculate_fold_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
            )

        except Exception as exc:
            status = STATUS_FAILED
            error_message = str(exc)

        fold_seconds = float(time.perf_counter() - fold_start)

        base_record = {
            "run_id": run_id,
            "layer_name": layer_name,
            "domain_name": domain_name,
            "model_key": model_key,
            "model_name": model_name,
            "target_column": target_column,
            "algorithm_key": algorithm_definition.algorithm_key,
            "algorithm_name": algorithm_definition.algorithm_name,
            "fold_number": fold_number,
            "fold_count": folds,
            "status": status,
            "error_message": error_message,
            "train_row_count": int(len(X_train)),
            "test_row_count": int(len(X_test)),
            "fold_seconds": fold_seconds,
            "event_timestamp_utc": event_timestamp_utc,
        }

        if status == STATUS_SUCCESS:
            for metric_name, metric_value in metrics.items():
                row = dict(base_record)
                row["metric_name"] = metric_name
                row["metric_value"] = metric_value
                rows.append(row)
        else:
            row = dict(base_record)
            row["metric_name"] = None
            row["metric_value"] = None
            rows.append(row)

        if logger is not None:
            logger.info(
                "COMPLETE CV Fold | Model: %s | Algorithm: %s | Fold: %s/%s | %.2f sec | Status: %s",
                model_key,
                algorithm_definition.algorithm_key,
                fold_number,
                folds,
                fold_seconds,
                status,
            )

    return pd.DataFrame(rows)


def build_cross_validation_summary(
    fold_metrics_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build algorithm-level cross-validation summary.
    """

    if fold_metrics_dataframe.empty:
        return pd.DataFrame()

    successful = fold_metrics_dataframe[
        fold_metrics_dataframe["status"] == STATUS_SUCCESS
    ].copy()

    successful["metric_value"] = pd.to_numeric(
        successful["metric_value"],
        errors="coerce",
    )

    grouped = (
        successful
        .groupby(
            [
                "run_id",
                "layer_name",
                "domain_name",
                "model_key",
                "model_name",
                "target_column",
                "algorithm_key",
                "algorithm_name",
                "metric_name",
            ],
            dropna=False,
        )
        .agg(
            cv_metric_mean=("metric_value", "mean"),
            cv_metric_std=("metric_value", "std"),
            cv_metric_min=("metric_value", "min"),
            cv_metric_max=("metric_value", "max"),
            cv_metric_count=("metric_value", "count"),
            cv_fold_count=("fold_number", "nunique"),
            cv_total_seconds=("fold_seconds", "sum"),
            event_timestamp_utc=("event_timestamp_utc", "max"),
        )
        .reset_index()
    )

    grouped["cv_metric_std"] = grouped["cv_metric_std"].fillna(0.0)

    return grouped


def run_cross_validation(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    model_key: str,
    model_name: str,
    algorithm_definitions: Dict[str, AlgorithmDefinition],
    modeling_defaults: Dict[str, Any],
    training_config: Dict[str, Any],
    run_id: str,
    event_timestamp_utc: str,
    layer_name: str,
    domain_name: str,
    logger: Optional[logging.Logger] = None,
) -> CrossValidationResult:
    """
    Run cross-validation for all enabled candidate algorithms.
    """

    frames: List[pd.DataFrame] = []

    for algorithm_definition in algorithm_definitions.values():
        frame = run_cross_validation_for_algorithm(
            dataframe=dataframe,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key=model_key,
            model_name=model_name,
            algorithm_definition=algorithm_definition,
            modeling_defaults=modeling_defaults,
            training_config=training_config,
            run_id=run_id,
            event_timestamp_utc=event_timestamp_utc,
            layer_name=layer_name,
            domain_name=domain_name,
            logger=logger,
        )
        frames.append(frame)

    fold_metrics_dataframe = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )

    summary_dataframe = build_cross_validation_summary(
        fold_metrics_dataframe=fold_metrics_dataframe,
    )

    return CrossValidationResult(
        fold_metrics_dataframe=fold_metrics_dataframe,
        summary_dataframe=summary_dataframe,
    )


def main() -> None:
    """
    Lightweight validation.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("medfabric.cross_validation.validation")

    dataframe = pd.DataFrame(
        {
            "member_id": range(1, 1001),
            "age": list(range(20, 120)) * 10,
            "cost": list(range(100, 100100, 100)),
            "gender": ["M", "F"] * 500,
        }
    )

    dataframe["target"] = (
        dataframe["cost"] >= dataframe["cost"].quantile(0.80)
    ).astype(int)

    algorithms_config = get_default_algorithms_config()
    algorithm_definitions = build_algorithm_definitions(
        algorithms_config={
            "dummy_classifier": algorithms_config["dummy_classifier"],
            "logistic_regression": algorithms_config["logistic_regression"],
        },
        random_state=42,
    )

    result = run_cross_validation(
        dataframe=dataframe,
        feature_columns=["age", "gender"],
        target_column="target",
        model_key="validation_model",
        model_name="Validation Model",
        algorithm_definitions=algorithm_definitions,
        modeling_defaults={
            "random_state": 42,
            "preprocessing": {
                "numeric_imputation_strategy": "median",
                "categorical_imputation_strategy": "most_frequent",
                "scale_numeric_features": False,
                "one_hot_encode_categorical_features": True,
            },
        },
        training_config={
            "cross_validation": {
                "enabled": True,
                "strategy": "stratified_kfold",
                "folds": 5,
                "shuffle": True,
                "random_state": 42,
                "save_fold_metrics": True,
                "save_fold_predictions": False,
            }
        },
        run_id="TEST_RUN",
        event_timestamp_utc="TEST_TIMESTAMP_UTC",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        logger=logger,
    )

    print("Cross-validation validation successful.")
    print(result.fold_metrics_dataframe.head())
    print(result.summary_dataframe.head())


if __name__ == "__main__":
    main()