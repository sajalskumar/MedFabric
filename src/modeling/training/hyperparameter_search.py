###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/training/hyperparameter_search.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Provides reusable hyperparameter search utilities for candidate algorithms.
#
# Run:
#     python -m src.modeling.training.hyperparameter_search
#
###############################################################################

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
STATUS_SKIPPED = "SKIPPED"


@dataclass
class HyperparameterSearchResult:
    algorithm_key: str
    status: str
    best_pipeline: Optional[Pipeline]
    best_params: Dict[str, Any]
    best_score: Optional[float]
    results_dataframe: pd.DataFrame
    error_message: Optional[str]
    search_seconds: float


@dataclass
class HyperparameterSearchOrchestrationResult:
    search_results_dataframe: pd.DataFrame
    algorithm_definitions: Dict[str, AlgorithmDefinition]


def is_hyperparameter_search_enabled(training_config: Dict[str, Any]) -> bool:
    return bool(
        training_config
        .get("hyperparameter_search", {})
        .get("enabled", False)
    )


def get_hyperparameter_search_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    return training_config.get("hyperparameter_search", {}) or {}


def get_default_parameter_distributions(algorithm_key: str) -> Dict[str, Any]:
    if algorithm_key == "logistic_regression":
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__max_iter": [500, 1000],
        }

    if algorithm_key == "random_forest":
        return {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [4, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 3, 5],
        }

    return {}


def run_hyperparameter_search_for_algorithm(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
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
) -> HyperparameterSearchResult:
    search_start = time.perf_counter()
    search_config = get_hyperparameter_search_config(training_config)

    if not bool(search_config.get("enabled", False)):
        return HyperparameterSearchResult(
            algorithm_key=algorithm_definition.algorithm_key,
            status=STATUS_SKIPPED,
            best_pipeline=None,
            best_params={},
            best_score=None,
            results_dataframe=pd.DataFrame(),
            error_message="Hyperparameter search disabled.",
            search_seconds=0.0,
        )

    algorithm_key = algorithm_definition.algorithm_key

    parameter_distributions = (
        search_config
        .get("parameter_distributions", {})
        .get(algorithm_key)
        or get_default_parameter_distributions(algorithm_key)
    )

    if not parameter_distributions:
        return HyperparameterSearchResult(
            algorithm_key=algorithm_key,
            status=STATUS_SKIPPED,
            best_pipeline=None,
            best_params={},
            best_score=None,
            results_dataframe=pd.DataFrame(),
            error_message=f"No parameter grid configured for algorithm: {algorithm_key}",
            search_seconds=0.0,
        )

    try:
        if logger is not None:
            logger.info(
                "START Hyperparameter Search | Model: %s | Algorithm: %s",
                model_key,
                algorithm_key,
            )

        random_state = int(
            search_config.get(
                "random_state",
                modeling_defaults.get("random_state", 42),
            )
        )
        folds = int(search_config.get("folds", 5))
        iterations = int(search_config.get("iterations", 10))
        scoring_metric = str(search_config.get("scoring_metric", "roc_auc"))

        preprocessing_config = modeling_defaults.get("preprocessing", {})

        X = prepare_features_for_preprocessing(dataframe[feature_columns].copy())
        y = dataframe[target_column].astype(int)

        preprocessor = build_preprocessor(
            dataframe=X,
            preprocessing_config=preprocessing_config,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(algorithm_definition.estimator)),
            ]
        )

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_distributions,
            n_iter=iterations,
            scoring=scoring_metric,
            cv=cv,
            random_state=random_state,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )

        search.fit(X, y)

        search_seconds = float(time.perf_counter() - search_start)
        cv_results = pd.DataFrame(search.cv_results_)

        rows = []

        for _, result_row in cv_results.iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "layer_name": layer_name,
                    "domain_name": domain_name,
                    "model_key": model_key,
                    "model_name": model_name,
                    "target_column": target_column,
                    "algorithm_key": algorithm_key,
                    "algorithm_name": algorithm_definition.algorithm_name,
                    "search_method": "random_search",
                    "scoring_metric": scoring_metric,
                    "rank_test_score": result_row.get("rank_test_score"),
                    "mean_test_score": result_row.get("mean_test_score"),
                    "std_test_score": result_row.get("std_test_score"),
                    "mean_train_score": result_row.get("mean_train_score"),
                    "std_train_score": result_row.get("std_train_score"),
                    "params": str(result_row.get("params")),
                    "is_best": result_row.get("rank_test_score") == 1,
                    "search_seconds": search_seconds,
                    "event_timestamp_utc": event_timestamp_utc,
                }
            )

        results_dataframe = pd.DataFrame(rows)

        if logger is not None:
            logger.info(
                "COMPLETE Hyperparameter Search | Model: %s | Algorithm: %s | %.2f sec | Best %s: %s",
                model_key,
                algorithm_key,
                search_seconds,
                scoring_metric,
                search.best_score_,
            )

        return HyperparameterSearchResult(
            algorithm_key=algorithm_key,
            status=STATUS_SUCCESS,
            best_pipeline=search.best_estimator_,
            best_params=dict(search.best_params_),
            best_score=float(search.best_score_),
            results_dataframe=results_dataframe,
            error_message=None,
            search_seconds=search_seconds,
        )

    except Exception as exc:
        search_seconds = float(time.perf_counter() - search_start)

        if logger is not None:
            logger.error(
                "FAILED Hyperparameter Search | Model: %s | Algorithm: %s | Error: %s",
                model_key,
                algorithm_key,
                exc,
            )

        return HyperparameterSearchResult(
            algorithm_key=algorithm_key,
            status=STATUS_FAILED,
            best_pipeline=None,
            best_params={},
            best_score=None,
            results_dataframe=pd.DataFrame(),
            error_message=str(exc),
            search_seconds=search_seconds,
        )


def run_hyperparameter_search(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
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
) -> HyperparameterSearchOrchestrationResult:
    result_frames: list[pd.DataFrame] = []
    updated_definitions = dict(algorithm_definitions)

    for algorithm_key, algorithm_definition in algorithm_definitions.items():
        result = run_hyperparameter_search_for_algorithm(
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

        if not result.results_dataframe.empty:
            result_frames.append(result.results_dataframe)

        if result.status == STATUS_SUCCESS and result.best_pipeline is not None:
            tuned_estimator = result.best_pipeline.named_steps["model"]

            updated_definitions[algorithm_key] = AlgorithmDefinition(
                algorithm_key=algorithm_definition.algorithm_key,
                algorithm_name=algorithm_definition.algorithm_name,
                model_type=algorithm_definition.model_type,
                enabled=algorithm_definition.enabled,
                estimator=tuned_estimator,
                parameters={
                    **algorithm_definition.parameters,
                    "best_params": result.best_params,
                    "best_score": result.best_score,
                },
            )

    search_results_dataframe = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )

    return HyperparameterSearchOrchestrationResult(
        search_results_dataframe=search_results_dataframe,
        algorithm_definitions=updated_definitions,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("medfabric.hyperparameter_search.validation")

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
            "logistic_regression": algorithms_config["logistic_regression"],
            "random_forest": algorithms_config["random_forest"],
        },
        random_state=42,
    )

    for algorithm_definition in algorithm_definitions.values():
        result = run_hyperparameter_search_for_algorithm(
            dataframe=dataframe,
            feature_columns=["age", "gender"],
            target_column="target",
            model_key="validation_model",
            model_name="Validation Model",
            algorithm_definition=algorithm_definition,
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
                "hyperparameter_search": {
                    "enabled": True,
                    "method": "random_search",
                    "folds": 3,
                    "iterations": 3,
                    "scoring_metric": "roc_auc",
                    "random_state": 42,
                }
            },
            run_id="TEST_RUN",
            event_timestamp_utc="TEST_TIMESTAMP_UTC",
            layer_name="Layer 2D - Enterprise Modeling Framework",
            domain_name="Modeling",
            logger=logger,
        )

        print("Hyperparameter search validation result:")
        print(f"Algorithm: {result.algorithm_key}")
        print(f"Status: {result.status}")
        print(f"Best score: {result.best_score}")
        print(f"Best params: {result.best_params}")
        print(result.results_dataframe.head())


if __name__ == "__main__":
    main()