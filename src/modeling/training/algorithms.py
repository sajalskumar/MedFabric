###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/training/algorithms.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Provides a centralized algorithm registry and factory for MedFabric
#     classification models.
#
# Run:
#     python -m src.modeling.training.algorithms
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


SUPPORTED_CLASSIFIERS = {
    "dummy_classifier",
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
}


@dataclass
class AlgorithmDefinition:
    algorithm_key: str
    algorithm_name: str
    model_type: str
    enabled: bool
    estimator: Any
    parameters: Dict[str, Any]


def build_classifier(
    algorithm_key: str,
    parameters: Dict[str, Any] | None = None,
    random_state: int = 42,
) -> Any:
    """
    Build a sklearn classifier from algorithm key and parameters.
    """

    params = parameters or {}

    if algorithm_key == "dummy_classifier":
        return DummyClassifier(
            strategy=params.get("strategy", "most_frequent"),
            random_state=random_state,
        )

    if algorithm_key == "logistic_regression":
        return LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=float(params.get("C", 1.0)),
            solver=params.get("solver", "lbfgs"),
            max_iter=int(params.get("max_iter", 1000)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
        )

    if algorithm_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
            n_jobs=int(params.get("n_jobs", -1)),
        )

    if algorithm_key == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
            n_jobs=int(params.get("n_jobs", -1)),
        )

    if algorithm_key == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=random_state,
        )

    if algorithm_key == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            random_state=random_state,
        )

    if algorithm_key == "adaboost":
        return AdaBoostClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=random_state,
        )

    raise ValueError(
        f"Unsupported classifier: {algorithm_key}. "
        f"Supported classifiers: {sorted(SUPPORTED_CLASSIFIERS)}"
    )


def build_algorithm_definitions(
    algorithms_config: Dict[str, Any],
    random_state: int = 42,
) -> Dict[str, AlgorithmDefinition]:
    """
    Build enabled algorithm definitions from configuration.
    """

    definitions: Dict[str, AlgorithmDefinition] = {}

    for algorithm_key, config in algorithms_config.items():
        enabled = bool(config.get("enabled", True))

        if not enabled:
            continue

        parameters = config.get("parameters", {})

        estimator = build_classifier(
            algorithm_key=algorithm_key,
            parameters=parameters,
            random_state=random_state,
        )

        definitions[algorithm_key] = AlgorithmDefinition(
            algorithm_key=algorithm_key,
            algorithm_name=config.get("name", algorithm_key),
            model_type=config.get("model_type", "classification"),
            enabled=enabled,
            estimator=estimator,
            parameters=parameters,
        )

    return definitions


def get_default_algorithms_config() -> Dict[str, Any]:
    """
    Return default algorithm configuration for local validation and fallback.
    """

    return {
        "dummy_classifier": {
            "enabled": true if False else True,
            "name": "Dummy Classifier",
            "parameters": {
                "strategy": "most_frequent",
            },
        },
        "logistic_regression": {
            "enabled": True,
            "name": "Logistic Regression",
            "parameters": {
                "max_iter": 1000,
                "solver": "lbfgs",
            },
        },
        "random_forest": {
            "enabled": True,
            "name": "Random Forest",
            "parameters": {
                "n_estimators": 100,
                "max_depth": None,
            },
        },
        "extra_trees": {
            "enabled": True,
            "name": "Extra Trees",
            "parameters": {
                "n_estimators": 100,
                "max_depth": None,
            },
        },
        "gradient_boosting": {
            "enabled": True,
            "name": "Gradient Boosting",
            "parameters": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
        },
        "hist_gradient_boosting": {
            "enabled": True,
            "name": "Histogram Gradient Boosting",
            "parameters": {
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
            },
        },
        "adaboost": {
            "enabled": True,
            "name": "AdaBoost",
            "parameters": {
                "n_estimators": 100,
                "learning_rate": 1.0,
            },
        },
    }


def main() -> None:
    """
    Lightweight module validation.
    """

    algorithms = build_algorithm_definitions(
        algorithms_config=get_default_algorithms_config(),
        random_state=42,
    )

    print("Algorithm registry validation successful.")
    print(f"Algorithms built: {list(algorithms.keys())}")

    for key, definition in algorithms.items():
        print(f"{key}: {type(definition.estimator).__name__}")


if __name__ == "__main__":
    main()