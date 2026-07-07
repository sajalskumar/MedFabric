###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_permutation_importance.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds permutation importance output for trained champion models.
#
#     Permutation importance measures how much model performance drops when one
#     feature is randomly shuffled. Larger drops mean the model depends more on
#     that feature.
#
# Run:
#     python -m src.modeling.evaluation.build_permutation_importance
#
###############################################################################

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from sklearn.inspection import permutation_importance


DEFAULT_OUTPUT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "model_key",
    "model_name",
    "algorithm_key",
    "algorithm_name",
    "feature_name",
    "permutation_importance_mean",
    "permutation_importance_std",
    "permutation_importance_rank",
    "scoring_metric",
    "n_repeats",
    "sample_row_count",
    "random_state",
]


def build_permutation_importance(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    pipeline: Any,
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    algorithm_key: str,
    algorithm_name: str,
    scoring_metric: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
    max_rows: Optional[int] = 5000,
) -> pd.DataFrame:
    """
    Build permutation importance summary for one champion model.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column missing: {target_column}")

    missing_features = [
        column for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Permutation importance missing feature columns: {missing_features}"
        )

    working_df = dataframe[feature_columns + [target_column]].dropna(
        subset=[target_column]
    ).copy()

    if working_df.empty:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    if max_rows is not None and len(working_df) > max_rows:
        working_df = working_df.sample(
            n=max_rows,
            random_state=random_state,
        ).copy()

    x_values = working_df[feature_columns].copy()
    y_values = working_df[target_column].astype(int)

    if y_values.nunique() < 2:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    result = permutation_importance(
        estimator=pipeline,
        X=x_values,
        y=y_values,
        scoring=scoring_metric,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    output_df = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    )

    output_df["permutation_importance_mean"] = pd.to_numeric(
        output_df["permutation_importance_mean"],
        errors="coerce",
    ).fillna(0.0)

    output_df["permutation_importance_std"] = pd.to_numeric(
        output_df["permutation_importance_std"],
        errors="coerce",
    ).fillna(0.0)

    output_df = output_df.sort_values(
        by="permutation_importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    output_df["permutation_importance_rank"] = range(1, len(output_df) + 1)

    output_df.insert(0, "algorithm_name", algorithm_name)
    output_df.insert(0, "algorithm_key", algorithm_key)
    output_df.insert(0, "model_name", model_name)
    output_df.insert(0, "model_key", model_key)
    output_df.insert(0, "domain_name", domain_name)
    output_df.insert(0, "layer_name", layer_name)
    output_df.insert(0, "run_id", run_id)

    output_df["scoring_metric"] = scoring_metric
    output_df["n_repeats"] = int(n_repeats)
    output_df["sample_row_count"] = int(len(working_df))
    output_df["random_state"] = int(random_state)

    return output_df[DEFAULT_OUTPUT_COLUMNS]


def main() -> None:
    """
    Lightweight module validation.
    """

    from src.modeling.training.algorithms import get_default_algorithms_config
    from src.modeling.training.trainer import train_model_candidates

    dataframe = pd.DataFrame(
        {
            "member_id": range(1, 301),
            "age": list(range(20, 320)),
            "cost": list(range(100, 30100, 100)),
            "gender": ["M", "F"] * 150,
        }
    )

    dataframe["target"] = (
        dataframe["cost"] >= dataframe["cost"].quantile(0.80)
    ).astype(int)

    training_result = train_model_candidates(
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
                "enable_training_sample": False,
            },
            "metrics": {
                "primary_metric": "roc_auc",
            },
            "algorithms": get_default_algorithms_config(),
        },
        run_id="TEST_RUN",
        event_timestamp_utc="TEST_TIMESTAMP_UTC",
    )

    output = build_permutation_importance(
        dataframe=dataframe,
        feature_columns=["age", "gender"],
        target_column="target",
        pipeline=training_result.champion_pipeline,
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        algorithm_key=training_result.champion_algorithm_key,
        algorithm_name=training_result.champion_algorithm_name,
    )

    print("Permutation importance validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()