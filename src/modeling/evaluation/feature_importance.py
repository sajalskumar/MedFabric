###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/feature_importance.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Extracts feature importance outputs from trained champion model pipelines.
#.    this is standalone module validation test, not a production modelling logic. 
#
# Run:
#     python -m src.modeling.evaluation.feature_importance
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.pipeline import Pipeline


def get_transformed_feature_names(
    pipeline: Pipeline,
    fallback_feature_columns: List[str],
) -> List[str]:
    """
    Return transformed feature names from a fitted preprocessing pipeline.
    """

    preprocessor = pipeline.named_steps.get("preprocessor")

    if preprocessor is None:
        return fallback_feature_columns

    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return fallback_feature_columns


def extract_feature_importance(
    pipeline: Pipeline,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Extract feature importance or coefficient values from trained pipeline.
    """

    model = pipeline.named_steps.get("model")

    if model is None:
        return pd.DataFrame(columns=["feature_name", "importance", "importance_type"])

    feature_names = get_transformed_feature_names(
        pipeline=pipeline,
        fallback_feature_columns=feature_columns,
    )

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        values = model.coef_[0]
        importance_type = "coefficient"
    else:
        return pd.DataFrame(columns=["feature_name", "importance", "importance_type"])

    length = min(len(feature_names), len(values))

    return (
        pd.DataFrame(
            {
                "feature_name": feature_names[:length],
                "importance": values[:length],
                "importance_type": importance_type,
            }
        )
        .sort_values("importance", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


def build_feature_importance_output(
    pipeline: Pipeline,
    feature_columns: List[str],
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    algorithm_key: str,
    algorithm_name: str,
) -> pd.DataFrame:
    """
    Build standardized feature importance output.
    """

    importance_df = extract_feature_importance(
        pipeline=pipeline,
        feature_columns=feature_columns,
    )

    if importance_df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "layer_name",
                "domain_name",
                "model_key",
                "model_name",
                "algorithm_key",
                "algorithm_name",
                "feature_name",
                "importance",
                "importance_type",
            ]
        )

    importance_df.insert(0, "algorithm_name", algorithm_name)
    importance_df.insert(0, "algorithm_key", algorithm_key)
    importance_df.insert(0, "model_name", model_name)
    importance_df.insert(0, "model_key", model_key)
    importance_df.insert(0, "domain_name", domain_name)
    importance_df.insert(0, "layer_name", layer_name)
    importance_df.insert(0, "run_id", run_id)

    return importance_df


def main() -> None:
    """
    Lightweight module validation.
    """

    import pandas as pd

    from src.modeling.training.algorithms import get_default_algorithms_config
    from src.modeling.training.trainer import train_model_candidates

    dataframe = pd.DataFrame(
        {
            "member_id": range(1, 101),
            "age": list(range(20, 120)),
            "cost": list(range(100, 10100, 100)),
            "gender": ["M", "F"] * 50,
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
            "prformance":{
                "enable_training_sample": False,
            },
            "metrics": {
                "primary_metric": "roc_auc",
            },
            "algorithm":get_default_algorithms_config(),
        },
        run_id="TEST_RUN",
        event_timestamp_utc="TEST_TIMESTAMP_UTC"
    )

    output = build_feature_importance_output(
        pipeline=training_result.champion_pipeline,
        feature_columns=["age", "gender"],
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        algorithm_key=training_result.champion_algorithm_key,
        algorithm_name=training_result.champion_algorithm_name,
    )

    print("Feature importance validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()