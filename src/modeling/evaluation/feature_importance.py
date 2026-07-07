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
#     Extracts standardized native feature importance outputs from trained
#     champion model pipelines.
#
# Supports:
#     - Plain sklearn Pipeline objects
#     - CalibratedClassifierCV wrapping a fitted Pipeline
#     - Tree-based feature_importances_
#     - Linear model coef_
#
# Outputs:
#     - feature_name
#     - importance
#     - importance_abs
#     - importance_rank
#     - importance_direction
#     - importance_type
#     - native_importance_available
#
# Run:
#     python -m src.modeling.evaluation.feature_importance
#
###############################################################################

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd


EMPTY_COLUMNS = [
    "feature_name",
    "importance",
    "importance_abs",
    "importance_rank",
    "importance_direction",
    "importance_type",
    "native_importance_available",
]


def unwrap_pipeline(model_object: Any) -> Any:
    """
    Return the underlying fitted pipeline/model from a possible calibrated model.
    """

    if hasattr(model_object, "named_steps"):
        return model_object

    if hasattr(model_object, "estimator") and model_object.estimator is not None:
        if hasattr(model_object.estimator, "named_steps"):
            return model_object.estimator

    if hasattr(model_object, "base_estimator") and model_object.base_estimator is not None:
        if hasattr(model_object.base_estimator, "named_steps"):
            return model_object.base_estimator

    if hasattr(model_object, "calibrated_classifiers_"):
        calibrated_classifiers = getattr(model_object, "calibrated_classifiers_", [])

        if calibrated_classifiers:
            calibrated_classifier = calibrated_classifiers[0]

            if hasattr(calibrated_classifier, "estimator"):
                estimator = calibrated_classifier.estimator
                if hasattr(estimator, "named_steps"):
                    return estimator
                return estimator

            if hasattr(calibrated_classifier, "base_estimator"):
                estimator = calibrated_classifier.base_estimator
                if hasattr(estimator, "named_steps"):
                    return estimator
                return estimator

    return model_object


def get_transformed_feature_names(
    pipeline: Any,
    fallback_feature_columns: List[str],
) -> List[str]:
    """
    Return transformed feature names from a fitted preprocessing pipeline.
    """

    pipeline = unwrap_pipeline(pipeline)

    if not hasattr(pipeline, "named_steps"):
        return fallback_feature_columns

    preprocessor = pipeline.named_steps.get("preprocessor")

    if preprocessor is None:
        return fallback_feature_columns

    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return fallback_feature_columns


def get_model_from_pipeline(pipeline: Any) -> Optional[Any]:
    """
    Return the final model step from a Pipeline, or the object itself when the
    supplied object is already a fitted estimator.
    """

    pipeline = unwrap_pipeline(pipeline)

    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model")

    return pipeline


def build_importance_dataframe(
    feature_names: List[str],
    values: Any,
    importance_type: str,
) -> pd.DataFrame:
    """
    Build standardized native feature importance dataframe.
    """

    length = min(len(feature_names), len(values))

    if length == 0:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    output_df = pd.DataFrame(
        {
            "feature_name": feature_names[:length],
            "importance": values[:length],
            "importance_type": importance_type,
        }
    )

    output_df["importance"] = pd.to_numeric(
        output_df["importance"],
        errors="coerce",
    ).fillna(0.0)

    output_df["importance_abs"] = output_df["importance"].abs()

    output_df = output_df.sort_values(
        by="importance_abs",
        ascending=False,
    ).reset_index(drop=True)

    output_df["importance_rank"] = range(1, len(output_df) + 1)

    output_df["importance_direction"] = output_df["importance"].apply(
        lambda value: (
            "positive"
            if value > 0
            else "negative"
            if value < 0
            else "neutral"
        )
    )

    output_df["native_importance_available"] = True

    return output_df[EMPTY_COLUMNS]


def extract_feature_importance(
    pipeline: Any,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Extract native feature importance or coefficient values from a trained model.
    """

    unwrapped_pipeline = unwrap_pipeline(pipeline)
    model = get_model_from_pipeline(unwrapped_pipeline)

    if model is None:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    feature_names = get_transformed_feature_names(
        pipeline=unwrapped_pipeline,
        fallback_feature_columns=feature_columns,
    )

    if hasattr(model, "feature_importances_"):
        return build_importance_dataframe(
            feature_names=feature_names,
            values=model.feature_importances_,
            importance_type="feature_importance",
        )

    if hasattr(model, "coef_"):
        coefficient_values = model.coef_[0]

        return build_importance_dataframe(
            feature_names=feature_names,
            values=coefficient_values,
            importance_type="coefficient",
        )

    return pd.DataFrame(columns=EMPTY_COLUMNS)


def build_feature_importance_output(
    pipeline: Any,
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
    Build standardized feature importance output with enterprise metadata.
    """

    importance_df = extract_feature_importance(
        pipeline=pipeline,
        feature_columns=feature_columns,
    )

    output_columns = [
        "run_id",
        "layer_name",
        "domain_name",
        "model_key",
        "model_name",
        "algorithm_key",
        "algorithm_name",
        "feature_name",
        "importance",
        "importance_abs",
        "importance_rank",
        "importance_direction",
        "importance_type",
        "native_importance_available",
    ]

    if importance_df.empty:
        return pd.DataFrame(columns=output_columns)

    importance_df.insert(0, "algorithm_name", algorithm_name)
    importance_df.insert(0, "algorithm_key", algorithm_key)
    importance_df.insert(0, "model_name", model_name)
    importance_df.insert(0, "model_key", model_key)
    importance_df.insert(0, "domain_name", domain_name)
    importance_df.insert(0, "layer_name", layer_name)
    importance_df.insert(0, "run_id", run_id)

    return importance_df[output_columns]


def main() -> None:
    """
    Lightweight module validation test.
    """

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

    print("Enhanced native feature importance validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()