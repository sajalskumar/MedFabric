###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_shap_explainability.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds SHAP explainability output for trained champion models.
#
# Run:
#     python -m src.modeling.evaluation.build_shap_explainability
#
###############################################################################

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "model_key",
    "model_name",
    "algorithm_key",
    "algorithm_name",
    "feature_name",
    "mean_absolute_shap_value",
    "shap_importance_rank",
    "sample_row_count",
    "background_row_count",
    "random_state",
]


def _predict_positive_probability(pipeline: Any, dataframe: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class probability for SHAP model-agnostic explanation.
    """

    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(dataframe)[:, 1]

    predictions = pipeline.predict(dataframe)
    return np.asarray(predictions, dtype=float)

def compute_shap_values(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    pipeline: Any,
    max_rows: Optional[int] = 300,
    background_rows: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Compute raw SHAP values for member-level and summary explainability.
    """

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Add 'shap' to requirements.txt and run pip install -r requirements.txt."
        ) from exc

    missing_features = [
        column for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(f"SHAP missing feature columns: {missing_features}")

    working_df = dataframe[feature_columns].dropna(how="all").copy()

    if working_df.empty:
        return working_df, np.empty((0, len(feature_columns))), working_df

    if max_rows is not None and len(working_df) > max_rows:
        working_df = working_df.sample(
            n=max_rows,
            random_state=random_state,
        ).copy()

    background_df = working_df

    if len(background_df) > background_rows:
        background_df = background_df.sample(
            n=background_rows,
            random_state=random_state,
        ).copy()

    explainer = shap.KernelExplainer(
        lambda input_data: _predict_positive_probability(
            pipeline=pipeline,
            dataframe=pd.DataFrame(input_data, columns=feature_columns),
        ),
        background_df,
    )

    shap_values = explainer.shap_values(working_df)
    values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[:, :, -1]

    return working_df, values, background_df

def compute_shap_values(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    pipeline: Any,
    max_rows: Optional[int] = 300,
    background_rows: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Compute raw SHAP values for member-level and summary explainability.
    """

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Add 'shap' to requirements.txt and run pip install -r requirements.txt."
        ) from exc

    missing_features = [
        column for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(f"SHAP missing feature columns: {missing_features}")

    working_df = dataframe[feature_columns].dropna(how="all").copy()

    if working_df.empty:
        return working_df, np.empty((0, len(feature_columns))), working_df

    if max_rows is not None and len(working_df) > max_rows:
        working_df = working_df.sample(
            n=max_rows,
            random_state=random_state,
        ).copy()

    background_df = working_df

    if len(background_df) > background_rows:
        background_df = background_df.sample(
            n=background_rows,
            random_state=random_state,
        ).copy()

    explainer = shap.KernelExplainer(
        lambda input_data: _predict_positive_probability(
            pipeline=pipeline,
            dataframe=pd.DataFrame(input_data, columns=feature_columns),
        ),
        background_df,
    )

    shap_values = explainer.shap_values(working_df)
    values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[:, :, -1]

    return working_df, values, background_df

def build_shap_explainability(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    pipeline: Any,
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    algorithm_key: str,
    algorithm_name: str,
    max_rows: Optional[int] = 300,
    background_rows: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build SHAP summary for one champion model.
    """

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Add 'shap' to requirements.txt and run pip install -r requirements.txt."
        ) from exc

    missing_features = [
        column for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(f"SHAP missing feature columns: {missing_features}")

    working_df = dataframe[feature_columns].dropna(how="all").copy()

    if working_df.empty:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    if max_rows is not None and len(working_df) > max_rows:
        working_df = working_df.sample(
            n=max_rows,
            random_state=random_state,
        ).copy()

    background_df = working_df

    if len(background_df) > background_rows:
        background_df = background_df.sample(
            n=background_rows,
            random_state=random_state,
        ).copy()

    explainer = shap.KernelExplainer(
        lambda input_data: _predict_positive_probability(
            pipeline=pipeline,
            dataframe=pd.DataFrame(input_data, columns=feature_columns),
        ),
        background_df,
    )

    shap_values = explainer.shap_values(working_df)

    values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs_values = np.abs(values).mean(axis=0)

    output_df = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "mean_absolute_shap_value": mean_abs_values,
        }
    )

    output_df["mean_absolute_shap_value"] = pd.to_numeric(
        output_df["mean_absolute_shap_value"],
        errors="coerce",
    ).fillna(0.0)

    output_df = output_df.sort_values(
        by="mean_absolute_shap_value",
        ascending=False,
    ).reset_index(drop=True)

    output_df["shap_importance_rank"] = range(1, len(output_df) + 1)

    output_df.insert(0, "algorithm_name", algorithm_name)
    output_df.insert(0, "algorithm_key", algorithm_key)
    output_df.insert(0, "model_name", model_name)
    output_df.insert(0, "model_key", model_key)
    output_df.insert(0, "domain_name", domain_name)
    output_df.insert(0, "layer_name", layer_name)
    output_df.insert(0, "run_id", run_id)

    output_df["sample_row_count"] = int(len(working_df))
    output_df["background_row_count"] = int(len(background_df))
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

    output = build_shap_explainability(
        dataframe=dataframe,
        feature_columns=["age", "gender"],
        pipeline=training_result.champion_pipeline,
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        algorithm_key=training_result.champion_algorithm_key,
        algorithm_name=training_result.champion_algorithm_name,
    )

    print("SHAP explainability validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()