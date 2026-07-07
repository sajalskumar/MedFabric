###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_member_level_explanations.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds member-level local SHAP explanations for scored/modeling members.
#
# Run:
#     python -m src.modeling.evaluation.build_member_level_explanations
#
###############################################################################

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from src.modeling.evaluation.build_shap_explainability import compute_shap_values


DEFAULT_OUTPUT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "model_key",
    "model_name",
    "algorithm_key",
    "algorithm_name",
    "member_id",
    "feature_name",
    "feature_value",
    "local_shap_value",
    "absolute_local_shap_value",
    "local_importance_rank",
    "sample_row_count",
    "background_row_count",
    "random_state",
]


def build_member_level_explanations(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    member_key: str,
    pipeline: Any,
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    algorithm_key: str,
    algorithm_name: str,
    max_members: Optional[int] = 100,
    top_n_features: int = 5,
    background_row_count: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build local SHAP-based explanations at member level.
    """

    if member_key not in dataframe.columns:
        raise ValueError(f"Member key missing: {member_key}")

    missing_features = [
        column for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Member-level explanations missing feature columns: {missing_features}"
        )

    working_df = dataframe[[member_key] + feature_columns].dropna(
        subset=[member_key]
    ).copy()

    if working_df.empty:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    if max_members is not None and len(working_df) > max_members:
        working_df = working_df.sample(
            n=max_members,
            random_state=random_state,
        ).copy()

    member_ids = working_df[member_key].tolist()
    explain_df = working_df[feature_columns].copy()

    shap_input_df, shap_values, background_df = compute_shap_values(
        dataframe=explain_df,
        feature_columns=feature_columns,
        pipeline=pipeline,
        max_rows=None,
        background_rows=background_row_count,
        random_state=random_state,
    )

    if shap_input_df.empty or shap_values.size == 0:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

    rows = []

    for row_position, member_id in enumerate(member_ids):
        member_feature_values = shap_input_df.iloc[row_position]
        member_rows = []

        for feature_position, feature_name in enumerate(feature_columns):
            shap_value = float(shap_values[row_position][feature_position])

            member_rows.append(
                {
                    "run_id": run_id,
                    "layer_name": layer_name,
                    "domain_name": domain_name,
                    "model_key": model_key,
                    "model_name": model_name,
                    "algorithm_key": algorithm_key,
                    "algorithm_name": algorithm_name,
                    "member_id": member_id,
                    "feature_name": feature_name,
                    "feature_value": member_feature_values[feature_name],
                    "local_shap_value": shap_value,
                    "absolute_local_shap_value": abs(shap_value),
                    "sample_row_count": len(shap_input_df),
                    "background_row_count": len(background_df),
                    "random_state": random_state,
                }
            )

        member_rows = sorted(
            member_rows,
            key=lambda item: item["absolute_local_shap_value"],
            reverse=True,
        )[:top_n_features]

        for rank, row in enumerate(member_rows, start=1):
            row["local_importance_rank"] = rank
            rows.append(row)

    output_df = pd.DataFrame(rows)

    if output_df.empty:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)

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
                "scale_numeric_features": True,
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

    output = build_member_level_explanations(
        dataframe=dataframe,
        feature_columns=["age", "gender"],
        member_key="member_id",
        pipeline=training_result.champion_pipeline,
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        algorithm_key=training_result.champion_algorithm_key,
        algorithm_name=training_result.champion_algorithm_name,
    )

    print("Member-level local explanations validation successful.")
    print(output.head())


if __name__ == "__main__":
    main()