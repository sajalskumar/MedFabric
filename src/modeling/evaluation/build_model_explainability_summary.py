###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_model_explainability_summary.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds model explainability summary outputs for the MedFabric Modeling layer.
#
###############################################################################

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd


DEFAULT_LAYER_NAME = "Layer 2D - Enterprise Modeling Framework"
DEFAULT_DOMAIN_NAME = "Modeling"

STATUS_SUCCESS = "SUCCESS"


def normalize_feature_importance_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize feature importance column names.

    Existing MedFabric feature importance output uses:
        importance

    Explainability summary uses:
        importance_value
    """

    working_df = dataframe.copy()

    if "importance_value" not in working_df.columns:
        if "importance" in working_df.columns:
            working_df["importance_value"] = working_df["importance"]
        elif "feature_importance" in working_df.columns:
            working_df["importance_value"] = working_df["feature_importance"]
        elif "coefficient" in working_df.columns:
            working_df["importance_value"] = working_df["coefficient"]

    return working_df


def validate_feature_importance_dataframe(dataframe: pd.DataFrame) -> None:
    required_columns = [
        "run_id",
        "layer_name",
        "domain_name",
        "model_key",
        "model_name",
        "algorithm_key",
        "algorithm_name",
        "feature_name",
        "importance_value",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature importance dataframe is missing required columns: "
            f"{missing_columns}"
        )


def build_model_explainability_summary(
    feature_importance_dataframe: pd.DataFrame,
    run_id: str,
    top_n_features: int = 10,
    layer_name: str = DEFAULT_LAYER_NAME,
    domain_name: str = DEFAULT_DOMAIN_NAME,
) -> pd.DataFrame:
    if feature_importance_dataframe is None or feature_importance_dataframe.empty:
        return pd.DataFrame()

    working_df = normalize_feature_importance_dataframe(
        feature_importance_dataframe
    )

    validate_feature_importance_dataframe(working_df)

    working_df["importance_value"] = pd.to_numeric(
        working_df["importance_value"],
        errors="coerce",
    ).fillna(0.0)

    working_df["absolute_importance_value"] = (
        working_df["importance_value"].abs()
    )

    summary_rows: List[Dict[str, Any]] = []

    group_columns = [
        "model_key",
        "model_name",
        "algorithm_key",
        "algorithm_name",
    ]

    for group_values, group_df in working_df.groupby(group_columns, dropna=False):
        model_key, model_name, algorithm_key, algorithm_name = group_values

        ranked_df = (
            group_df.sort_values(
                by="absolute_importance_value",
                ascending=False,
            )
            .head(top_n_features)
            .copy()
        )

        total_importance = float(ranked_df["absolute_importance_value"].sum())

        for feature_rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
            absolute_value = float(row["absolute_importance_value"])

            importance_share = (
                absolute_value / total_importance
                if total_importance > 0
                else 0.0
            )

            summary_rows.append(
                {
                    "run_id": run_id,
                    "layer_name": layer_name,
                    "domain_name": domain_name,
                    "model_key": model_key,
                    "model_name": model_name,
                    "algorithm_key": algorithm_key,
                    "algorithm_name": algorithm_name,
                    "feature_rank": feature_rank,
                    "feature_name": row["feature_name"],
                    "importance_value": float(row["importance_value"]),
                    "absolute_importance_value": absolute_value,
                    "importance_share": float(importance_share),
                    "explainability_note": (
                        f"Rank {feature_rank} explanatory feature for "
                        f"{model_key} champion model."
                    ),
                }
            )

    return pd.DataFrame(summary_rows)


def build_model_explainability_executive_summary(
    explainability_summary_dataframe: pd.DataFrame,
    run_id: str,
    layer_name: str = DEFAULT_LAYER_NAME,
    domain_name: str = DEFAULT_DOMAIN_NAME,
) -> pd.DataFrame:
    if (
        explainability_summary_dataframe is None
        or explainability_summary_dataframe.empty
    ):
        return pd.DataFrame()

    required_columns = [
        "model_key",
        "model_name",
        "algorithm_key",
        "algorithm_name",
        "feature_rank",
        "feature_name",
        "absolute_importance_value",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in explainability_summary_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Explainability summary dataframe is missing required columns: "
            f"{missing_columns}"
        )

    rows: List[Dict[str, Any]] = []

    group_columns = [
        "model_key",
        "model_name",
        "algorithm_key",
        "algorithm_name",
    ]

    for group_values, group_df in explainability_summary_dataframe.groupby(
        group_columns,
        dropna=False,
    ):
        model_key, model_name, algorithm_key, algorithm_name = group_values

        ranked_df = group_df.sort_values("feature_rank").copy()
        primary_row = ranked_df.iloc[0]

        top_3_features = (
            ranked_df.head(3)["feature_name"].astype(str).tolist()
        )

        rows.append(
            {
                "run_id": run_id,
                "layer_name": layer_name,
                "domain_name": domain_name,
                "model_key": model_key,
                "model_name": model_name,
                "algorithm_key": algorithm_key,
                "algorithm_name": algorithm_name,
                "top_feature_count": int(len(ranked_df)),
                "primary_driver_feature": primary_row["feature_name"],
                "primary_driver_importance": float(
                    primary_row["absolute_importance_value"]
                ),
                "top_3_driver_features": ", ".join(top_3_features),
                "explainability_status": STATUS_SUCCESS,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("medfabric.explainability.validation")

    sample_df = pd.DataFrame(
        [
            {
                "run_id": "TEST_RUN",
                "layer_name": DEFAULT_LAYER_NAME,
                "domain_name": DEFAULT_DOMAIN_NAME,
                "model_key": "high_cost",
                "model_name": "High Cost Risk",
                "algorithm_key": "logistic_regression",
                "algorithm_name": "Logistic Regression",
                "feature_name": "total_allowed_amount",
                "importance": 0.91,
            },
            {
                "run_id": "TEST_RUN",
                "layer_name": DEFAULT_LAYER_NAME,
                "domain_name": DEFAULT_DOMAIN_NAME,
                "model_key": "high_cost",
                "model_name": "High Cost Risk",
                "algorithm_key": "logistic_regression",
                "algorithm_name": "Logistic Regression",
                "feature_name": "inpatient_claim_count",
                "importance": 0.62,
            },
            {
                "run_id": "TEST_RUN",
                "layer_name": DEFAULT_LAYER_NAME,
                "domain_name": DEFAULT_DOMAIN_NAME,
                "model_key": "high_cost",
                "model_name": "High Cost Risk",
                "algorithm_key": "logistic_regression",
                "algorithm_name": "Logistic Regression",
                "feature_name": "ed_visit_count",
                "importance": 0.31,
            },
        ]
    )

    explainability_df = build_model_explainability_summary(
        feature_importance_dataframe=sample_df,
        run_id="TEST_RUN",
        top_n_features=3,
    )

    executive_df = build_model_explainability_executive_summary(
        explainability_summary_dataframe=explainability_df,
        run_id="TEST_RUN",
    )

    logger.info("Model explainability summary validation successful.")

    print("Explainability summary:")
    print(explainability_df)

    print("\nExecutive explainability summary:")
    print(executive_df)


if __name__ == "__main__":
    main()