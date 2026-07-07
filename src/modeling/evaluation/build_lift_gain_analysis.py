###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_lift_gain_analysis.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds lift, gain, and decile analysis outputs for champion model scoring.
#
# Run:
#     python -m src.modeling.evaluation.build_lift_gain_analysis
#
###############################################################################

from __future__ import annotations

import pandas as pd


def build_lift_gain_analysis(
    scoring_dataframe: pd.DataFrame,
    source_dataframe: pd.DataFrame,
    target_column: str,
    score_column: str,
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    algorithm_key: str,
    algorithm_name: str,
    decile_count: int = 10,
) -> pd.DataFrame:
    """
    Build lift, gain, and decile analysis for one scored champion model.
    """

    if target_column not in source_dataframe.columns:
        raise ValueError(f"Target column missing from source dataframe: {target_column}")

    if score_column not in scoring_dataframe.columns:
        raise ValueError(f"Score column missing from scoring dataframe: {score_column}")

    analysis_df = pd.DataFrame(
        {
            "actual": source_dataframe[target_column].astype(int).reset_index(drop=True),
            "score": scoring_dataframe[score_column].astype(float).reset_index(drop=True),
        }
    )

    analysis_df = analysis_df.sort_values("score", ascending=False).reset_index(drop=True)

    analysis_df["decile"] = (
        pd.qcut(
            analysis_df.index + 1,
            q=decile_count,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    total_members = int(len(analysis_df))
    total_positives = int(analysis_df["actual"].sum())
    population_rate = float(total_positives / total_members) if total_members else 0.0

    grouped = (
        analysis_df.groupby("decile", as_index=False)
        .agg(
            member_count=("actual", "count"),
            positive_count=("actual", "sum"),
            average_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .sort_values("decile")
        .reset_index(drop=True)
    )

    grouped["negative_count"] = grouped["member_count"] - grouped["positive_count"]
    grouped["response_rate"] = grouped["positive_count"] / grouped["member_count"]
    grouped["lift"] = grouped["response_rate"] / population_rate if population_rate else 0.0
    grouped["cumulative_member_count"] = grouped["member_count"].cumsum()
    grouped["cumulative_positive_count"] = grouped["positive_count"].cumsum()

    grouped["cumulative_gain_pct"] = (
        grouped["cumulative_positive_count"] / total_positives
        if total_positives
        else 0.0
    )

    grouped["cumulative_population_pct"] = (
        grouped["cumulative_member_count"] / total_members
        if total_members
        else 0.0
    )

    grouped["cumulative_lift"] = (
        grouped["cumulative_gain_pct"] / grouped["cumulative_population_pct"]
    )

    grouped.insert(0, "algorithm_name", algorithm_name)
    grouped.insert(0, "algorithm_key", algorithm_key)
    grouped.insert(0, "model_name", model_name)
    grouped.insert(0, "model_key", model_key)
    grouped.insert(0, "domain_name", domain_name)
    grouped.insert(0, "layer_name", layer_name)
    grouped.insert(0, "run_id", run_id)

    grouped["target_column"] = target_column
    grouped["score_column"] = score_column
    grouped["decile_count"] = decile_count
    grouped["total_member_count"] = total_members
    grouped["total_positive_count"] = total_positives
    grouped["population_response_rate"] = population_rate

    return grouped


def main() -> None:
    """
    Lightweight module validation.
    """

    source_df = pd.DataFrame(
        {
            "target": [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        }
    )

    scoring_df = pd.DataFrame(
        {
            "risk_score": [0.99, 0.91, 0.82, 0.74, 0.66, 0.55, 0.43, 0.31, 0.22, 0.10],
        }
    )

    output = build_lift_gain_analysis(
        scoring_dataframe=scoring_df,
        source_dataframe=source_df,
        target_column="target",
        score_column="risk_score",
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        algorithm_key="logistic_regression",
        algorithm_name="Logistic Regression",
    )

    print("Lift/gain/decile validation successful.")
    print(output)


if __name__ == "__main__":
    main()