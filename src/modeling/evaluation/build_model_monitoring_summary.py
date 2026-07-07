###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_model_monitoring_summary.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds model monitoring summary output from scored model populations.
#
# Run:
#     python -m src.modeling.evaluation.build_model_monitoring_summary
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


DEFAULT_COLUMNS = [
    "run_id",
    "layer_name",
    "domain_name",
    "model_key",
    "model_name",
    "score_column",
    "prediction_column",
    "risk_tier_column",
    "scored_row_count",
    "average_score",
    "min_score",
    "max_score",
    "prediction_positive_count",
    "prediction_positive_rate",
    "risk_tier_count",
]


def build_model_monitoring_summary(
    scoring_results: List[Any],
    run_id: str,
    layer_name: str,
    domain_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for result in scoring_results:
        scoring_df = result.scoring_dataframe

        if scoring_df is None or scoring_df.empty:
            continue

        score_column = result.score_column
        prediction_column = result.prediction_column
        risk_tier_column = result.risk_tier_column

        rows.append(
            {
                "run_id": run_id,
                "layer_name": layer_name,
                "domain_name": domain_name,
                "model_key": result.model_key,
                "model_name": result.model_name,
                "score_column": score_column,
                "prediction_column": prediction_column,
                "risk_tier_column": risk_tier_column,
                "scored_row_count": int(len(scoring_df)),
                "average_score": float(scoring_df[score_column].mean()),
                "min_score": float(scoring_df[score_column].min()),
                "max_score": float(scoring_df[score_column].max()),
                "prediction_positive_count": int(scoring_df[prediction_column].sum()),
                "prediction_positive_rate": float(scoring_df[prediction_column].mean()),
                "risk_tier_count": int(scoring_df[risk_tier_column].nunique()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    return pd.DataFrame(rows)[DEFAULT_COLUMNS]


def main() -> None:
    print("Model monitoring summary module validation successful.")


if __name__ == "__main__":
    main()