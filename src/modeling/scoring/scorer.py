###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/scoring/scorer.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Scores the full modeling population using a trained champion model.
#
# Run:
#     python -m src.modeling.scoring.scorer
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from src.modeling.scoring.risk_tiers import assign_risk_tiers


@dataclass
class ScoringResult:
    """
    Standard scoring result returned by the Scoring Framework.
    """

    model_key: str
    model_name: str
    scoring_dataframe: pd.DataFrame
    score_column: str
    prediction_column: str
    risk_tier_column: str
    row_count: int
    scored_at_utc: str


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as ISO string.
    """

    return datetime.now(timezone.utc).isoformat()


def get_prediction_scores(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Return probability-like model scores.

    Preferred:
        predict_proba[:, 1]

    Fallback:
        decision_function converted using sigmoid

    Final fallback:
        predict output
    """

    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]

    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(features)
        return 1 / (1 + np.exp(-raw_scores))

    return pipeline.predict(features)


def score_population(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    member_key: str,
    model_key: str,
    model_name: str,
    pipeline: Pipeline,
    model_config: Dict[str, Any],
    risk_tiers_config: Dict[str, Any],
    run_id: str,
) -> ScoringResult:
    """
    Score full population for one trained champion model.
    """

    if member_key not in dataframe.columns:
        raise ValueError(f"Member key missing from scoring dataframe: {member_key}")

    missing_features = [
        column for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Scoring dataframe missing required feature columns: {missing_features}"
        )

    features = dataframe[feature_columns].copy()

    score_column = model_config.get("score_column", f"{model_key}_score")
    prediction_column = model_config.get(
        "prediction_column",
        f"{model_key}_prediction",
    )
    risk_tier_column = model_config.get(
        "risk_tier_column",
        f"{model_key}_risk_tier",
    )

    scores = get_prediction_scores(
        pipeline=pipeline,
        features=features,
    )

    predictions = pipeline.predict(features)

    scoring_dataframe = pd.DataFrame(
        {
            member_key: dataframe[member_key],
            score_column: scores,
            prediction_column: predictions,
        }
    )

    scoring_dataframe[risk_tier_column] = assign_risk_tiers(
        scores=pd.Series(scores),
        risk_tiers_config=risk_tiers_config,
    )

    scoring_dataframe["model_key"] = model_key
    scoring_dataframe["model_name"] = model_name
    scoring_dataframe["modeling_layer_run_id"] = run_id
    scoring_dataframe["scored_at_utc"] = utc_now_iso()

    return ScoringResult(
        model_key=model_key,
        model_name=model_name,
        scoring_dataframe=scoring_dataframe,
        score_column=score_column,
        prediction_column=prediction_column,
        risk_tier_column=risk_tier_column,
        row_count=len(scoring_dataframe),
        scored_at_utc=scoring_dataframe["scored_at_utc"].iloc[0],
    )


def main() -> None:
    """
    Lightweight module validation.
    """

    from src.modeling.training.trainer import train_model_candidates
    from src.modeling.training.algorithms import get_default_algorithms_config

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
        algorithms_config=get_default_algorithms_config(),
        run_id="TEST_RUN",
    )

    risk_tiers_config = {
        "very_high": {"min_value": 0.85, "max_value": 1.00, "label": "Very High"},
        "high": {"min_value": 0.70, "max_value": 0.849999, "label": "High"},
        "moderate": {"min_value": 0.40, "max_value": 0.699999, "label": "Moderate"},
        "low": {"min_value": 0.00, "max_value": 0.399999, "label": "Low"},
    }

    result = score_population(
        dataframe=dataframe,
        feature_columns=["age", "gender"],
        member_key="member_id",
        model_key="high_cost",
        model_name="High Cost Model",
        pipeline=training_result.champion_pipeline,
        model_config={
            "score_column": "high_cost_risk_score",
            "prediction_column": "high_cost_prediction",
            "risk_tier_column": "high_cost_risk_tier",
        },
        risk_tiers_config=risk_tiers_config,
        run_id="TEST_RUN",
    )

    print("Scorer validation successful.")
    print(f"Rows scored: {result.row_count}")
    print(result.scoring_dataframe.head())


if __name__ == "__main__":
    main()