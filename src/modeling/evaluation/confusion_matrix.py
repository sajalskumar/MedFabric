###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/confusion_matrix.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds standardized confusion matrix outputs for champion model scoring.
#
# Run:
#     python -m src.modeling.evaluation.confusion_matrix
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import confusion_matrix


def build_confusion_matrix_output(
    scoring_dataframe: pd.DataFrame,
    source_dataframe: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    run_id: str,
    layer_name: str,
    domain_name: str,
    model_key: str,
    model_name: str,
    algorithm_key: str,
    algorithm_name: str,
) -> pd.DataFrame:
    """
    Build standardized confusion matrix summary for one champion model.
    """

    if target_column not in source_dataframe.columns:
        raise ValueError(f"Target column missing from source dataframe: {target_column}")

    if prediction_column not in scoring_dataframe.columns:
        raise ValueError(
            f"Prediction column missing from scoring dataframe: {prediction_column}"
        )

    y_true = source_dataframe[target_column].astype(int)
    y_pred = scoring_dataframe[prediction_column].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=[0, 1],
    ).ravel()

    total = int(tn + fp + fn + tp)

    records = [
        ("true_negative", int(tn)),
        ("false_positive", int(fp)),
        ("false_negative", int(fn)),
        ("true_positive", int(tp)),
    ]

    output_records = []

    for matrix_cell, count in records:
        output_records.append(
            {
                "run_id": run_id,
                "layer_name": layer_name,
                "domain_name": domain_name,
                "model_key": model_key,
                "model_name": model_name,
                "algorithm_key": algorithm_key,
                "algorithm_name": algorithm_name,
                "target_column": target_column,
                "prediction_column": prediction_column,
                "matrix_cell": matrix_cell,
                "cell_count": count,
                "cell_pct": float(count / total) if total else None,
                "total_count": total,
            }
        )

    return pd.DataFrame(output_records)


def main() -> None:
    """
    Lightweight module validation.
    """

    source_df = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
        }
    )

    scoring_df = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4],
            "prediction": [0, 1, 0, 1],
        }
    )

    output = build_confusion_matrix_output(
        scoring_dataframe=scoring_df,
        source_dataframe=source_df,
        target_column="target",
        prediction_column="prediction",
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
        model_key="high_cost",
        model_name="High Cost Model",
        algorithm_key="logistic_regression",
        algorithm_name="Logistic Regression",
    )

    print("Confusion matrix validation successful.")
    print(output)


if __name__ == "__main__":
    main()