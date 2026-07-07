###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/evaluation/build_target_quality_report.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds target quality statistics for modeling targets before model training.
#
# Run:
#     python -m src.modeling.evaluation.build_target_quality_report
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from math import log2
from typing import Any, Dict, List, Optional

import pandas as pd


PASS_STATUS = "PASS"
WARNING_STATUS = "WARNING"
FAIL_STATUS = "FAIL"

DEFAULT_REPORT_TYPE = "target_quality_report"
DEFAULT_REPORT_VERSION = "1.0"


def utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO-formatted string.
    """

    return datetime.now(timezone.utc).isoformat()


def calculate_entropy(positive_pct: float, negative_pct: float) -> float:
    """
    Calculate binary entropy.

    Entropy is highest when classes are balanced 50/50.
    Entropy is lowest when one class dominates.
    """

    entropy = 0.0

    for pct in [positive_pct, negative_pct]:
        if pct > 0:
            entropy -= pct * log2(pct)

    return float(entropy)


def is_binary_target(series: pd.Series) -> bool:
    """
    Check whether target contains only binary values.
    """

    non_null_series = series.dropna()

    if non_null_series.empty:
        return False

    normalized_values = set(
        non_null_series.astype(str).str.strip().str.lower().unique().tolist()
    )

    valid_binary_values = {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "y", "n"}

    return normalized_values.issubset(valid_binary_values)


def normalize_binary_target(series: pd.Series) -> pd.Series:
    """
    Normalize common binary target values to 0 and 1.
    """

    normalized = series.astype(str).str.strip().str.lower()

    positive_values = {"1", "1.0", "true", "yes", "y"}
    negative_values = {"0", "0.0", "false", "no", "n"}

    return normalized.map(
        lambda value: 1 if value in positive_values else 0 if value in negative_values else None
    )


def assess_target_quality(
    row_count: int,
    null_count: int,
    unique_count: int,
    binary_target: bool,
    positive_pct: Optional[float],
    imbalance_ratio: Optional[float],
) -> Dict[str, str]:
    """
    Assign target quality status and message.
    """

    messages: List[str] = []

    if row_count == 0:
        return {
            "quality_status": FAIL_STATUS,
            "quality_message": "Target has zero rows.",
            "recommended_action": "Check target generation input data.",
        }

    if unique_count == 0:
        return {
            "quality_status": FAIL_STATUS,
            "quality_message": "Target is fully null.",
            "recommended_action": "Fix target generation logic.",
        }

    if unique_count == 1:
        return {
            "quality_status": FAIL_STATUS,
            "quality_message": "Target has only one unique value.",
            "recommended_action": "Review target definition; model cannot learn from a constant target.",
        }

    if not binary_target:
        return {
            "quality_status": FAIL_STATUS,
            "quality_message": "Classification target is not binary.",
            "recommended_action": "Ensure classification target is encoded as 0/1.",
        }

    if null_count > 0:
        messages.append("Target contains null values.")

    if positive_pct is not None:
        if positive_pct < 0.01:
            messages.append("Positive class rate is below 1%; target is extremely imbalanced.")
        elif positive_pct < 0.05:
            messages.append("Positive class rate is below 5%; target is imbalanced.")
        elif positive_pct > 0.95:
            messages.append("Positive class rate is above 95%; target is imbalanced.")
        elif positive_pct > 0.99:
            messages.append("Positive class rate is above 99%; target is extremely imbalanced.")

    if imbalance_ratio is not None and imbalance_ratio >= 50:
        messages.append("Imbalance ratio is very high.")

    if messages:
        return {
            "quality_status": WARNING_STATUS,
            "quality_message": " ".join(messages),
            "recommended_action": "Review target definition, class balance, sampling strategy, and class-weight settings.",
        }

    return {
        "quality_status": PASS_STATUS,
        "quality_message": "Target passed basic quality checks.",
        "recommended_action": "Proceed with model training.",
    }


def build_target_quality_record(
    dataframe: pd.DataFrame,
    target_name: str,
    target_column: str,
    run_id: str,
    layer_name: str,
    domain_name: str,
    problem_type: str = "classification",
    report_created_utc: Optional[str] = None,
    report_type: str = DEFAULT_REPORT_TYPE,
    report_version: str = DEFAULT_REPORT_VERSION,
) -> Dict[str, Any]:
    """
    Build one target quality record.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found in dataframe: {target_column}")

    series = dataframe[target_column]

    row_count = int(len(series))
    null_count = int(series.isna().sum())
    non_null_count = int(row_count - null_count)
    unique_count = int(series.nunique(dropna=True))
    null_pct = null_count / row_count if row_count else 0.0

    binary_target = is_binary_target(series)

    positive_count: Optional[int] = None
    negative_count: Optional[int] = None
    positive_pct: Optional[float] = None
    negative_pct: Optional[float] = None
    majority_class: Optional[int] = None
    minority_class: Optional[int] = None
    majority_count: Optional[int] = None
    minority_count: Optional[int] = None
    majority_pct: Optional[float] = None
    minority_pct: Optional[float] = None
    imbalance_ratio: Optional[float] = None
    target_entropy: Optional[float] = None

    if binary_target:
        binary_series = normalize_binary_target(series).dropna()

        positive_count = int((binary_series == 1).sum())
        negative_count = int((binary_series == 0).sum())

        positive_pct = positive_count / row_count if row_count else 0.0
        negative_pct = negative_count / row_count if row_count else 0.0

        class_counts = {0: negative_count, 1: positive_count}

        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)

        majority_count = int(class_counts[majority_class])
        minority_count = int(class_counts[minority_class])

        majority_pct = majority_count / row_count if row_count else 0.0
        minority_pct = minority_count / row_count if row_count else 0.0

        imbalance_ratio = (
            majority_count / minority_count
            if minority_count and minority_count > 0
            else None
        )

        target_entropy = calculate_entropy(
            positive_pct=positive_pct,
            negative_pct=negative_pct,
        )

    assessment = assess_target_quality(
        row_count=row_count,
        null_count=null_count,
        unique_count=unique_count,
        binary_target=binary_target,
        positive_pct=positive_pct,
        imbalance_ratio=imbalance_ratio,
    )

    return {
        "run_id": run_id,
        "layer_name": layer_name,
        "domain_name": domain_name,
        "report_type": report_type,
        "report_version": report_version,
        "target_name": target_name,
        "target_column": target_column,
        "problem_type": problem_type,
        "dtype": str(series.dtype),
        "row_count": row_count,
        "non_null_count": non_null_count,
        "null_count": null_count,
        "null_pct": null_pct,
        "missing_pct": null_pct,
        "unique_count": unique_count,
        "is_binary_target": binary_target,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "majority_class": majority_class,
        "majority_count": majority_count,
        "majority_pct": majority_pct,
        "minority_class": minority_class,
        "minority_count": minority_count,
        "minority_pct": minority_pct,
        "imbalance_ratio": imbalance_ratio,
        "target_entropy": target_entropy,
        "quality_status": assessment["quality_status"],
        "quality_message": assessment["quality_message"],
        "recommended_action": assessment["recommended_action"],
        "report_created_utc": report_created_utc or utc_now_iso(),
    }


def build_target_quality_report(
    dataframe: pd.DataFrame,
    targets: List[Dict[str, Any]],
    run_id: str,
    layer_name: str,
    domain_name: str,
) -> pd.DataFrame:
    """
    Build target quality report for all configured targets.

    Expected target config item:
        {
            "target_name": "high_cost",
            "target_column": "high_cost_target",
            "problem_type": "classification"
        }
    """

    records: List[Dict[str, Any]] = []
    report_created_utc = utc_now_iso()

    for target in targets:
        target_name = target.get("target_name") or target.get("name")
        target_column = target.get("target_column")
        problem_type = target.get("problem_type", "classification")

        if not target_name:
            raise ValueError(f"Target config is missing target_name/name: {target}")

        if not target_column:
            raise ValueError(f"Target config is missing target_column: {target}")

        if target_column not in dataframe.columns:
            continue

        records.append(
            build_target_quality_record(
                dataframe=dataframe,
                target_name=target_name,
                target_column=target_column,
                run_id=run_id,
                layer_name=layer_name,
                domain_name=domain_name,
                problem_type=problem_type,
                report_created_utc=report_created_utc,
            )
        )

    return pd.DataFrame(records)


def main() -> None:
    """
    Lightweight local validation.
    """

    sample = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4, 5, 6],
            "high_cost_target": [0, 0, 0, 0, 1, 1],
            "readmission_target": [0, 0, 0, 0, 0, 1],
            "bad_constant_target": [0, 0, 0, 0, 0, 0],
            "bad_null_target": [None, None, None, None, None, None],
        }
    )

    targets = [
        {
            "target_name": "high_cost",
            "target_column": "high_cost_target",
            "problem_type": "classification",
        },
        {
            "target_name": "readmission",
            "target_column": "readmission_target",
            "problem_type": "classification",
        },
        {
            "target_name": "bad_constant",
            "target_column": "bad_constant_target",
            "problem_type": "classification",
        },
        {
            "target_name": "bad_null",
            "target_column": "bad_null_target",
            "problem_type": "classification",
        },
    ]

    output = build_target_quality_report(
        dataframe=sample,
        targets=targets,
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    print("Target quality report validation successful.")
    print(
        output[
            [
                "target_name",
                "target_column",
                "row_count",
                "positive_pct",
                "imbalance_ratio",
                "quality_status",
                "quality_message",
            ]
        ]
    )


if __name__ == "__main__":
    main()