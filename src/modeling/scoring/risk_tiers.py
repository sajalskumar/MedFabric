###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/scoring/risk_tiers.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Assigns configured risk tier labels to model probability scores.
#
# Run:
#     python -m src.modeling.scoring.risk_tiers
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def assign_risk_tiers(
    scores: pd.Series,
    risk_tiers_config: Dict[str, Any],
) -> pd.Series:
    """
    Assign configured risk tier labels to probability scores.
    """

    output = pd.Series("Unassigned", index=scores.index)

    for _, tier_config in risk_tiers_config.items():
        min_value = float(tier_config.get("min_value", 0.0))
        max_value = float(tier_config.get("max_value", 1.0))
        label = tier_config.get("label", "Unassigned")

        mask = (scores >= min_value) & (scores <= max_value)
        output.loc[mask] = label

    return output


def main() -> None:
    scores = pd.Series([0.10, 0.45, 0.75, 0.90])

    risk_tiers_config = {
        "very_high": {"min_value": 0.85, "max_value": 1.00, "label": "Very High"},
        "high": {"min_value": 0.70, "max_value": 0.849999, "label": "High"},
        "moderate": {"min_value": 0.40, "max_value": 0.699999, "label": "Moderate"},
        "low": {"min_value": 0.00, "max_value": 0.399999, "label": "Low"},
    }

    tiers = assign_risk_tiers(scores, risk_tiers_config)

    print("Risk tier validation successful.")
    print(tiers.tolist())


if __name__ == "__main__":
    main()