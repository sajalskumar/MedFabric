###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/common/rules.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Provides shared rule evaluation helpers for Analytics Platform domains.
#
# Business Context:
#     Analytics Platform domains frequently apply YAML-driven business rules:
#
#         - Population cohort thresholds
#         - Risk tier ranges
#         - Member segmentation rules
#         - Quality measure eligibility logic
#         - Provider performance thresholds
#         - Care management assignment logic
#         - Value-based care opportunity rules
#
#     This module centralizes common rule operators so each analytics domain
#     does not duplicate comparison logic.
#
# Inputs:
#     pandas.Series
#     YAML-configured operator names
#     YAML-configured comparison values
#
# Outputs:
#     pandas.Series boolean mask
#
# Used By:
#     src/analytics_platform/population_health/*
#     src/analytics_platform/clinical_analytics/*
#     src/analytics_platform/quality_analytics/*
#     src/analytics_platform/predictive_analytics/*
#     src/analytics_platform/provider_analytics/*
#     src/analytics_platform/care_management/*
#     src/analytics_platform/value_based_care/*
#
# Public Interface:
#     apply_operator()
#
###############################################################################

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.exception_manager import ValidationError


def apply_operator(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Purpose
    -------
    Apply a configured comparison operator to a pandas Series.

    Business Context
    ----------------
    MedFabric analytics domains are intentionally configuration-driven. Rules
    should be declared in YAML and evaluated consistently across domains rather
    than hardcoded inside each domain builder.

    Parameters
    ----------
    series:
        Input pandas Series to evaluate.

    operator:
        Operator name from YAML configuration.

        Supported values:
            - equals
            - not_equals
            - greater_than
            - greater_than_or_equal
            - less_than
            - less_than_or_equal
            - in
            - not_in
            - is_null
            - is_not_null

    value:
        Comparison value from YAML configuration.

    Returns
    -------
    pandas.Series
        Boolean mask that can be used to filter a dataframe.

    Raises
    ------
    ValidationError
        Raised when an unsupported operator is provided.

    Notes
    -----
    This helper intentionally returns a boolean Series instead of filtering the
    dataframe directly. That keeps calling code flexible and makes rule
    composition easier for multi-condition segmentation, quality measures, and
    care management assignment logic.
    """

    if operator == "equals":
        return series == value

    if operator == "not_equals":
        return series != value

    if operator == "greater_than":
        return series > value

    if operator == "greater_than_or_equal":
        return series >= value

    if operator == "less_than":
        return series < value

    if operator == "less_than_or_equal":
        return series <= value

    if operator == "in":
        return series.isin(value)

    if operator == "not_in":
        return ~series.isin(value)

    if operator == "is_null":
        return series.isna()

    if operator == "is_not_null":
        return series.notna()

    raise ValidationError(f"Unsupported analytics rule operator: {operator}")


def main() -> None:
    """
    Purpose
    -------
    Standalone validation entry point for shared rule helpers.

    Run
    ---
    python -m src.analytics_platform.common.rules

    Expected Output
    ---------------
    MedFabric Analytics Platform rules validation completed successfully.
    """

    sample = pd.Series([1, 2, 3, None])

    checks = {
        "equals": apply_operator(sample, "equals", 2).tolist(),
        "greater_than": apply_operator(sample, "greater_than", 1).tolist(),
        "is_null": apply_operator(sample, "is_null", None).tolist(),
    }

    if checks["equals"] != [False, True, False, False]:
        raise ValidationError("equals operator validation failed.")

    if checks["greater_than"] != [False, True, True, False]:
        raise ValidationError("greater_than operator validation failed.")

    if checks["is_null"] != [False, False, False, True]:
        raise ValidationError("is_null operator validation failed.")

    print("MedFabric Analytics Platform rules validation completed successfully.")


if __name__ == "__main__":
    main()