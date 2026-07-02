###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/targets/target_validation.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Validates generated model targets for correctness, distribution quality,
#     and modeling readiness.
#
# Run:
#     python -m src.modeling.targets.target_validation
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


STATUS_SUCCESS = "SUCCESS"
STATUS_WARNING = "WARNING"
STATUS_FAILED = "FAILED"


@dataclass
class TargetValidationResult:
    """
    Standard result produced by TargetValidator.
    """

    model_key: str
    target_column: str
    status: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_record(self, run_id: str, layer_name: str, domain_name: str) -> Dict[str, Any]:
        """
        Convert validation result to metadata/audit record.
        """

        return {
            "run_id": run_id,
            "layer_name": layer_name,
            "domain_name": domain_name,
            "model_key": self.model_key,
            "target_column": self.target_column,
            "status": self.status,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": str(self.errors),
            "warnings": str(self.warnings),
            "statistics": str(self.statistics),
            "event_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }


class TargetValidator:
    """
    Validates one generated target column.

    Validation checks:
        - Target exists
        - Target is not empty
        - Target has no nulls
        - Target is binary
        - Target has positive and negative examples
        - Positive rate is not dangerously imbalanced
    """

    def __init__(
        self,
        model_key: str,
        target_column: str,
        target_series: pd.Series,
        minimum_positive_count: int = 1,
        minimum_negative_count: int = 1,
        minimum_positive_rate: float = 0.01,
        maximum_positive_rate: float = 0.99,
    ) -> None:
        self.model_key = model_key
        self.target_column = target_column
        self.target_series = target_series
        self.minimum_positive_count = minimum_positive_count
        self.minimum_negative_count = minimum_negative_count
        self.minimum_positive_rate = minimum_positive_rate
        self.maximum_positive_rate = maximum_positive_rate
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.statistics: Dict[str, Any] = {}

    def validate(self) -> TargetValidationResult:
        """
        Run all target validation checks.
        """

        self._validate_not_empty()
        self._calculate_statistics()
        self._validate_no_nulls()
        self._validate_binary_values()
        self._validate_class_presence()
        self._validate_class_balance()

        passed = len(self.errors) == 0
        status = STATUS_FAILED if self.errors else STATUS_WARNING if self.warnings else STATUS_SUCCESS

        return TargetValidationResult(
            model_key=self.model_key,
            target_column=self.target_column,
            status=status,
            passed=passed,
            errors=self.errors,
            warnings=self.warnings,
            statistics=self.statistics,
        )

    def _validate_not_empty(self) -> None:
        """
        Validate target series is not empty.
        """

        if self.target_series is None or len(self.target_series) == 0:
            self.errors.append("Target series is empty.")

    def _calculate_statistics(self) -> None:
        """
        Calculate target distribution statistics.
        """

        if self.target_series is None or len(self.target_series) == 0:
            self.statistics = {
                "row_count": 0,
                "null_count": None,
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": 0.0,
                "unique_values": [],
            }
            return

        clean_series = self.target_series.dropna()

        row_count = int(len(self.target_series))
        null_count = int(self.target_series.isna().sum())
        positive_count = int((clean_series == 1).sum())
        negative_count = int((clean_series == 0).sum())
        positive_rate = positive_count / row_count if row_count else 0.0

        self.statistics = {
            "row_count": row_count,
            "null_count": null_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
            "unique_values": sorted([str(value) for value in clean_series.unique().tolist()]),
        }

    def _validate_no_nulls(self) -> None:
        """
        Validate target has no nulls.
        """

        null_count = int(self.statistics.get("null_count") or 0)

        if null_count > 0:
            self.errors.append(f"Target contains null values: {null_count}")

    def _validate_binary_values(self) -> None:
        """
        Validate target contains only binary values 0 and 1.
        """

        if self.target_series is None or len(self.target_series) == 0:
            return

        allowed_values = {0, 1}
        observed_values = set(self.target_series.dropna().astype(int).unique().tolist())

        invalid_values = observed_values - allowed_values

        if invalid_values:
            self.errors.append(f"Target contains non-binary values: {sorted(invalid_values)}")

    def _validate_class_presence(self) -> None:
        """
        Validate target has enough positive and negative examples.
        """

        positive_count = int(self.statistics.get("positive_count", 0))
        negative_count = int(self.statistics.get("negative_count", 0))

        if positive_count < self.minimum_positive_count:
            self.errors.append(
                f"Target has insufficient positive examples: {positive_count}"
            )

        if negative_count < self.minimum_negative_count:
            self.errors.append(
                f"Target has insufficient negative examples: {negative_count}"
            )

    def _validate_class_balance(self) -> None:
        """
        Warn when target class distribution is highly imbalanced.
        """

        positive_rate = float(self.statistics.get("positive_rate", 0.0))

        if positive_rate < self.minimum_positive_rate:
            self.warnings.append(
                f"Target positive rate is very low: {positive_rate:.6f}"
            )

        if positive_rate > self.maximum_positive_rate:
            self.warnings.append(
                f"Target positive rate is very high: {positive_rate:.6f}"
            )


def validate_target(
    model_key: str,
    target_column: str,
    target_series: pd.Series,
) -> TargetValidationResult:
    """
    Convenience function for validating one target.
    """

    validator = TargetValidator(
        model_key=model_key,
        target_column=target_column,
        target_series=target_series,
    )

    return validator.validate()


def main() -> None:
    """
    Lightweight module validation.
    """

    sample_target = pd.Series([0, 0, 0, 1, 1], name="high_cost_target")

    result = validate_target(
        model_key="high_cost",
        target_column="high_cost_target",
        target_series=sample_target,
    )

    print("Target validation module validation successful.")
    print(f"Status: {result.status}")
    print(f"Passed: {result.passed}")
    print(f"Statistics: {result.statistics}")


if __name__ == "__main__":
    main()