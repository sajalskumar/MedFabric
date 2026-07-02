###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/common/validation.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Provides shared Analytics Platform validation helpers.
#
#     These helpers standardize validation records across Analytics Platform
#     domains and enforce fail-fast behavior for required validations.
#
# Used By:
#     src/analytics_platform/*/build_*_layer.py
#
###############################################################################

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.analytics_platform.common.runtime import (
    AnalyticsDomainRuntime,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    utc_now,
)
from src.common.exception_manager import ValidationError


def add_validation_record(
    runtime: AnalyticsDomainRuntime,
    dataset_name: str,
    rule_name: str,
    status: str,
    message: str,
    failed_count: Optional[int] = None,
    severity: str = "ERROR",
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a standardized Analytics Platform validation record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataset_name:
        Dataset being validated.

    rule_name:
        Validation rule name.

    status:
        Validation status.

    message:
        Human-readable validation message.

    failed_count:
        Optional failed record or failed item count.

    severity:
        Validation severity.

    source_layer:
        Optional upstream source layer name.

    source_dataset:
        Optional upstream source dataset name.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This function records validation results only. Fail-fast behavior is handled
    by the stricter helper functions below.
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.context.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "rule_name": rule_name,
            "status": status,
            "severity": severity,
            "message": message,
            "failed_count": failed_count,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def require_non_empty_dataframe(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    severity: str = "ERROR",
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that a dataframe is not empty.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataframe:
        Dataframe to validate.

    dataset_name:
        Dataset name for validation records.

    severity:
        Validation severity.

    source_layer:
        Optional upstream source layer name.

    source_dataset:
        Optional upstream source dataset name.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when the dataframe is empty.

    Notes
    -----
    Required input datasets and required intermediate datasets should call this
    function so downstream logic does not run against empty data.
    """

    if dataframe.empty:
        message = f"Dataset is empty: {dataset_name}"

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="dataset_not_empty",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            severity=severity,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="dataset_not_empty",
        status=STATUS_SUCCESS,
        message="Dataset is not empty.",
        failed_count=0,
        severity=severity,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def require_columns(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_columns: Iterable[str],
    severity: str = "ERROR",
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that all required columns exist.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataframe:
        Dataframe to validate.

    dataset_name:
        Dataset name for validation records.

    required_columns:
        Required column names.

    severity:
        Validation severity.

    source_layer:
        Optional upstream source layer name.

    source_dataset:
        Optional upstream source dataset name.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when one or more required columns are missing.

    Notes
    -----
    This function enforces the stricter standard discussed for Layer 2:
    required-column failures stop immediately before downstream logic touches
    missing columns.
    """

    required = [column for column in required_columns if column is not None]
    missing = [column for column in required if column not in dataframe.columns]

    if missing:
        message = (
            f"Dataset '{dataset_name}' is missing required columns: {missing}"
        )

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="required_columns_exist",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(missing),
            severity=severity,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="required_columns_exist",
        status=STATUS_SUCCESS,
        message="All required columns are present.",
        failed_count=0,
        severity=severity,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def require_key_not_null(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    key_column: str,
    severity: str = "ERROR",
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that a key column exists and does not contain nulls.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataframe:
        Dataframe to validate.

    dataset_name:
        Dataset name for validation records.

    key_column:
        Key column to validate.

    severity:
        Validation severity.

    source_layer:
        Optional upstream source layer name.

    source_dataset:
        Optional upstream source dataset name.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when the key column is missing or contains null values.

    Notes
    -----
    This helper is used for member_id, provider_id, model_key, and similar
    analytics grain keys.
    """

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=[key_column],
        severity=severity,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    null_count = int(dataframe[key_column].isna().sum())

    if null_count > 0:
        message = (
            f"Dataset '{dataset_name}' key column '{key_column}' contains "
            f"{null_count} null values."
        )

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="key_not_null",
            status=STATUS_FAILED,
            message=message,
            failed_count=null_count,
            severity=severity,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="key_not_null",
        status=STATUS_SUCCESS,
        message=f"Key column is not null: {key_column}",
        failed_count=0,
        severity=severity,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def warn_on_duplicate_key(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    key_columns: Iterable[str],
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Record a warning when duplicate key rows exist.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataframe:
        Dataframe to validate.

    dataset_name:
        Dataset name for validation records.

    key_columns:
        Columns defining the expected dataset grain.

    source_layer:
        Optional upstream source layer name.

    source_dataset:
        Optional upstream source dataset name.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This helper does not fail the pipeline. Some Analytics Platform outputs are
    intentionally member-model, member-program, or provider-rule grain and may
    have repeated member or provider keys.
    """

    keys = list(key_columns)

    try:
        require_columns(
            runtime=runtime,
            dataframe=dataframe,
            dataset_name=dataset_name,
            required_columns=keys,
            severity="WARNING",
            source_layer=source_layer,
            source_dataset=source_dataset,
        )
    except ValidationError:
        return

    duplicate_count = int(dataframe.duplicated(subset=keys).sum())

    if duplicate_count > 0:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="duplicate_key_warning",
            status=STATUS_WARNING,
            message=f"Duplicate key rows found for columns {keys}.",
            failed_count=duplicate_count,
            severity="WARNING",
            source_layer=source_layer,
            source_dataset=source_dataset,
        )
        return

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="duplicate_key_warning",
        status=STATUS_SUCCESS,
        message=f"No duplicate key rows found for columns {keys}.",
        failed_count=0,
        severity="WARNING",
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def validate_required_input_dataset(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_columns: Optional[Iterable[str]] = None,
    key_column: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Run standard required input validation for an Analytics Platform dataset.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataframe:
        Dataframe to validate.

    dataset_name:
        Dataset name.

    required_columns:
        Optional required columns.

    key_column:
        Optional key column that must be non-null.

    source_layer:
        Optional upstream source layer name.

    source_dataset:
        Optional upstream source dataset name.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when required validation fails.

    Notes
    -----
    This helper should be used immediately after reading required input datasets.
    """

    require_non_empty_dataframe(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    if required_columns:
        require_columns(
            runtime=runtime,
            dataframe=dataframe,
            dataset_name=dataset_name,
            required_columns=required_columns,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

    if key_column:
        require_key_not_null(
            runtime=runtime,
            dataframe=dataframe,
            dataset_name=dataset_name,
            key_column=key_column,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )