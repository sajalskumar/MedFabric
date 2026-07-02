###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/common/validation.py
#
# Layer:
#     Layer 3 - Insights
#
# Purpose:
#     Provides shared validation helpers for the MedFabric Insights layer.
#
# Business Context:
#     The Insights layer produces executive-ready scorecards, reporting marts,
#     operational reporting summaries, and dashboard-ready datasets.
#
#     Because these outputs are intended for business consumption, the layer
#     must validate:
#
#         - required input datasets
#         - required columns
#         - required keys
#         - null counts
#         - duplicate keys
#         - empty datasets
#         - numeric metric ranges
#         - allowed values
#         - output dataset readiness
#
# Architectural Rule:
#     This module contains reusable validation infrastructure only.
#
#     It does NOT contain:
#         - reporting metric calculations
#         - executive KPI logic
#         - domain-specific transformations
#         - dashboard logic
#
#     All Insights reporting builders should use this module for validation.
#
# Inputs:
#     pandas.DataFrame objects loaded from Layer 2 Analytics Platform outputs.
#
# Outputs:
#     Validation records stored in runtime.validation_records.
#
# Used By:
#     src/insights/common/io.py
#     src/insights/executive/build_executive_insights.py
#     src/insights/financial/build_financial_reporting.py
#     src/insights/clinical/build_clinical_reporting.py
#     src/insights/population/build_population_reporting.py
#     src/insights/provider/build_provider_reporting.py
#     src/insights/quality/build_quality_reporting.py
#     src/insights/care_management/build_care_management_reporting.py
#     src/insights/value_based_care/build_value_based_reporting.py
#     src/insights/build_insights_platform.py
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.common.exception_manager import ValidationError
from src.insights.common.runtime import (
    InsightsDomainRuntime,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    utc_now,
)


###############################################################################
# Validation Record Helpers
###############################################################################

def add_validation_record(
    runtime: InsightsDomainRuntime,
    dataset_name: str,
    rule_name: str,
    status: str,
    message: str,
    failed_count: Optional[int] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add one validation record to the Insights runtime.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataset_name:
        Dataset being validated.

    rule_name:
        Validation rule name.

    status:
        Validation status. Expected values are SUCCESS, WARNING, or FAILED.

    message:
        Human-readable validation message.

    failed_count:
        Optional number of records, columns, or checks that failed.

    source_layer:
        Optional upstream layer name for lineage.

    source_dataset:
        Optional upstream dataset name for lineage.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Validation records are written later by the audit/metadata output helpers.
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.context.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "rule_name": rule_name,
            "status": status,
            "message": message,
            "failed_count": failed_count,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


###############################################################################
# Required Dataset Validation
###############################################################################

def require_dataframe(
    runtime: InsightsDomainRuntime,
    dataframe: Optional[pd.DataFrame],
    dataset_name: str,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that an object is a pandas DataFrame.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Candidate dataframe.

    dataset_name:
        Dataset name used for validation records.

    source_layer:
        Optional upstream source layer.

    source_dataset:
        Optional upstream source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when dataframe is missing or not a pandas DataFrame.

    Notes
    -----
    This is the most basic validation rule and is used by input and output
    validation helpers.
    """

    if dataframe is None or not isinstance(dataframe, pd.DataFrame):
        message = f"Dataset '{dataset_name}' is not a pandas DataFrame."

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="is_dataframe",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="is_dataframe",
        status=STATUS_SUCCESS,
        message="Dataset is a pandas DataFrame.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def require_not_empty(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
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
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when dataframe is empty.

    Notes
    -----
    Required reporting inputs should not be empty. Optional reporting inputs may
    be skipped before this function is called.
    """

    if dataframe.empty:
        message = f"Dataset '{dataset_name}' is empty."

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="not_empty",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="not_empty",
        status=STATUS_SUCCESS,
        message="Dataset is not empty.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


###############################################################################
# Required Column and Key Validation
###############################################################################

def require_columns(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_columns: Optional[Iterable[str]],
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that required columns exist in a dataframe.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    required_columns:
        Required column names.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when one or more required columns are missing.

    Notes
    -----
    None and blank column names are ignored so configuration can omit optional
    metric columns for count_rows calculations.
    """

    if not required_columns:
        return

    required = [
        column
        for column in list(required_columns)
        if column is not None and str(column).strip() != ""
    ]

    if not required:
        return

    missing = [column for column in required if column not in dataframe.columns]

    if missing:
        message = f"Dataset '{dataset_name}' is missing required columns: {missing}"

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="required_columns",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(missing),
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="required_columns",
        status=STATUS_SUCCESS,
        message="All required columns are present.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def require_key_not_null(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    key_column: Optional[str],
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that a key column exists and contains no null values.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    key_column:
        Key column name.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when key column is missing or contains nulls.

    Notes
    -----
    Summary-level datasets should not be passed to this function unless they
    truly have a configured grain key.
    """

    if key_column is None or str(key_column).strip() == "":
        return

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=[key_column],
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
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="key_not_null",
        status=STATUS_SUCCESS,
        message=f"Key column '{key_column}' contains no null values.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def warn_duplicate_key(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    key_columns: Iterable[str],
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Record a warning when duplicate keys exist.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    key_columns:
        Key columns defining expected uniqueness.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Duplicate keys are warnings rather than hard failures because some Insights
    datasets intentionally contain multiple rows per member, provider, or rule.
    """

    keys = [column for column in key_columns if column in dataframe.columns]

    if not keys:
        return

    duplicate_count = int(dataframe.duplicated(subset=keys).sum())

    if duplicate_count > 0:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="duplicate_key_warning",
            status=STATUS_WARNING,
            message=f"Duplicate rows found for key columns: {keys}",
            failed_count=duplicate_count,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )
        return

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="duplicate_key_warning",
        status=STATUS_SUCCESS,
        message=f"No duplicate rows found for key columns: {keys}.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


###############################################################################
# Column Quality Validation
###############################################################################

def warn_null_count(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    column_name: str,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Record null-count validation information for a column.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    column_name:
        Column to check.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when the requested column is missing.

    Notes
    -----
    Nulls are recorded as warnings because reporting outputs can legitimately
    contain nulls for unavailable business measures.
    """

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=[column_name],
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    null_count = int(dataframe[column_name].isna().sum())

    status = STATUS_WARNING if null_count > 0 else STATUS_SUCCESS

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name=f"null_count:{column_name}",
        status=status,
        message=f"Column '{column_name}' null count: {null_count}.",
        failed_count=null_count,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def require_numeric_range(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    column_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that a numeric column falls within an optional range.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    column_name:
        Numeric column to validate.

    min_value:
        Optional minimum allowed value.

    max_value:
        Optional maximum allowed value.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when values fall outside the configured range.

    Notes
    -----
    This is useful for percentages, rates, score values, ranks, and count-like
    metrics used in dashboards.
    """

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=[column_name],
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    numeric_series = pd.to_numeric(dataframe[column_name], errors="coerce")

    invalid_mask = pd.Series(False, index=dataframe.index)

    if min_value is not None:
        invalid_mask = invalid_mask | (numeric_series < min_value)

    if max_value is not None:
        invalid_mask = invalid_mask | (numeric_series > max_value)

    failed_count = int(invalid_mask.fillna(False).sum())

    if failed_count > 0:
        message = (
            f"Column '{column_name}' in dataset '{dataset_name}' contains "
            f"{failed_count} values outside range "
            f"[{min_value}, {max_value}]."
        )

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name=f"numeric_range:{column_name}",
            status=STATUS_FAILED,
            message=message,
            failed_count=failed_count,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name=f"numeric_range:{column_name}",
        status=STATUS_SUCCESS,
        message=f"Column '{column_name}' passed numeric range validation.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def require_allowed_values(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    column_name: str,
    allowed_values: Iterable[Any],
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate that a column only contains configured allowed values.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Dataframe being validated.

    dataset_name:
        Dataset name.

    column_name:
        Column to validate.

    allowed_values:
        Allowed values.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when unexpected values are present.

    Notes
    -----
    Useful for business-facing categorical fields such as status, category,
    domain name, or priority label.
    """

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=[column_name],
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    allowed = set(allowed_values)

    invalid_mask = dataframe[column_name].notna() & ~dataframe[column_name].isin(allowed)
    failed_count = int(invalid_mask.sum())

    if failed_count > 0:
        invalid_values = sorted(dataframe.loc[invalid_mask, column_name].dropna().unique())

        message = (
            f"Column '{column_name}' in dataset '{dataset_name}' contains "
            f"unexpected values: {invalid_values}"
        )

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name=f"allowed_values:{column_name}",
            status=STATUS_FAILED,
            message=message,
            failed_count=failed_count,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

        raise ValidationError(message)

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name=f"allowed_values:{column_name}",
        status=STATUS_SUCCESS,
        message=f"Column '{column_name}' contains only allowed values.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


###############################################################################
# Input and Output Dataset Validation
###############################################################################

def validate_required_input_dataset(
    runtime: InsightsDomainRuntime,
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
    Validate a required input dataset.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Input dataframe.

    dataset_name:
        Dataset name.

    required_columns:
        Optional required columns.

    key_column:
        Optional key column requiring non-null validation.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when required validation checks fail.

    Notes
    -----
    This is called by src.insights.common.io.load_input_datasets() for required
    inputs.
    """

    require_dataframe(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    require_not_empty(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=required_columns,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    require_key_not_null(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        key_column=key_column,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def validate_output_dataset(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_columns: Optional[Iterable[str]] = None,
    allow_empty: bool = True,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Validate an Insights output dataset before writing.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataframe:
        Output dataframe.

    dataset_name:
        Output dataset name.

    required_columns:
        Optional required output columns.

    allow_empty:
        Whether empty outputs are allowed.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when output validation fails.

    Notes
    -----
    Some Insights outputs are allowed to be empty if upstream optional inputs are
    not available. Business-critical outputs can set allow_empty=False.
    """

    require_dataframe(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )

    if not allow_empty:
        require_not_empty(
            runtime=runtime,
            dataframe=dataframe,
            dataset_name=dataset_name,
            source_layer=source_layer,
            source_dataset=source_dataset,
        )

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=dataset_name,
        required_columns=required_columns,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def validate_output_assets(
    runtime: InsightsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    allow_empty: bool = True,
) -> None:
    """
    Purpose
    -------
    Validate a dictionary of Insights output assets.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    output_assets:
        Mapping of output dataset name to dataframe.

    allow_empty:
        Whether empty outputs are allowed.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when any output asset is invalid.

    Notes
    -----
    This helper is useful for orchestrators and domain builders before writing
    outputs to disk.
    """

    for dataset_name, dataframe in output_assets.items():
        validate_output_dataset(
            runtime=runtime,
            dataframe=dataframe,
            dataset_name=dataset_name,
            allow_empty=allow_empty,
            source_layer=runtime.layer_name,
            source_dataset=dataset_name,
        )


###############################################################################
# Cross-Dataset Validation
###############################################################################

def require_dataset_available(
    runtime: InsightsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
    dataset_name: str,
    source_layer: Optional[str] = None,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Require that a dataset exists in the loaded dataset dictionary.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    datasets:
        Loaded dataset dictionary.

    dataset_name:
        Required dataset name.

    source_layer:
        Optional source layer name.

    Returns
    -------
    pandas.DataFrame
        Requested dataframe.

    Raises
    ------
    ValidationError
        Raised when dataset is missing.

    Notes
    -----
    Reporting builders should use this instead of directly indexing the datasets
    dictionary when a dataset is required for a report.
    """

    if dataset_name not in datasets:
        message = f"Required dataset is not loaded: {dataset_name}"

        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="dataset_available",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            source_layer=source_layer,
            source_dataset=dataset_name,
        )

        raise ValidationError(message)

    dataframe = datasets[dataset_name]

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="dataset_available",
        status=STATUS_SUCCESS,
        message="Required dataset is loaded.",
        failed_count=0,
        source_layer=source_layer,
        source_dataset=dataset_name,
    )

    return dataframe


def warn_dataset_missing(
    runtime: InsightsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
    dataset_name: str,
    source_layer: Optional[str] = None,
) -> bool:
    """
    Purpose
    -------
    Record a warning when an optional dataset is not loaded.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    datasets:
        Loaded dataset dictionary.

    dataset_name:
        Optional dataset name.

    source_layer:
        Optional source layer name.

    Returns
    -------
    bool
        True if dataset is available, False if missing.

    Raises
    ------
    None

    Notes
    -----
    This is useful for executive summaries where some optional reporting domains
    may be absent.
    """

    if dataset_name in datasets:
        return True

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="optional_dataset_missing",
        status=STATUS_WARNING,
        message="Optional dataset is not loaded.",
        failed_count=1,
        source_layer=source_layer,
        source_dataset=dataset_name,
    )

    return False


###############################################################################
# End of File
###############################################################################