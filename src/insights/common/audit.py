###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/common/audit.py
#
# Layer:
#     Layer 3 - Insights
#
# Purpose:
#     Provides shared audit and execution-summary helpers for the MedFabric
#     Insights layer.
#
# Business Context:
#     The Insights layer produces executive-ready reporting outputs. These
#     outputs must be auditable so users can understand:
#
#         - when the reporting layer ran
#         - which steps completed
#         - which steps failed
#         - how many records were produced
#         - which validations were recorded
#         - how long the execution took
#
# Architectural Rule:
#     This module contains shared Insights audit infrastructure only.
#
#     It does NOT contain:
#         - reporting metric calculations
#         - executive KPI logic
#         - dashboard logic
#         - domain-specific transformations
#
# Inputs:
#     InsightsDomainRuntime
#     Runtime audit records
#     Runtime validation records
#     Output pandas DataFrames
#
# Outputs:
#     data/insights/audit/insights_audit_records.parquet
#     data/insights/audit/insights_validation_results.parquet
#     data/insights/audit/insights_execution_summary.parquet
#
# Used By:
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

from typing import Dict, Optional

import pandas as pd

from src.insights.common.io import get_audit_output_path, write_dataset
from src.insights.common.runtime import (
    InsightsDomainRuntime,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    utc_now,
)


###############################################################################
# Audit Record Helpers
###############################################################################

def add_audit_record(
    runtime: InsightsDomainRuntime,
    step_name: str,
    status: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add one audit record to the Insights runtime.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    step_name:
        Name of the processing step.

    status:
        Step status. Expected values are SUCCESS, WARNING, FAILED, or SKIPPED.

    message:
        Human-readable audit message.

    row_count:
        Optional row count associated with the step.

    output_path:
        Optional output path written by the step.

    source_layer:
        Optional upstream source layer used by the step.

    source_dataset:
        Optional upstream source dataset used by the step.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Audit records are kept in memory during execution and written at the end of
    the Insights run.
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.context.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "step_name": step_name,
            "status": status,
            "message": message,
            "row_count": row_count,
            "output_path": output_path,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_success_audit(
    runtime: InsightsDomainRuntime,
    step_name: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a SUCCESS audit record.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    step_name:
        Name of the completed step.

    message:
        Human-readable success message.

    row_count:
        Optional row count.

    output_path:
        Optional output path.

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
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_SUCCESS,
        message=message,
        row_count=row_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def add_warning_audit(
    runtime: InsightsDomainRuntime,
    step_name: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a WARNING audit record.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    step_name:
        Name of the warning step.

    message:
        Human-readable warning message.

    row_count:
        Optional row count.

    output_path:
        Optional output path.

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
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_WARNING,
        message=message,
        row_count=row_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def add_failed_audit(
    runtime: InsightsDomainRuntime,
    step_name: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a FAILED audit record.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    step_name:
        Name of the failed step.

    message:
        Human-readable failure message.

    row_count:
        Optional row count.

    output_path:
        Optional output path.

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
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_FAILED,
        message=message,
        row_count=row_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


###############################################################################
# Audit Output Builders
###############################################################################

def build_audit_records(runtime: InsightsDomainRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the Insights audit records dataframe.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    Returns
    -------
    pandas.DataFrame
        Audit records dataframe.

    Raises
    ------
    None

    Notes
    -----
    If no audit records were collected, an empty dataframe is returned.
    """

    return pd.DataFrame(runtime.audit_records)


def build_validation_results(runtime: InsightsDomainRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the Insights validation results dataframe.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    Returns
    -------
    pandas.DataFrame
        Validation results dataframe.

    Raises
    ------
    None

    Notes
    -----
    Validation records are generated by src.insights.common.validation.
    """

    return pd.DataFrame(runtime.validation_records)


def build_execution_summary(
    runtime: InsightsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build one execution summary row for the Insights run.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    output_assets:
        Business output assets produced by the run.

    Returns
    -------
    pandas.DataFrame
        Execution summary dataframe.

    Raises
    ------
    None

    Notes
    -----
    The summary captures run-level totals useful for audit review, platform
    monitoring, and final documentation.
    """

    end_time = utc_now()

    start_time = getattr(runtime.context, "start_time", None)

    if start_time is not None:
        try:
            duration_seconds = (end_time - start_time).total_seconds()
        except Exception:
            duration_seconds = None
    else:
        duration_seconds = None

    failed_validation_count = sum(
        1
        for record in runtime.validation_records
        if record.get("status") == STATUS_FAILED
    )

    warning_validation_count = sum(
        1
        for record in runtime.validation_records
        if record.get("status") == STATUS_WARNING
    )

    failed_audit_count = sum(
        1
        for record in runtime.audit_records
        if record.get("status") == STATUS_FAILED
    )

    total_output_rows = sum(len(dataframe) for dataframe in output_assets.values())
    total_output_columns = sum(
        len(dataframe.columns) for dataframe in output_assets.values()
    )

    if failed_audit_count > 0 or failed_validation_count > 0:
        overall_status = STATUS_FAILED
    elif warning_validation_count > 0:
        overall_status = STATUS_WARNING
    else:
        overall_status = STATUS_SUCCESS

    summary = {
        "run_id": runtime.context.run_id,
        "layer_name": runtime.layer_name,
        "domain_name": runtime.domain_name,
        "config_file": runtime.config_file,
        "end_time_utc": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "output_asset_count": len(output_assets),
        "total_output_rows": int(total_output_rows),
        "total_output_columns": int(total_output_columns),
        "audit_record_count": len(runtime.audit_records),
        "failed_audit_count": int(failed_audit_count),
        "validation_record_count": len(runtime.validation_records),
        "failed_validation_count": int(failed_validation_count),
        "warning_validation_count": int(warning_validation_count),
        "dataset_record_count": len(runtime.dataset_records),
        "rule_record_count": len(runtime.rule_records),
        "status": overall_status,
        "event_timestamp_utc": utc_now().isoformat(),
    }

    return pd.DataFrame([summary])


###############################################################################
# Audit Output Writer
###############################################################################

def write_audit_outputs(
    runtime: InsightsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
    audit_records_name: str = "insights_audit_records",
    validation_results_name: str = "insights_validation_results",
    execution_summary_name: str = "insights_execution_summary",
) -> Dict[str, str]:
    """
    Purpose
    -------
    Build and write Insights audit outputs.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    output_assets:
        Business output assets produced by the run.

    output_format:
        Output file format.

    audit_records_name:
        Audit records output name from paths.audit_outputs.

    validation_results_name:
        Validation results output name from paths.audit_outputs.

    execution_summary_name:
        Execution summary output name from paths.audit_outputs.

    Returns
    -------
    dict[str, str]
        Written audit output paths keyed by audit asset name.

    Raises
    ------
    StorageError
        Raised by IO helpers if writing fails.

    Notes
    -----
    Audit outputs are written at the end of every Insights domain run and by the
    Insights platform orchestrator.
    """

    audit_assets: Dict[str, pd.DataFrame] = {
        audit_records_name: build_audit_records(runtime),
        validation_results_name: build_validation_results(runtime),
        execution_summary_name: build_execution_summary(
            runtime=runtime,
            output_assets=output_assets,
        ),
    }

    written_paths: Dict[str, str] = {}

    for output_name, dataframe in audit_assets.items():
        output_path = get_audit_output_path(
            runtime=runtime,
            output_name=output_name,
            output_format=output_format,
        )

        write_dataset(
            runtime=runtime,
            dataframe=dataframe,
            path=output_path,
            file_format=output_format,
        )

        written_paths[output_name] = str(output_path)

    return written_paths


###############################################################################
# End of File
###############################################################################