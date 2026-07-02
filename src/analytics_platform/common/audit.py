###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/common/audit.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Provides shared Analytics Platform audit record helpers.
#
#     This module standardizes audit records across all Analytics Platform
#     domains while still relying on PipelineContext for shared platform
#     services.
#
# Used By:
#     src/analytics_platform/*/build_*_layer.py
#
###############################################################################

from __future__ import annotations

from typing import Optional

from src.analytics_platform.common.runtime import (
    AnalyticsDomainRuntime,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    utc_now,
)


def add_audit_record(
    runtime: AnalyticsDomainRuntime,
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
    Add a standardized Analytics Platform audit record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    step_name:
        Name of the step being audited.

    status:
        Step execution status.

    message:
        Human-readable audit message.

    row_count:
        Optional number of rows associated with the audited step.

    output_path:
        Optional output path associated with the audited step.

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
    Audit records are stored in memory on the domain runtime. Output writing is
    handled separately by the domain builder or shared IO helpers.
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
    runtime: AnalyticsDomainRuntime,
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
    Add a successful audit record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    step_name:
        Step name.

    message:
        Success message.

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


def add_failed_audit(
    runtime: AnalyticsDomainRuntime,
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
    Add a failed audit record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    step_name:
        Step name.

    message:
        Failure message.

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


def add_warning_audit(
    runtime: AnalyticsDomainRuntime,
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
    Add a warning audit record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    step_name:
        Step name.

    message:
        Warning message.

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


def add_skipped_audit(
    runtime: AnalyticsDomainRuntime,
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
    Add a skipped audit record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    step_name:
        Step name.

    message:
        Skip message.

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
        status=STATUS_SKIPPED,
        message=message,
        row_count=row_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )