###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/pipeline/common/audit.py
#
# Layer:
#     Enterprise Pipeline
#
# Purpose:
#     Provides shared audit helpers for the MedFabric master pipeline
#     orchestrator.
#
# Business Context:
#     The Pipeline layer coordinates the entire MedFabric platform. Because it
#     is the top-level execution entry point, it must capture a complete audit
#     trail for:
#
#         - pipeline start
#         - pipeline completion
#         - layer start
#         - layer completion
#         - skipped layers
#         - failed layers
#         - warning conditions
#         - validation outcomes
#         - execution history
#
# Architectural Rule:
#     This module contains Pipeline audit infrastructure only.
#
#     It does NOT contain:
#         - layer execution logic
#         - data transformation logic
#         - feature engineering logic
#         - modeling logic
#         - analytics logic
#         - reporting logic
#
# Inputs:
#     PipelineRuntime
#     PipelineBuildResult objects
#     Runtime audit records
#     Runtime validation records
#     Runtime layer execution records
#
# Outputs:
#     data/pipeline/audit/pipeline_audit_records.parquet
#     data/pipeline/audit/pipeline_validation_results.parquet
#     data/pipeline/audit/pipeline_execution_history.parquet
#
# Used By:
#     src/pipeline/build_medfabric_platform.py
#
###############################################################################

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.pipeline.common.io import write_audit_output
from src.pipeline.common.metadata import build_pipeline_execution_summary
from src.pipeline.common.runtime import (
    PipelineBuildResult,
    PipelineRuntime,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    utc_now,
)


###############################################################################
# Audit Record Helpers
###############################################################################

def add_audit_record(
    runtime: PipelineRuntime,
    step_name: str,
    status: str,
    message: str,
    layer_name: Optional[str] = None,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
    output_path: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add one audit record to the Pipeline runtime.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    step_name:
        Name of the pipeline or layer step.

    status:
        Step status. Expected values are SUCCESS, FAILED, WARNING, SKIPPED, or
        RUNNING.

    message:
        Human-readable audit message.

    layer_name:
        Optional platform layer associated with this audit event.

    row_count:
        Optional row count associated with the step.

    column_count:
        Optional column count associated with the step.

    output_path:
        Optional output path associated with the step.

    source_layer:
        Optional upstream source layer.

    source_dataset:
        Optional source dataset or configuration file.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Audit records are retained in memory during execution and written at the end
    of the master pipeline run.
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.run_id,
            "pipeline_name": runtime.pipeline_name,
            "orchestration_layer_name": runtime.layer_name,
            "layer_name": layer_name or runtime.layer_name,
            "step_name": step_name,
            "status": status,
            "message": message,
            "row_count": row_count,
            "column_count": column_count,
            "output_path": output_path,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_success_audit(
    runtime: PipelineRuntime,
    step_name: str,
    message: str,
    layer_name: Optional[str] = None,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
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
        Pipeline runtime.

    step_name:
        Completed step name.

    message:
        Success message.

    layer_name:
        Optional associated layer.

    row_count:
        Optional row count.

    column_count:
        Optional column count.

    output_path:
        Optional output path.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_SUCCESS,
        message=message,
        layer_name=layer_name,
        row_count=row_count,
        column_count=column_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def add_failed_audit(
    runtime: PipelineRuntime,
    step_name: str,
    message: str,
    layer_name: Optional[str] = None,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
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
        Pipeline runtime.

    step_name:
        Failed step name.

    message:
        Failure message.

    layer_name:
        Optional associated layer.

    row_count:
        Optional row count.

    column_count:
        Optional column count.

    output_path:
        Optional output path.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_FAILED,
        message=message,
        layer_name=layer_name,
        row_count=row_count,
        column_count=column_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def add_warning_audit(
    runtime: PipelineRuntime,
    step_name: str,
    message: str,
    layer_name: Optional[str] = None,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
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
        Pipeline runtime.

    step_name:
        Warning step name.

    message:
        Warning message.

    layer_name:
        Optional associated layer.

    row_count:
        Optional row count.

    column_count:
        Optional column count.

    output_path:
        Optional output path.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_WARNING,
        message=message,
        layer_name=layer_name,
        row_count=row_count,
        column_count=column_count,
        output_path=output_path,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def add_skipped_audit(
    runtime: PipelineRuntime,
    step_name: str,
    message: str,
    layer_name: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a SKIPPED audit record.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    step_name:
        Skipped step name.

    message:
        Skipped message.

    layer_name:
        Optional associated layer.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_SKIPPED,
        message=message,
        layer_name=layer_name,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


def add_running_audit(
    runtime: PipelineRuntime,
    step_name: str,
    message: str,
    layer_name: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a RUNNING audit record.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    step_name:
        Running step name.

    message:
        Running message.

    layer_name:
        Optional associated layer.

    source_layer:
        Optional source layer.

    source_dataset:
        Optional source dataset.

    Returns
    -------
    None
    """

    add_audit_record(
        runtime=runtime,
        step_name=step_name,
        status=STATUS_RUNNING,
        message=message,
        layer_name=layer_name,
        source_layer=source_layer,
        source_dataset=source_dataset,
    )


###############################################################################
# Layer Audit Helpers
###############################################################################

def add_layer_start_audit(
    runtime: PipelineRuntime,
    layer_name: str,
    module_name: str,
) -> None:
    """
    Purpose
    -------
    Add a layer-start audit record.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_name:
        Name of the platform layer starting execution.

    module_name:
        Python module configured for the layer.

    Returns
    -------
    None
    """

    add_running_audit(
        runtime=runtime,
        step_name=f"start_layer:{layer_name}",
        message=f"Started layer execution: {layer_name}",
        layer_name=layer_name,
        source_layer="Enterprise Pipeline",
        source_dataset=module_name,
    )


def add_layer_completion_audit(
    runtime: PipelineRuntime,
    layer_name: str,
    module_name: str,
    result: PipelineBuildResult,
) -> None:
    """
    Purpose
    -------
    Add a layer-completion audit record using a normalized build result.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_name:
        Configured platform layer name.

    module_name:
        Python module configured for the layer.

    result:
        Normalized layer execution result.

    Returns
    -------
    None
    """

    if result.status == STATUS_SUCCESS:
        add_success_audit(
            runtime=runtime,
            step_name=f"complete_layer:{layer_name}",
            message=result.message,
            layer_name=layer_name,
            row_count=result.row_count,
            column_count=result.column_count,
            source_layer="Enterprise Pipeline",
            source_dataset=module_name,
        )
        return

    if result.status == STATUS_WARNING:
        add_warning_audit(
            runtime=runtime,
            step_name=f"complete_layer:{layer_name}",
            message=result.message,
            layer_name=layer_name,
            row_count=result.row_count,
            column_count=result.column_count,
            source_layer="Enterprise Pipeline",
            source_dataset=module_name,
        )
        return

    if result.status == STATUS_SKIPPED:
        add_skipped_audit(
            runtime=runtime,
            step_name=f"skip_layer:{layer_name}",
            message=result.message,
            layer_name=layer_name,
            source_layer="Enterprise Pipeline",
            source_dataset=module_name,
        )
        return

    add_failed_audit(
        runtime=runtime,
        step_name=f"fail_layer:{layer_name}",
        message=result.message,
        layer_name=layer_name,
        row_count=result.row_count,
        column_count=result.column_count,
        source_layer="Enterprise Pipeline",
        source_dataset=module_name,
    )


###############################################################################
# Audit Output Builders
###############################################################################

def build_audit_records(runtime: PipelineRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the pipeline audit records dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    pandas.DataFrame
        Audit records dataframe.

    Raises
    ------
    None

    Notes
    -----
    Audit records are generated throughout pipeline execution.
    """

    return pd.DataFrame(runtime.audit_records)


def build_validation_results(runtime: PipelineRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the pipeline validation results dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    pandas.DataFrame
        Validation results dataframe.

    Raises
    ------
    None

    Notes
    -----
    Validation records are generated by src.pipeline.common.validation.
    """

    return pd.DataFrame(runtime.validation_records)


def build_execution_history(
    runtime: PipelineRuntime,
    layer_results: list[PipelineBuildResult],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the pipeline execution history dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_results:
        Normalized layer execution results.

    Returns
    -------
    pandas.DataFrame
        Pipeline execution history dataframe.

    Raises
    ------
    None

    Notes
    -----
    This combines the one-row pipeline execution summary with environment and
    runtime context. It is intended for long-term run history tracking.
    """

    return build_pipeline_execution_summary(
        runtime=runtime,
        layer_results=layer_results,
    )


###############################################################################
# Audit Output Writer
###############################################################################

def write_audit_outputs(
    runtime: PipelineRuntime,
    layer_results: list[PipelineBuildResult],
    output_format: str,
    audit_records_name: str = "pipeline_audit_records",
    validation_results_name: str = "pipeline_validation_results",
    execution_history_name: str = "pipeline_execution_history",
) -> Dict[str, str]:
    """
    Purpose
    -------
    Build and write pipeline audit outputs.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_results:
        Normalized layer build results.

    output_format:
        Output file format.

    audit_records_name:
        Audit records output asset name.

    validation_results_name:
        Validation results output asset name.

    execution_history_name:
        Execution history output asset name.

    Returns
    -------
    dict[str, str]
        Written audit output paths keyed by audit asset name.

    Raises
    ------
    ValueError
        Raised by IO helpers if writing fails.

    Notes
    -----
    This function is called at the end of the master pipeline. It should also be
    called in exception handling so audit history is preserved when the pipeline
    fails.
    """

    audit_assets: Dict[str, pd.DataFrame] = {
        audit_records_name: build_audit_records(runtime),
        validation_results_name: build_validation_results(runtime),
        execution_history_name: build_execution_history(
            runtime=runtime,
            layer_results=layer_results,
        ),
    }

    written_paths: Dict[str, str] = {}

    for output_name, dataframe in audit_assets.items():
        output_path = write_audit_output(
            runtime=runtime,
            output_name=output_name,
            dataframe=dataframe,
            output_format=output_format,
        )

        written_paths[output_name] = str(output_path)

    return written_paths


###############################################################################
# End of File
###############################################################################