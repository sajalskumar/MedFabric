###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/pipeline/common/metadata.py
#
# Layer:
#     Enterprise Pipeline
#
# Purpose:
#     Provides shared metadata helpers for the MedFabric master pipeline
#     orchestrator.
#
# Business Context:
#     The Pipeline layer coordinates execution of the complete MedFabric
#     platform. Because this is the top-level enterprise workflow, every run
#     must produce metadata explaining:
#
#         - which layers executed
#         - which layers succeeded
#         - which layers failed
#         - how long execution took
#         - how many outputs were produced
#         - what configuration was used
#         - what audit and validation records were captured
#
# Architectural Rule:
#     This module contains Pipeline metadata infrastructure only.
#
#     It does NOT contain:
#         - layer execution logic
#         - data transformation logic
#         - modeling logic
#         - analytics logic
#         - reporting logic
#
# Inputs:
#     PipelineRuntime
#     PipelineBuildResult objects
#     pandas.DataFrame output assets
#
# Outputs:
#     data/pipeline/metadata/pipeline_dataset_inventory.parquet
#     data/pipeline/metadata/pipeline_column_dictionary.parquet
#     data/pipeline/metadata/pipeline_rule_catalog.parquet
#
# Used By:
#     src/pipeline/build_medfabric_platform.py
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.pipeline.common.io import write_metadata_output
from src.pipeline.common.runtime import (
    PipelineBuildResult,
    PipelineRuntime,
    STATUS_SUCCESS,
    utc_now,
)


###############################################################################
# Safe Serialization Helpers
###############################################################################

def safe_string(value: Any) -> str:
    """
    Purpose
    -------
    Convert a value to a safe string representation for metadata storage.

    Parameters
    ----------
    value:
        Any Python value.

    Returns
    -------
    str
        Safe string representation.

    Raises
    ------
    None

    Notes
    -----
    Metadata outputs may be written as parquet, csv, or json. Complex objects
    are converted to strings so the metadata schema remains stable.
    """

    if value is None:
        return ""

    return str(value)


###############################################################################
# Dataset Inventory Records
###############################################################################

def add_dataset_record(
    runtime: PipelineRuntime,
    dataset_name: str,
    dataset_type: str,
    status: str,
    path: Optional[str],
    row_count: int,
    column_count: int,
    message: str,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add one dataset inventory record to the Pipeline runtime.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    dataset_name:
        Dataset or output asset name.

    dataset_type:
        Dataset classification such as execution_summary, metadata, audit, or
        layer_result.

    status:
        Dataset status.

    path:
        Optional physical output path.

    row_count:
        Number of rows in the dataset.

    column_count:
        Number of columns in the dataset.

    message:
        Human-readable metadata message.

    source_layer:
        Optional source layer name.

    source_dataset:
        Optional source dataset or configuration artifact.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    These records are later written to pipeline_dataset_inventory.
    """

    runtime.dataset_records.append(
        {
            "run_id": runtime.run_id,
            "pipeline_name": runtime.pipeline_name,
            "layer_name": runtime.layer_name,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "status": status,
            "path": path,
            "row_count": int(row_count),
            "column_count": int(column_count),
            "message": message,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def register_output_asset(
    runtime: PipelineRuntime,
    dataset_name: str,
    dataframe: pd.DataFrame,
    dataset_type: str,
    status: str = STATUS_SUCCESS,
    path: Optional[str] = None,
    message: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Register a pipeline output dataframe in the dataset inventory.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    dataset_name:
        Output dataset name.

    dataframe:
        Output dataframe.

    dataset_type:
        Dataset type.

    status:
        Output status.

    path:
        Optional physical path.

    message:
        Optional output message.

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

    add_dataset_record(
        runtime=runtime,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        status=status,
        path=path,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        message=message or f"Pipeline output registered: {dataset_name}",
        source_layer=source_layer or runtime.layer_name,
        source_dataset=source_dataset,
    )


###############################################################################
# Rule Catalog Records
###############################################################################

def add_rule_record(
    runtime: PipelineRuntime,
    rule_group: str,
    rule_name: str,
    rule_type: str,
    description: str,
    source_dataset: str,
    rule_config: Dict[str, Any],
    source_layer: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add one pipeline rule catalog record.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    rule_group:
        Rule group or orchestration section.

    rule_name:
        Rule name.

    rule_type:
        Rule type such as layer_execution, dependency_validation, or
        platform_orchestration.

    description:
        Human-readable rule description.

    source_dataset:
        Source configuration or dataset.

    rule_config:
        Configuration dictionary for the rule.

    source_layer:
        Optional source layer.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    For the Pipeline layer, rule records primarily document orchestration,
    execution order, enabled layers, and platform-level governance behavior.
    """

    runtime.rule_records.append(
        {
            "run_id": runtime.run_id,
            "pipeline_name": runtime.pipeline_name,
            "layer_name": runtime.layer_name,
            "rule_group": rule_group,
            "rule_name": rule_name,
            "rule_type": rule_type,
            "description": description,
            "source_layer": source_layer or runtime.layer_name,
            "source_dataset": source_dataset,
            "rule_config_json": safe_string(rule_config),
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def register_layer_execution_rule(
    runtime: PipelineRuntime,
    layer_config: Dict[str, Any],
) -> None:
    """
    Purpose
    -------
    Register one configured pipeline layer as an orchestration rule.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_config:
        Layer configuration entry from medfabric_platform.yaml.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This documents the configured layer execution registry in the rule catalog.
    """

    layer_name = str(layer_config.get("name", ""))

    add_rule_record(
        runtime=runtime,
        rule_group="pipeline_layer_registry",
        rule_name=layer_name,
        rule_type="configured_layer",
        description=layer_config.get("description", ""),
        source_dataset=str(runtime.config_path),
        source_layer="Enterprise Pipeline",
        rule_config=layer_config,
    )


###############################################################################
# Layer Execution Records
###############################################################################

def add_layer_result_record(
    runtime: PipelineRuntime,
    layer_name: str,
    status: str,
    message: str,
    row_count: int = 0,
    column_count: int = 0,
    module_name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Purpose
    -------
    Add one layer execution result record to the Pipeline runtime.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_name:
        Executed layer name.

    status:
        Execution status.

    message:
        Human-readable layer execution message.

    row_count:
        Row count reported by the layer.

    column_count:
        Column count reported by the layer.

    module_name:
        Python module used to execute the layer.

    duration_seconds:
        Optional execution duration in seconds.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    These records are used to build layer_execution_summary and pipeline
    execution history outputs.
    """

    runtime.layer_results.append(
        {
            "run_id": runtime.run_id,
            "pipeline_name": runtime.pipeline_name,
            "orchestration_layer_name": runtime.layer_name,
            "layer_name": layer_name,
            "module_name": module_name,
            "status": status,
            "message": message,
            "row_count": int(row_count),
            "column_count": int(column_count),
            "duration_seconds": duration_seconds,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_layer_result_from_build_result(
    runtime: PipelineRuntime,
    result: PipelineBuildResult,
    module_name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Purpose
    -------
    Add a layer execution record from a PipelineBuildResult.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    result:
        Normalized layer build result.

    module_name:
        Module used to execute the layer.

    duration_seconds:
        Optional execution duration.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    add_layer_result_record(
        runtime=runtime,
        layer_name=result.name,
        status=result.status,
        message=result.message,
        row_count=result.row_count,
        column_count=result.column_count,
        module_name=module_name,
        duration_seconds=duration_seconds,
    )


###############################################################################
# Metadata Builders
###############################################################################

def build_dataset_inventory(runtime: PipelineRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the pipeline dataset inventory dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    pandas.DataFrame
        Dataset inventory dataframe.

    Raises
    ------
    None

    Notes
    -----
    This metadata output summarizes pipeline-level outputs and registered
    execution artifacts.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: PipelineRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the pipeline column dictionary dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_assets:
        Dictionary of pipeline output asset names to dataframes.

    Returns
    -------
    pandas.DataFrame
        Column dictionary dataframe.

    Raises
    ------
    None

    Notes
    -----
    This documents the columns produced by pipeline-level outputs such as
    execution summaries and layer execution summaries.
    """

    rows: List[Dict[str, Any]] = []

    for dataset_name, dataframe in output_assets.items():
        for column_name in dataframe.columns:
            rows.append(
                {
                    "run_id": runtime.run_id,
                    "pipeline_name": runtime.pipeline_name,
                    "layer_name": runtime.layer_name,
                    "dataset_name": dataset_name,
                    "column_name": column_name,
                    "data_type": str(dataframe[column_name].dtype),
                    "row_count": int(len(dataframe)),
                    "non_null_count": int(dataframe[column_name].notna().sum()),
                    "null_count": int(dataframe[column_name].isna().sum()),
                    "event_timestamp_utc": utc_now().isoformat(),
                }
            )

    return pd.DataFrame(rows)


def build_rule_catalog(runtime: PipelineRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the pipeline rule catalog dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    pandas.DataFrame
        Rule catalog dataframe.

    Raises
    ------
    None

    Notes
    -----
    Pipeline rule records document orchestration behavior, layer registry,
    execution order, and configured control rules.
    """

    return pd.DataFrame(runtime.rule_records)


def build_layer_execution_summary(runtime: PipelineRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the layer execution summary dataframe.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    pandas.DataFrame
        Layer execution summary dataframe.

    Raises
    ------
    None

    Notes
    -----
    This is one of the most important pipeline outputs. It provides one row per
    platform layer executed by the master orchestrator.
    """

    return pd.DataFrame(runtime.layer_results)


def build_pipeline_execution_summary(
    runtime: PipelineRuntime,
    layer_results: List[PipelineBuildResult],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build one pipeline execution summary row.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_results:
        Normalized layer build results.

    Returns
    -------
    pandas.DataFrame
        One-row pipeline execution summary.

    Raises
    ------
    None

    Notes
    -----
    The summary captures final pipeline status, total duration, layer counts,
    and runtime environment details.
    """

    end_time_utc = utc_now()
    duration_seconds = (end_time_utc - runtime.start_time_utc).total_seconds()

    failed_layer_count = sum(
        1 for result in layer_results if result.status == "FAILED"
    )

    warning_layer_count = sum(
        1 for result in layer_results if result.status == "WARNING"
    )

    skipped_layer_count = sum(
        1 for result in layer_results if result.status == "SKIPPED"
    )

    successful_layer_count = sum(
        1 for result in layer_results if result.status == "SUCCESS"
    )

    if failed_layer_count > 0:
        status = "FAILED"
    elif warning_layer_count > 0 or skipped_layer_count > 0:
        status = "WARNING"
    else:
        status = "SUCCESS"

    summary = {
        "run_id": runtime.run_id,
        "pipeline_name": runtime.pipeline_name,
        "layer_name": runtime.layer_name,
        "config_path": str(runtime.config_path),
        "platform_version": runtime.config.get("project", {}).get("platform_version"),
        "release": runtime.config.get("project", {}).get("release"),
        "start_time_utc": runtime.start_time_utc.isoformat(),
        "end_time_utc": end_time_utc.isoformat(),
        "duration_seconds": duration_seconds,
        "layer_count": len(layer_results),
        "successful_layer_count": successful_layer_count,
        "failed_layer_count": failed_layer_count,
        "warning_layer_count": warning_layer_count,
        "skipped_layer_count": skipped_layer_count,
        "total_row_count": int(sum(result.row_count for result in layer_results)),
        "total_column_count": int(sum(result.column_count for result in layer_results)),
        "audit_record_count": len(runtime.audit_records),
        "validation_record_count": len(runtime.validation_records),
        "dataset_record_count": len(runtime.dataset_records),
        "rule_record_count": len(runtime.rule_records),
        "user": runtime.user,
        "hostname": runtime.hostname,
        "python_version": runtime.python_version,
        "platform_name": runtime.platform_name,
        "status": status,
        "event_timestamp_utc": utc_now().isoformat(),
    }

    return pd.DataFrame([summary])


###############################################################################
# Metadata Output Writer
###############################################################################

def write_metadata_outputs(
    runtime: PipelineRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
    dataset_inventory_name: str = "pipeline_dataset_inventory",
    column_dictionary_name: str = "pipeline_column_dictionary",
    rule_catalog_name: str = "pipeline_rule_catalog",
) -> Dict[str, str]:
    """
    Purpose
    -------
    Build and write pipeline metadata outputs.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_assets:
        Pipeline output assets used to build the column dictionary.

    output_format:
        Output file format.

    dataset_inventory_name:
        Dataset inventory output name.

    column_dictionary_name:
        Column dictionary output name.

    rule_catalog_name:
        Rule catalog output name.

    Returns
    -------
    dict[str, str]
        Written metadata paths keyed by metadata asset name.

    Raises
    ------
    ValueError
        Raised by IO helpers if output writing fails.

    Notes
    -----
    Metadata is written after pipeline business outputs are created.
    """

    metadata_assets: Dict[str, pd.DataFrame] = {
        dataset_inventory_name: build_dataset_inventory(runtime),
        column_dictionary_name: build_column_dictionary(runtime, output_assets),
        rule_catalog_name: build_rule_catalog(runtime),
    }

    written_paths: Dict[str, str] = {}

    for output_name, dataframe in metadata_assets.items():
        output_path = write_metadata_output(
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