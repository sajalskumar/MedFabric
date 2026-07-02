###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/pipeline/common/io.py
#
# Layer:
#     Enterprise Pipeline
#
# Purpose:
#     Provides shared input/output helpers for the MedFabric master pipeline
#     orchestrator.
#
# Business Context:
#     The Pipeline layer coordinates execution of the complete MedFabric
#     platform. It produces orchestration-level outputs such as:
#
#         - pipeline execution summary
#         - layer execution summary
#         - pipeline dataset inventory
#         - pipeline column dictionary
#         - pipeline rule catalog
#         - pipeline audit records
#         - pipeline validation results
#         - pipeline execution history
#
# Architectural Rule:
#     This module contains Pipeline IO infrastructure only.
#
#     It does NOT contain:
#         - data transformation logic
#         - modeling logic
#         - analytics logic
#         - reporting logic
#         - layer-specific business rules
#
# Inputs:
#     config/pipeline/medfabric_platform.yaml
#
# Outputs:
#     data/pipeline/execution_summary/
#     data/pipeline/metadata/
#     data/pipeline/audit/
#
# Used By:
#     src/pipeline/build_medfabric_platform.py
#     src/pipeline/common/metadata.py
#     src/pipeline/common/audit.py
#
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.pipeline.common.runtime import (
    DEFAULT_OUTPUT_FORMAT,
    PipelineRuntime,
    normalize_path,
)


###############################################################################
# Output Format Helpers
###############################################################################

def get_output_format(runtime: PipelineRuntime) -> str:
    """
    Purpose
    -------
    Return the configured pipeline output format.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    str
        Output format. Defaults to parquet.

    Raises
    ------
    None

    Notes
    -----
    The master pipeline currently uses parquet by default. If an output_format
    key is later added to config/pipeline/medfabric_platform.yaml, this helper
    will use it automatically.
    """

    return (
        runtime.config.get("pipeline", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Purpose
    -------
    Ensure an output path has the configured file extension.

    Parameters
    ----------
    path:
        Base output path.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Output path with the correct extension.

    Raises
    ------
    None

    Notes
    -----
    Pipeline YAML paths are normally configured without file extensions.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


###############################################################################
# Directory Helpers
###############################################################################

def ensure_directory(path: Path) -> None:
    """
    Purpose
    -------
    Ensure a directory exists.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    None

    Raises
    ------
    OSError
        Raised by pathlib when directory creation fails.

    Notes
    -----
    Used before writing pipeline metadata, audit, and execution outputs.
    """

    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_directory(path: Path) -> None:
    """
    Purpose
    -------
    Ensure the parent directory of a file path exists.

    Parameters
    ----------
    path:
        File path.

    Returns
    -------
    None

    Raises
    ------
    OSError
        Raised by pathlib when directory creation fails.
    """

    ensure_directory(path.parent)


###############################################################################
# Configured Path Helpers
###############################################################################

def get_path_entry(
    runtime: PipelineRuntime,
    output_group: str,
    output_name: str,
) -> Optional[Any]:
    """
    Purpose
    -------
    Return a configured path entry from the pipeline YAML.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_group:
        Path group such as outputs, metadata_outputs, or audit_outputs.

    output_name:
        Output asset key.

    Returns
    -------
    Any
        Configured path entry or None.

    Raises
    ------
    None

    Notes
    -----
    Path entries may be strings or dictionaries containing a path key.
    """

    return (
        runtime.config
        .get("paths", {})
        .get(output_group, {})
        .get(output_name)
    )


def resolve_configured_output_path(
    runtime: PipelineRuntime,
    output_group: str,
    output_name: str,
    default_path: Optional[str] = None,
) -> Path:
    """
    Purpose
    -------
    Resolve an output path from config/pipeline/medfabric_platform.yaml.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_group:
        Path group name.

    output_name:
        Output asset name.

    default_path:
        Optional fallback path.

    Returns
    -------
    pathlib.Path
        Resolved output path.

    Raises
    ------
    ValueError
        Raised when no configured path or default path is available.

    Notes
    -----
    This function resolves paths relative to the MedFabric project root.
    """

    output_entry = get_path_entry(
        runtime=runtime,
        output_group=output_group,
        output_name=output_name,
    )

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        raw_path = default_path

    if not raw_path:
        raise ValueError(
            f"No configured path found for paths.{output_group}.{output_name}"
        )

    return normalize_path(runtime.project_root, raw_path)


def get_business_output_path(
    runtime: PipelineRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Resolve a pipeline business output path.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_name:
        Output asset name under paths.outputs.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Resolved output file path.

    Raises
    ------
    ValueError
        Raised when output path cannot be resolved.
    """

    base_path = resolve_configured_output_path(
        runtime=runtime,
        output_group="outputs",
        output_name=output_name,
        default_path=f"data/pipeline/execution_summary/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def get_metadata_output_path(
    runtime: PipelineRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Resolve a pipeline metadata output path.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_name:
        Metadata output asset name under paths.metadata_outputs.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Resolved metadata output file path.
    """

    base_path = resolve_configured_output_path(
        runtime=runtime,
        output_group="metadata_outputs",
        output_name=output_name,
        default_path=f"data/pipeline/metadata/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def get_audit_output_path(
    runtime: PipelineRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Resolve a pipeline audit output path.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_name:
        Audit output asset name under paths.audit_outputs.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Resolved audit output file path.
    """

    base_path = resolve_configured_output_path(
        runtime=runtime,
        output_group="audit_outputs",
        output_name=output_name,
        default_path=f"data/pipeline/audit/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


###############################################################################
# Dataset Writers
###############################################################################

def write_dataset(
    dataframe: pd.DataFrame,
    path: Path,
    file_format: str,
) -> None:
    """
    Purpose
    -------
    Write a dataframe to disk.

    Parameters
    ----------
    dataframe:
        Dataframe to write.

    path:
        Output file path.

    file_format:
        Output file format.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when output format is unsupported.

    Notes
    -----
    The Pipeline layer uses direct pandas writes because the Pipeline runtime is
    intentionally independent of lower-layer PipelineContext storage managers.
    """

    ensure_parent_directory(path)

    if file_format == "parquet":
        dataframe.to_parquet(path, index=False)
        return

    if file_format == "csv":
        dataframe.to_csv(path, index=False)
        return

    if file_format == "json":
        dataframe.to_json(path, orient="records", indent=2)
        return

    raise ValueError(f"Unsupported pipeline output format: {file_format}")


def write_business_output(
    runtime: PipelineRuntime,
    output_name: str,
    dataframe: pd.DataFrame,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Write one pipeline business output.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_name:
        Output asset name under paths.outputs.

    dataframe:
        Dataframe to write.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Written output path.

    Raises
    ------
    ValueError
        Raised when path resolution or writing fails.
    """

    output_path = get_business_output_path(
        runtime=runtime,
        output_name=output_name,
        output_format=output_format,
    )

    write_dataset(
        dataframe=dataframe,
        path=output_path,
        file_format=output_format,
    )

    runtime.logger.info(
        "Wrote pipeline output: %s | Rows: %s | Columns: %s | Path: %s",
        output_name,
        len(dataframe),
        len(dataframe.columns),
        output_path,
    )

    return output_path


def write_business_outputs(
    runtime: PipelineRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
) -> Dict[str, Path]:
    """
    Purpose
    -------
    Write multiple pipeline business outputs.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_assets:
        Mapping of output asset names to dataframes.

    output_format:
        Output file format.

    Returns
    -------
    dict[str, pathlib.Path]
        Written output paths.

    Raises
    ------
    ValueError
        Raised when any output fails to write.
    """

    written_paths: Dict[str, Path] = {}

    for output_name, dataframe in output_assets.items():
        written_paths[output_name] = write_business_output(
            runtime=runtime,
            output_name=output_name,
            dataframe=dataframe,
            output_format=output_format,
        )

    return written_paths


def write_metadata_output(
    runtime: PipelineRuntime,
    output_name: str,
    dataframe: pd.DataFrame,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Write one pipeline metadata output.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_name:
        Metadata output name.

    dataframe:
        Metadata dataframe.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Written output path.
    """

    output_path = get_metadata_output_path(
        runtime=runtime,
        output_name=output_name,
        output_format=output_format,
    )

    write_dataset(
        dataframe=dataframe,
        path=output_path,
        file_format=output_format,
    )

    runtime.logger.info(
        "Wrote pipeline metadata output: %s | Rows: %s | Path: %s",
        output_name,
        len(dataframe),
        output_path,
    )

    return output_path


def write_audit_output(
    runtime: PipelineRuntime,
    output_name: str,
    dataframe: pd.DataFrame,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Write one pipeline audit output.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    output_name:
        Audit output name.

    dataframe:
        Audit dataframe.

    output_format:
        Output file format.

    Returns
    -------
    pathlib.Path
        Written output path.
    """

    output_path = get_audit_output_path(
        runtime=runtime,
        output_name=output_name,
        output_format=output_format,
    )

    write_dataset(
        dataframe=dataframe,
        path=output_path,
        file_format=output_format,
    )

    runtime.logger.info(
        "Wrote pipeline audit output: %s | Rows: %s | Path: %s",
        output_name,
        len(dataframe),
        output_path,
    )

    return output_path


###############################################################################
# Output Directory Preparation
###############################################################################

def prepare_pipeline_output_directories(runtime: PipelineRuntime) -> None:
    """
    Purpose
    -------
    Create standard Pipeline output directories before execution.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    None

    Raises
    ------
    OSError
        Raised when directory creation fails.

    Notes
    -----
    This prepares the top-level data/pipeline directory structure used by the
    master orchestrator.
    """

    directories = [
        runtime.project_root / "data/pipeline",
        runtime.project_root / "data/pipeline/execution_summary",
        runtime.project_root / "data/pipeline/metadata",
        runtime.project_root / "data/pipeline/audit",
        runtime.project_root / "logs/pipeline",
    ]

    for directory in directories:
        ensure_directory(directory)

    runtime.logger.info("Pipeline output directories prepared successfully.")


###############################################################################
# End of File
###############################################################################