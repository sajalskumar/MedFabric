###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/common/io.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Provides shared Analytics Platform IO helpers.
#
#     These helpers do not replace src/common StorageManager or PathManager.
#     They provide Analytics Platform-specific convenience wrappers around
#     PipelineContext path and storage services.
#
# Used By:
#     src/analytics_platform/*/build_*_layer.py
#
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.analytics_platform.common.audit import add_success_audit
from src.analytics_platform.common.runtime import (
    AnalyticsDomainRuntime,
    DEFAULT_OUTPUT_FORMAT,
)
from src.analytics_platform.common.validation import validate_required_input_dataset
from src.common.exception_manager import StorageError


###############################################################################
# Path Helpers
###############################################################################

def get_output_format(runtime: AnalyticsDomainRuntime, domain_section: str) -> str:
    """
    Purpose
    -------
    Return the configured output format for an analytics domain.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    domain_section:
        Top-level domain config section name.

    Returns
    -------
    str
        Configured output format. Defaults to parquet.

    Raises
    ------
    None

    Notes
    -----
    This helper centralizes the common pattern:
    config[domain_section]["output_format"].
    """

    return (
        runtime.config.get(domain_section, {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Purpose
    -------
    Ensure an output path has the requested file suffix.

    Parameters
    ----------
    path:
        Base output path.

    output_format:
        Output format such as parquet, csv, or json.

    Returns
    -------
    pathlib.Path
        Output path with the requested suffix.

    Raises
    ------
    None

    Notes
    -----
    YAML output paths are usually configured without a file extension. This
    helper appends the extension consistently.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


def get_configured_path(
    runtime: AnalyticsDomainRuntime,
    output_group: str,
    output_name: str,
    default_path: Optional[str] = None,
) -> Path:
    """
    Purpose
    -------
    Resolve a configured path from a domain YAML path group.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_group:
        Path group name such as outputs, metadata_outputs, or audit_outputs.

    output_name:
        Output asset key inside the path group.

    default_path:
        Optional fallback path when the output is not explicitly configured.

    Returns
    -------
    pathlib.Path
        Resolved absolute path.

    Raises
    ------
    StorageError
        Raised when no path is configured and no default path is provided.

    Notes
    -----
    This function resolves paths using Layer 0 PathManager from PipelineContext.
    """

    group_config = runtime.config.get("paths", {}).get(output_group, {})
    output_entry = group_config.get(output_name)

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        raw_path = default_path

    if not raw_path:
        raise StorageError(
            f"No configured path found for paths.{output_group}.{output_name}"
        )

    return runtime.context.paths.resolve_path(raw_path)


def get_input_path(
    runtime: AnalyticsDomainRuntime,
    dataset_name: str,
) -> Path:
    """
    Purpose
    -------
    Resolve an input dataset path from paths.inputs.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataset_name:
        Input dataset key from YAML.

    Returns
    -------
    pathlib.Path
        Resolved input path.

    Raises
    ------
    StorageError
        Raised when the input path is missing.

    Notes
    -----
    This helper assumes the domain YAML follows the standard Analytics Platform
    paths.inputs structure.
    """

    return get_configured_path(
        runtime=runtime,
        output_group="inputs",
        output_name=dataset_name,
    )


def get_output_path(
    runtime: AnalyticsDomainRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Resolve an Analytics Platform domain output path with file extension.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_name:
        Output asset name from paths.outputs.

    output_format:
        Output format.

    Returns
    -------
    pathlib.Path
        Resolved output file path.

    Raises
    ------
    StorageError
        Raised when the output path is missing.

    Notes
    -----
    This helper is for business outputs under paths.outputs.
    """

    base_path = get_configured_path(
        runtime=runtime,
        output_group="outputs",
        output_name=output_name,
        default_path=f"data/analytics_platform/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def get_metadata_output_path(
    runtime: AnalyticsDomainRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Resolve a metadata output path with file extension.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_name:
        Metadata output asset name.

    output_format:
        Output format.

    Returns
    -------
    pathlib.Path
        Resolved metadata output file path.

    Raises
    ------
    StorageError
        Raised when the path cannot be resolved.
    """

    base_path = get_configured_path(
        runtime=runtime,
        output_group="metadata_outputs",
        output_name=output_name,
        default_path=f"data/analytics_platform/metadata/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def get_audit_output_path(
    runtime: AnalyticsDomainRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Resolve an audit output path with file extension.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_name:
        Audit output asset name.

    output_format:
        Output format.

    Returns
    -------
    pathlib.Path
        Resolved audit output file path.

    Raises
    ------
    StorageError
        Raised when the path cannot be resolved.
    """

    base_path = get_configured_path(
        runtime=runtime,
        output_group="audit_outputs",
        output_name=output_name,
        default_path=f"data/analytics_platform/audit/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


###############################################################################
# Read Helpers
###############################################################################

def read_dataset(
    runtime: AnalyticsDomainRuntime,
    path: Path,
    file_format: Optional[str],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Read a dataset using StorageManager.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    path:
        Dataset path.

    file_format:
        Optional file format. If omitted, format is inferred from suffix.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    StorageError
        Raised when the format is unsupported or the dataset cannot be read.

    Notes
    -----
    This wrapper delegates actual IO to Layer 0 StorageManager.
    """

    resolved_format = file_format or path.suffix.replace(".", "").lower()

    try:
        if resolved_format == "parquet":
            return runtime.context.storage.read_parquet(path)

        if resolved_format == "csv":
            return runtime.context.storage.read_csv(path)

        if resolved_format == "json":
            data = runtime.context.storage.read_json(path)
            return pd.DataFrame(data)

    except Exception as error:
        raise StorageError(f"Failed to read dataset: {path}") from error

    raise StorageError(f"Unsupported input file format: {resolved_format}")


def load_input_datasets(
    runtime: AnalyticsDomainRuntime,
    input_source_layer_map: Optional[Dict[str, str]] = None,
    key_column_map: Optional[Dict[str, str]] = None,
    required_columns_map: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Purpose
    -------
    Load all configured input datasets for an analytics domain.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    input_source_layer_map:
        Optional mapping of input dataset name to source layer name.

    key_column_map:
        Optional mapping of input dataset name to key column requiring non-null
        validation.

    required_columns_map:
        Optional mapping of input dataset name to required columns.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Loaded input datasets keyed by dataset name.

    Raises
    ------
    StorageError
        Raised when a required input cannot be read.

    Notes
    -----
    This function performs standard read + required input validation. Optional
    missing inputs are skipped with an audit record.
    """

    logger = runtime.get_logger(f"medfabric.analytics_platform.{runtime.domain_name}")
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    source_layer_map = input_source_layer_map or {}
    key_map = key_column_map or {}
    required_map = required_columns_map or {}

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load input datasets for %s", runtime.domain_name)

    for dataset_name, dataset_config in inputs_config.items():
        raw_path = dataset_config.get("path")
        file_format = dataset_config.get("format")
        required = bool(dataset_config.get("required", True))

        source_layer = source_layer_map.get(dataset_name)
        source_dataset = dataset_name

        if not raw_path:
            if required:
                raise StorageError(f"No path configured for required input: {dataset_name}")

            continue

        input_path = runtime.context.paths.resolve_path(raw_path)

        try:
            dataframe = read_dataset(
                runtime=runtime,
                path=input_path,
                file_format=file_format,
            )

            datasets[dataset_name] = dataframe

            if required:
                validate_required_input_dataset(
                    runtime=runtime,
                    dataframe=dataframe,
                    dataset_name=dataset_name,
                    required_columns=required_map.get(dataset_name),
                    key_column=key_map.get(dataset_name),
                    source_layer=source_layer,
                    source_dataset=source_dataset,
                )

            add_success_audit(
                runtime=runtime,
                step_name=f"load_input_dataset:{dataset_name}",
                message="Input dataset loaded successfully.",
                row_count=len(dataframe),
                output_path=str(input_path),
                source_layer=source_layer,
                source_dataset=source_dataset,
            )

            logger.info(
                "Loaded input dataset: %s | Rows: %s | Columns: %s | Path: %s",
                dataset_name,
                len(dataframe),
                len(dataframe.columns),
                input_path,
            )

        except Exception as error:
            if required:
                raise StorageError(
                    f"Failed to load required input dataset: {dataset_name}"
                ) from error

            logger.warning(
                "Optional input skipped: %s | Reason: %s",
                dataset_name,
                error,
            )

    logger.info(
        "COMPLETE: Load input datasets for %s | Count: %s",
        runtime.domain_name,
        len(datasets),
    )

    return datasets


###############################################################################
# Write Helpers
###############################################################################

def write_dataset(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    path: Path,
    file_format: str,
) -> None:
    """
    Purpose
    -------
    Write a dataframe using StorageManager.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataframe:
        Dataframe to write.

    path:
        Output path.

    file_format:
        Output file format.

    Returns
    -------
    None

    Raises
    ------
    StorageError
        Raised when the output format is unsupported or writing fails.
    """

    try:
        if file_format == "parquet":
            runtime.context.storage.write_parquet(dataframe, path, index=False)
            return

        if file_format == "csv":
            runtime.context.storage.write_csv(dataframe, path, index=False)
            return

        if file_format == "json":
            runtime.context.storage.write_json(
                dataframe.to_dict(orient="records"),
                path,
            )
            return

    except Exception as error:
        raise StorageError(f"Failed to write dataset: {path}") from error

    raise StorageError(f"Unsupported output file format: {file_format}")


def write_output_asset(
    runtime: AnalyticsDomainRuntime,
    output_name: str,
    dataframe: pd.DataFrame,
    output_format: str,
) -> Path:
    """
    Purpose
    -------
    Write one configured business output asset.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_name:
        Output asset name under paths.outputs.

    dataframe:
        Output dataframe.

    output_format:
        Output format.

    Returns
    -------
    pathlib.Path
        Written output path.

    Raises
    ------
    StorageError
        Raised when writing fails.
    """

    output_path = get_output_path(
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

    add_success_audit(
        runtime=runtime,
        step_name=f"write_output:{output_name}",
        message="Analytics output written successfully.",
        row_count=len(dataframe),
        output_path=str(output_path),
    )

    return output_path


def write_output_assets(
    runtime: AnalyticsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
) -> Dict[str, Path]:
    """
    Purpose
    -------
    Write multiple configured business output assets.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_assets:
        Mapping of output asset name to dataframe.

    output_format:
        Output format.

    Returns
    -------
    dict[str, pathlib.Path]
        Written paths keyed by output asset name.

    Raises
    ------
    StorageError
        Raised when any required output cannot be written.
    """

    logger = runtime.get_logger(f"medfabric.analytics_platform.{runtime.domain_name}")
    written_paths: Dict[str, Path] = {}

    for output_name, dataframe in output_assets.items():
        output_path = write_output_asset(
            runtime=runtime,
            output_name=output_name,
            dataframe=dataframe,
            output_format=output_format,
        )

        written_paths[output_name] = output_path

        logger.info(
            "Wrote output asset: %s | Rows: %s | Columns: %s | Path: %s",
            output_name,
            len(dataframe),
            len(dataframe.columns),
            output_path,
        )

    return written_paths