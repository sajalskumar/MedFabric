###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/common/io.py
#
# Layer:
#     Layer 3 - Insights
#
# Purpose:
#     Provides shared input/output helpers for the MedFabric Insights layer.
#
# Business Context:
#     The Insights layer consumes completed Layer 2 Analytics Platform outputs
#     and writes executive-ready reporting datasets, scorecards, and dashboard
#     marts under data/insights/.
#
# Architectural Rule:
#     This module contains shared Insights IO infrastructure only.
#
#     IMPORTANT:
#     This module must NOT import src.insights.common.audit because audit.py
#     imports IO helpers to write audit outputs. Importing audit from IO creates
#     a circular import.
#
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.common.exception_manager import StorageError
from src.insights.common.runtime import InsightsDomainRuntime, STATUS_SUCCESS, utc_now
from src.insights.common.validation import validate_required_input_dataset


DEFAULT_OUTPUT_FORMAT = "parquet"


def add_io_audit_record(
    runtime: InsightsDomainRuntime,
    step_name: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Add an IO audit record without importing the audit module.

    This avoids circular imports between io.py and audit.py.
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.context.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "step_name": step_name,
            "status": STATUS_SUCCESS,
            "message": message,
            "row_count": row_count,
            "output_path": output_path,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def get_output_format(
    runtime: InsightsDomainRuntime,
    domain_section: str,
) -> str:
    """
    Return the configured output format for an Insights domain.
    """

    return (
        runtime.config.get(domain_section, {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Ensure an output path has the requested file extension.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


def get_configured_path(
    runtime: InsightsDomainRuntime,
    output_group: str,
    output_name: str,
    default_path: Optional[str] = None,
) -> Path:
    """
    Resolve a configured path from the Insights YAML path group.
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
    runtime: InsightsDomainRuntime,
    dataset_name: str,
) -> Path:
    """
    Resolve an input dataset path from paths.inputs.
    """

    return get_configured_path(
        runtime=runtime,
        output_group="inputs",
        output_name=dataset_name,
    )


def get_output_path(
    runtime: InsightsDomainRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Resolve an Insights business output path with file extension.
    """

    base_path = get_configured_path(
        runtime=runtime,
        output_group="outputs",
        output_name=output_name,
        default_path=f"data/insights/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def get_metadata_output_path(
    runtime: InsightsDomainRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Resolve an Insights metadata output path with file extension.
    """

    base_path = get_configured_path(
        runtime=runtime,
        output_group="metadata_outputs",
        output_name=output_name,
        default_path=f"data/insights/metadata/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def get_audit_output_path(
    runtime: InsightsDomainRuntime,
    output_name: str,
    output_format: str,
) -> Path:
    """
    Resolve an Insights audit output path with file extension.
    """

    base_path = get_configured_path(
        runtime=runtime,
        output_group="audit_outputs",
        output_name=output_name,
        default_path=f"data/insights/audit/{output_name}",
    )

    return output_path_with_format(base_path, output_format)


def read_dataset(
    runtime: InsightsDomainRuntime,
    path: Path,
    file_format: Optional[str],
) -> pd.DataFrame:
    """
    Read a dataset using the shared MedFabric StorageManager.
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
    runtime: InsightsDomainRuntime,
    input_source_layer_map: Optional[Dict[str, str]] = None,
    key_column_map: Optional[Dict[str, str]] = None,
    required_columns_map: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all configured input datasets for an Insights domain.
    """

    logger = runtime.get_logger(f"medfabric.insights.{runtime.domain_name}")
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

        source_layer = source_layer_map.get(dataset_name, "Layer 2 - Analytics Platform")
        source_dataset = dataset_name

        if not raw_path:
            if required:
                raise StorageError(
                    f"No path configured for required input: {dataset_name}"
                )

            logger.warning(
                "Optional input skipped because no path is configured: %s",
                dataset_name,
            )
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

            add_io_audit_record(
                runtime=runtime,
                step_name=f"load_input_dataset:{dataset_name}",
                message="Insights input dataset loaded successfully.",
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


def write_dataset(
    runtime: InsightsDomainRuntime,
    dataframe: pd.DataFrame,
    path: Path,
    file_format: str,
) -> None:
    """
    Write a dataframe using the shared MedFabric StorageManager.
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
    runtime: InsightsDomainRuntime,
    output_name: str,
    dataframe: pd.DataFrame,
    output_format: str,
) -> Path:
    """
    Write one configured Insights business output asset.
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

    add_io_audit_record(
        runtime=runtime,
        step_name=f"write_output:{output_name}",
        message="Insights output written successfully.",
        row_count=len(dataframe),
        output_path=str(output_path),
        source_layer=runtime.layer_name,
        source_dataset=output_name,
    )

    return output_path


def write_output_assets(
    runtime: InsightsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
) -> Dict[str, Path]:
    """
    Write multiple configured Insights business output assets.
    """

    logger = runtime.get_logger(f"medfabric.insights.{runtime.domain_name}")
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
            "Wrote Insights output asset: %s | Rows: %s | Columns: %s | Path: %s",
            output_name,
            len(dataframe),
            len(dataframe.columns),
            output_path,
        )

    return written_paths


###############################################################################
# End of File
###############################################################################