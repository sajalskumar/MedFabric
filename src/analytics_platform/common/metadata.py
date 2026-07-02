###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/common/metadata.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Provides shared Analytics Platform metadata helpers.
#
#     These helpers standardize dataset inventory, column dictionary, rule
#     catalog, and execution summary outputs across Analytics Platform domains.
#
# Used By:
#     src/analytics_platform/*/build_*_layer.py
#
###############################################################################

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.analytics_platform.common.audit import add_success_audit
from src.analytics_platform.common.io import (
    get_audit_output_path,
    get_metadata_output_path,
    write_dataset,
)
from src.analytics_platform.common.runtime import (
    AnalyticsDomainRuntime,
    STATUS_FAILED,
    STATUS_SUCCESS,
    utc_now,
)


###############################################################################
# Record Helpers
###############################################################################

def add_dataset_record(
    runtime: AnalyticsDomainRuntime,
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
    Add a standardized Analytics Platform dataset inventory record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    dataset_name:
        Dataset name.

    dataset_type:
        Dataset type such as input, analytics_output, metadata_output, or
        audit_output.

    status:
        Dataset status.

    path:
        Optional dataset path.

    row_count:
        Dataset row count.

    column_count:
        Dataset column count.

    message:
        Dataset inventory message.

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

    runtime.dataset_records.append(
        {
            "run_id": runtime.context.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "status": status,
            "path": path,
            "row_count": row_count,
            "column_count": column_count,
            "message": message,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_rule_record(
    runtime: AnalyticsDomainRuntime,
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
    Add a standardized Analytics Platform rule catalog record.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    rule_group:
        Rule group name.

    rule_name:
        Rule name.

    rule_type:
        Rule type.

    description:
        Human-readable rule description.

    source_dataset:
        Source dataset used by the rule.

    rule_config:
        YAML rule configuration.

    source_layer:
        Optional source layer.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    runtime.rule_records.append(
        {
            "run_id": runtime.context.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "rule_group": rule_group,
            "rule_name": rule_name,
            "rule_type": rule_type,
            "description": description,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "rule_config_json": str(rule_config),
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


###############################################################################
# Metadata Builders
###############################################################################

def build_dataset_inventory(runtime: AnalyticsDomainRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build dataset inventory dataframe from runtime records.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    Returns
    -------
    pandas.DataFrame
        Dataset inventory.

    Raises
    ------
    None
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: AnalyticsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build column dictionary for Analytics Platform outputs.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_assets:
        Mapping of output asset name to dataframe.

    Returns
    -------
    pandas.DataFrame
        Column dictionary.

    Raises
    ------
    None

    Notes
    -----
    Includes explicit source_layer and source_dataset columns to satisfy the
    Layer 2 architecture metadata standard.
    """

    rows: list[dict[str, Any]] = []

    for dataset_name, dataframe in output_assets.items():
        for column in dataframe.columns:
            rows.append(
                {
                    "run_id": runtime.context.run_id,
                    "layer_name": runtime.layer_name,
                    "domain_name": runtime.domain_name,
                    "dataset_name": dataset_name,
                    "column_name": column,
                    "data_type": str(dataframe[column].dtype),
                    "non_null_count": int(dataframe[column].notna().sum()),
                    "null_count": int(dataframe[column].isna().sum()),
                    "row_count": int(len(dataframe)),
                    "source_layer": runtime.layer_name,
                    "source_dataset": dataset_name,
                    "event_timestamp_utc": utc_now().isoformat(),
                }
            )

    return pd.DataFrame(rows)


def build_rule_catalog(runtime: AnalyticsDomainRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build rule catalog dataframe from runtime records.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    Returns
    -------
    pandas.DataFrame
        Rule catalog.

    Raises
    ------
    None
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: AnalyticsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build one-row execution summary for an Analytics Platform domain.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_assets:
        Output asset mapping.

    Returns
    -------
    pandas.DataFrame
        Execution summary.

    Raises
    ------
    None
    """

    end_time = utc_now()
    duration_seconds = (end_time - runtime.start_time_utc).total_seconds()

    failed_validation_count = sum(
        1
        for record in runtime.validation_records
        if record.get("status") == STATUS_FAILED
    )

    summary = {
        "run_id": runtime.context.run_id,
        "layer_name": runtime.layer_name,
        "domain_name": runtime.domain_name,
        "config_file": runtime.config_file,
        "source_layer": "Layer 2 - Analytics Platform",
        "source_dataset": runtime.config_file,
        "start_time_utc": runtime.start_time_utc.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "output_asset_count": len(output_assets),
        "audit_record_count": len(runtime.audit_records),
        "validation_record_count": len(runtime.validation_records),
        "failed_validation_count": failed_validation_count,
        "dataset_record_count": len(runtime.dataset_records),
        "rule_record_count": len(runtime.rule_records),
        "status": STATUS_SUCCESS if failed_validation_count == 0 else "WARNING",
    }

    return pd.DataFrame([summary])


###############################################################################
# Metadata and Audit Writers
###############################################################################

def write_metadata_outputs(
    runtime: AnalyticsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
    dataset_inventory_name: str,
    column_dictionary_name: str,
    rule_catalog_name: str,
) -> Dict[str, str]:
    """
    Purpose
    -------
    Write standard Analytics Platform metadata outputs.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_assets:
        Output asset mapping.

    output_format:
        Output format.

    dataset_inventory_name:
        Configured metadata output key for dataset inventory.

    column_dictionary_name:
        Configured metadata output key for column dictionary.

    rule_catalog_name:
        Configured metadata output key for rule catalog.

    Returns
    -------
    dict[str, str]
        Written metadata paths keyed by metadata asset name.

    Raises
    ------
    StorageError
        Raised by IO helpers if writing fails.
    """

    metadata_assets = {
        dataset_inventory_name: build_dataset_inventory(runtime),
        column_dictionary_name: build_column_dictionary(runtime, output_assets),
        rule_catalog_name: build_rule_catalog(runtime),
    }

    written_paths: Dict[str, str] = {}

    for output_name, dataframe in metadata_assets.items():
        output_path = get_metadata_output_path(
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

        add_success_audit(
            runtime=runtime,
            step_name=f"write_metadata:{output_name}",
            message="Metadata output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
            source_layer=runtime.layer_name,
            source_dataset=output_name,
        )

    return written_paths


def write_audit_outputs(
    runtime: AnalyticsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
    audit_records_name: str,
    validation_results_name: str,
    execution_summary_name: str,
) -> Dict[str, str]:
    """
    Purpose
    -------
    Write standard Analytics Platform audit outputs.

    Parameters
    ----------
    runtime:
        Analytics domain runtime.

    output_assets:
        Output asset mapping.

    output_format:
        Output format.

    audit_records_name:
        Configured audit output key for audit records.

    validation_results_name:
        Configured audit output key for validation results.

    execution_summary_name:
        Configured audit output key for execution summary.

    Returns
    -------
    dict[str, str]
        Written audit paths keyed by audit asset name.

    Raises
    ------
    StorageError
        Raised by IO helpers if writing fails.
    """

    audit_assets = {
        audit_records_name: pd.DataFrame(runtime.audit_records),
        validation_results_name: pd.DataFrame(runtime.validation_records),
        execution_summary_name: build_execution_summary(runtime, output_assets),
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