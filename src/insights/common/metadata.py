###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/common/metadata.py
#
# Layer:
#     Layer 3 - Insights
#
# Purpose:
#     Provides shared metadata helpers for the MedFabric Insights layer.
#
# Business Context:
#     The Insights layer produces executive-ready reporting assets from Layer 2
#     Analytics Platform outputs. These reporting assets must be governed,
#     traceable, explainable, and easy to inspect.
#
#     This module creates and manages metadata records for:
#
#         - dataset inventory
#         - column dictionary
#         - rule catalog
#         - reporting metric lineage
#         - output asset registration
#
# Architectural Rule:
#     This module contains shared Insights metadata infrastructure only.
#
#     It does NOT contain:
#         - executive KPI calculation logic
#         - reporting transformation logic
#         - dashboard rendering logic
#         - file-specific reporting logic
#
# Inputs:
#     Runtime metadata containers
#     Output pandas DataFrames
#
# Outputs:
#     data/insights/metadata/insights_dataset_inventory.parquet
#     data/insights/metadata/insights_column_dictionary.parquet
#     data/insights/metadata/insights_rule_catalog.parquet
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

from typing import Any, Dict, Optional

import pandas as pd

from src.insights.common.io import get_metadata_output_path, write_dataset
from src.insights.common.runtime import (
    InsightsDomainRuntime,
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
    Metadata outputs are written to parquet/csv/json. Complex dictionaries and
    lists are converted to strings so the metadata catalog remains stable.
    """

    if value is None:
        return ""

    return str(value)


###############################################################################
# Dataset Inventory Records
###############################################################################

def add_dataset_record(
    runtime: InsightsDomainRuntime,
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
    Add one dataset inventory record to the Insights runtime.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataset_name:
        Name of the dataset.

    dataset_type:
        Dataset classification, such as input, output, metadata, or audit.

    status:
        Dataset status.

    path:
        Optional physical dataset path.

    row_count:
        Number of rows.

    column_count:
        Number of columns.

    message:
        Human-readable dataset message.

    source_layer:
        Optional source layer used for lineage.

    source_dataset:
        Optional source dataset used for lineage.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Dataset inventory records provide a concise catalog of Insights inputs,
    outputs, and generated reporting assets.
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
            "row_count": int(row_count),
            "column_count": int(column_count),
            "message": message,
            "source_layer": source_layer,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def register_output_asset(
    runtime: InsightsDomainRuntime,
    dataset_name: str,
    dataframe: pd.DataFrame,
    dataset_type: str = "insights_output",
    status: str = STATUS_SUCCESS,
    path: Optional[str] = None,
    message: Optional[str] = None,
    source_layer: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Register an Insights output dataframe in the dataset inventory.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    dataset_name:
        Output dataset name.

    dataframe:
        Output dataframe.

    dataset_type:
        Dataset type label.

    status:
        Dataset status.

    path:
        Optional output path.

    message:
        Optional dataset message.

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
    Reporting builders should call this after creating each business output.
    """

    resolved_message = message or f"Insights output built successfully: {dataset_name}"

    add_dataset_record(
        runtime=runtime,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        status=status,
        path=path,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        message=resolved_message,
        source_layer=source_layer or runtime.layer_name,
        source_dataset=source_dataset,
    )


###############################################################################
# Rule Catalog Records
###############################################################################

def add_rule_record(
    runtime: InsightsDomainRuntime,
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
    Add one rule catalog record to the Insights runtime.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    rule_group:
        Rule group or reporting section.

    rule_name:
        Rule name.

    rule_type:
        Rule type, such as KPI, aggregation, domain status, or reporting metric.

    description:
        Business description of the rule.

    source_dataset:
        Source dataset used by the rule.

    rule_config:
        YAML configuration or rule details.

    source_layer:
        Optional upstream source layer.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Rule catalog records explain how reporting metrics and executive KPIs were
    produced. This makes dashboards traceable and auditable.
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
            "rule_config_json": safe_string(rule_config),
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_metric_rule_record(
    runtime: InsightsDomainRuntime,
    reporting_domain: str,
    metric_name: str,
    metric_config: Dict[str, Any],
    description: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add a standard metric rule record for an Insights reporting metric.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    reporting_domain:
        Reporting domain name, such as financial_reporting or provider_reporting.

    metric_name:
        Metric name.

    metric_config:
        Metric configuration from insights.yaml.

    description:
        Optional override description.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This helper standardizes metric catalog records across all reporting
    domains.
    """

    source_dataset = metric_config.get("source_dataset", "")

    add_rule_record(
        runtime=runtime,
        rule_group=reporting_domain,
        rule_name=metric_name,
        rule_type=metric_config.get("calculation_type", "reporting_metric"),
        description=description or f"Insights reporting metric: {metric_name}",
        source_dataset=source_dataset,
        source_layer="Layer 2 - Analytics Platform",
        rule_config=metric_config,
    )


###############################################################################
# Dataset Inventory Builder
###############################################################################

def build_dataset_inventory(runtime: InsightsDomainRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the Insights dataset inventory dataframe.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    Returns
    -------
    pandas.DataFrame
        Dataset inventory dataframe.

    Raises
    ------
    None

    Notes
    -----
    The dataset inventory summarizes all registered datasets for the current
    Insights run.
    """

    return pd.DataFrame(runtime.dataset_records)


###############################################################################
# Column Dictionary Builder
###############################################################################

def build_column_dictionary(
    runtime: InsightsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the Insights column dictionary dataframe.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    output_assets:
        Dictionary of output asset names to dataframes.

    Returns
    -------
    pandas.DataFrame
        Column dictionary dataframe.

    Raises
    ------
    None

    Notes
    -----
    The column dictionary documents every column produced by the Insights layer,
    including data type, null count, and non-null count.
    """

    rows: list[dict[str, Any]] = []

    for dataset_name, dataframe in output_assets.items():
        for column_name in dataframe.columns:
            rows.append(
                {
                    "run_id": runtime.context.run_id,
                    "layer_name": runtime.layer_name,
                    "domain_name": runtime.domain_name,
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


###############################################################################
# Rule Catalog Builder
###############################################################################

def build_rule_catalog(runtime: InsightsDomainRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the Insights rule catalog dataframe.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    Returns
    -------
    pandas.DataFrame
        Rule catalog dataframe.

    Raises
    ------
    None

    Notes
    -----
    The rule catalog documents KPI, reporting metric, scorecard, and summary
    rules applied during the Insights run.
    """

    return pd.DataFrame(runtime.rule_records)


###############################################################################
# Metadata Output Writer
###############################################################################

def write_metadata_outputs(
    runtime: InsightsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
    output_format: str,
    dataset_inventory_name: str = "insights_dataset_inventory",
    column_dictionary_name: str = "insights_column_dictionary",
    rule_catalog_name: str = "insights_rule_catalog",
) -> Dict[str, str]:
    """
    Purpose
    -------
    Build and write Insights metadata outputs.

    Parameters
    ----------
    runtime:
        Insights domain runtime.

    output_assets:
        Business output assets produced by an Insights build.

    output_format:
        Output file format.

    dataset_inventory_name:
        Dataset inventory output name from paths.metadata_outputs.

    column_dictionary_name:
        Column dictionary output name from paths.metadata_outputs.

    rule_catalog_name:
        Rule catalog output name from paths.metadata_outputs.

    Returns
    -------
    dict[str, str]
        Written metadata output paths keyed by metadata asset name.

    Raises
    ------
    StorageError
        Raised by IO helpers if writing fails.

    Notes
    -----
    Metadata is written after business outputs are produced. This function keeps
    metadata generation consistent across every Insights domain and the
    platform orchestrator.
    """

    metadata_assets: Dict[str, pd.DataFrame] = {
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

    return written_paths


###############################################################################
# End of File
###############################################################################