###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/provider_analytics/build_provider_analytics_layer.py
#
# Layer:
#     Layer 2F - Provider Analytics
#
# Purpose:
#     Builds Provider Analytics outputs using the shared Analytics Platform
#     common framework.
#
# Architectural Rule:
#     Provider Analytics consumes curated provider performance and attribution
#     outputs from upstream Gold / Analytics datasets.
#
#     This file does NOT generate raw data.
#     This file does NOT build Silver dimensional models.
#     This file does NOT train or score predictive models.
#
# Dependency Flow:
#     Data Platform / Gold Layer
#         ↓
#     Provider Performance / Attribution Outputs
#         ↓
#     Layer 2F - Provider Analytics
#         ↓
#     data/analytics_platform/provider_analytics/
#
# Architecture:
#     This file contains Provider Analytics business logic only.
#
#     Shared Analytics Platform concerns are handled by:
#         - src.analytics_platform.common.runtime
#         - src.analytics_platform.common.io
#         - src.analytics_platform.common.audit
#         - src.analytics_platform.common.validation
#         - src.analytics_platform.common.metadata
#
# Inputs:
#     config/analytics_platform/provider_analytics.yaml
#
# Outputs:
#     data/analytics_platform/provider_analytics/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.provider_analytics.build_provider_analytics_layer
#
###############################################################################

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analytics_platform.common.audit import (
    add_failed_audit,
    add_success_audit,
)
from src.analytics_platform.common.io import (
    get_output_format,
    load_input_datasets,
    write_output_assets,
)
from src.analytics_platform.common.metadata import (
    add_dataset_record,
    add_rule_record,
    write_audit_outputs,
    write_metadata_outputs,
)
from src.analytics_platform.common.rules import apply_operator
from src.analytics_platform.common.runtime import (
    AnalyticsBuildResult,
    AnalyticsDomainRuntime,
    STATUS_FAILED,
    STATUS_SUCCESS,
    create_domain_runtime,
    normalize_config_file,
    utc_now,
)
from src.analytics_platform.common.validation import (
    require_columns,
    require_key_not_null,
)
from src.common.exception_manager import PipelineError, ValidationError
from src.common.pipeline_context import create_pipeline_context


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/analytics_platform/provider_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2F - Provider Analytics"
DEFAULT_DOMAIN_NAME = "Provider Analytics"
DOMAIN_SECTION = "provider_analytics"

LOGGER_NAME = "medfabric.analytics_platform.provider_analytics"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Provider Analytics configuration sections.

    Parameters
    ----------
    config:
        Loaded Provider Analytics YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required configuration sections are missing.

    Notes
    -----
    YAML loading is handled by ConfigurationManager through PipelineContext.
    This function validates only the Provider Analytics domain contract.
    """

    required_sections = [
        "provider_analytics",
        "paths",
        "join_keys",
        "provider_performance_summary",
        "provider_network_summary",
        "provider_specialty_summary",
        "provider_cost_summary",
        "provider_utilization_summary",
        "high_performing_providers",
        "validation",
        "metadata",
        "audit",
    ]

    missing_sections = [
        section for section in required_sections if section not in config
    ]

    paths_config = config.get("paths", {})

    for subsection in ["inputs", "outputs", "metadata_outputs", "audit_outputs"]:
        if subsection not in paths_config:
            missing_sections.append(f"paths.{subsection}")

    if missing_sections:
        raise PipelineError(
            "Provider Analytics configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Provider Analytics runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Provider Analytics configuration path.

    Returns
    -------
    AnalyticsDomainRuntime
        Initialized Provider Analytics runtime.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This follows the same runtime pattern as the working Predictive Analytics
    layer:
        - normalize config path
        - create PipelineContext using create_pipeline_context()
        - load YAML through context.configuration
        - validate domain contract
        - create AnalyticsDomainRuntime through create_domain_runtime()
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2F - Provider Analytics"
    )

    config = context.configuration.load_yaml(config_file)
    validate_config(config)

    runtime = create_domain_runtime(
        context=context,
        config=config,
        config_file=config_file,
        domain_section=DOMAIN_SECTION,
        default_layer_name=DEFAULT_LAYER_NAME,
        default_domain_name=DEFAULT_DOMAIN_NAME,
    )

    add_success_audit(
        runtime=runtime,
        step_name="initialize_runtime",
        message="Provider Analytics runtime initialized successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Calculation Helpers
###############################################################################

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Purpose
    -------
    Safely divide two numeric pandas Series.

    Parameters
    ----------
    numerator:
        Numeric numerator Series.

    denominator:
        Numeric denominator Series.

    Returns
    -------
    pandas.Series
        Division result with zero denominators handled safely.

    Raises
    ------
    None

    Notes
    -----
    Zero denominators are treated as null before division. Missing results are
    returned as zero so downstream analytics outputs do not contain infinite or
    null ratio values.
    """

    clean_denominator = denominator.replace({0: pd.NA})
    result = numerator / clean_denominator

    return result.fillna(0)


def calculate_group_metric(
    dataframe: pd.DataFrame,
    group_by: List[str],
    metric_name: str,
    metric_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Calculate one configured grouped provider metric.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    group_by:
        Columns defining the grouped output grain.

    metric_name:
        Output metric column name.

    metric_config:
        YAML metric configuration.

    Returns
    -------
    pandas.DataFrame
        Grouped metric dataframe containing group_by columns and metric_name.

    Raises
    ------
    ValidationError
        Raised when metric configuration is invalid or required columns are
        missing.

    Notes
    -----
    Supported calculation types:
        - count_rows
        - count_distinct
        - sum
        - mean
        - sum_boolean
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

    if calculation_type == "count_rows":
        return dataframe.groupby(group_by).size().reset_index(name=metric_name)

    if not column:
        raise ValidationError(
            f"Metric '{metric_name}' requires a configured column. "
            f"Calculation type: {calculation_type}"
        )

    if column not in dataframe.columns:
        raise ValidationError(
            f"Metric column missing for '{metric_name}': {column}"
        )

    if calculation_type == "count_distinct":
        return (
            dataframe.groupby(group_by)[column]
            .nunique(dropna=True)
            .reset_index(name=metric_name)
        )

    if calculation_type == "sum":
        return (
            dataframe.groupby(group_by)[column]
            .sum()
            .reset_index(name=metric_name)
        )

    if calculation_type == "mean":
        return (
            dataframe.groupby(group_by)[column]
            .mean()
            .reset_index(name=metric_name)
        )

    if calculation_type == "sum_boolean":
        return (
            dataframe.groupby(group_by)[column]
            .apply(lambda values: values.fillna(False).astype(bool).sum())
            .reset_index(name=metric_name)
        )

    raise ValidationError(
        f"Unsupported provider metric calculation_type: {calculation_type}"
    )


def build_group_summary(
    runtime: AnalyticsDomainRuntime,
    dataframe: pd.DataFrame,
    summary_config: Dict[str, Any],
    output_name: str,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build a configurable grouped provider summary.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    dataframe:
        Source dataframe for grouped analytics.

    summary_config:
        YAML configuration for the grouped summary.

    output_name:
        Name of the output analytics asset.

    Returns
    -------
    pandas.DataFrame
        Grouped provider analytics summary.

    Raises
    ------
    ValidationError
        Raised when group_by columns or metric configuration are invalid.

    Notes
    -----
    This shared helper supports provider network summaries, provider specialty
    summaries, and future provider grouped analytics assets.
    """

    source_dataset = summary_config.get("source_dataset", "")
    group_by = summary_config.get("group_by", [])
    metrics = summary_config.get("metrics", {})

    if not group_by:
        raise ValidationError(f"group_by is required for output: {output_name}")

    require_columns(
        runtime=runtime,
        dataframe=dataframe,
        dataset_name=source_dataset,
        required_columns=group_by,
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    output_df = dataframe[group_by].drop_duplicates().copy()

    for metric_name, metric_config in metrics.items():
        add_rule_record(
            runtime=runtime,
            rule_group=output_name,
            rule_name=metric_name,
            rule_type=metric_config.get("calculation_type", ""),
            description=metric_config.get(
                "description",
                f"Provider grouped metric: {metric_name}",
            ),
            source_dataset=source_dataset,
            source_layer="Gold Layer",
            rule_config=metric_config,
        )

        metric_df = calculate_group_metric(
            dataframe=dataframe,
            group_by=group_by,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        output_df = output_df.merge(metric_df, on=group_by, how="left")

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = output_name
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = source_dataset
    output_df["built_at_utc"] = utc_now().isoformat()

    return output_df


###############################################################################
# Provider Performance Summary
###############################################################################

def build_provider_performance_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider-level performance summary.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Provider-level performance summary.

    Raises
    ------
    ValidationError
        Raised when the configured source dataset or required provider columns
        are missing.

    Notes
    -----
    Output grain is one row per provider.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_performance_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider performance summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance")
    provider_key = config.get("provider_key", "provider_id")

    if source_dataset not in datasets:
        raise ValidationError(
            f"Provider performance source dataset missing: {source_dataset}"
        )

    source_df = datasets[source_dataset]

    required_columns = config.get("required_columns", [])
    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=required_columns,
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    require_key_not_null(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        key_column=provider_key,
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    output_columns = [
        column
        for column in config.get("output_columns", [])
        if column in source_df.columns
    ]

    if provider_key not in output_columns:
        output_columns.insert(0, provider_key)

    output_df = source_df[output_columns].drop_duplicates(subset=[provider_key]).copy()

    for metric_name, metric_config in config.get("calculated_metrics", {}).items():
        numerator = metric_config.get("numerator")
        denominator = metric_config.get("denominator")

        require_columns(
            runtime=runtime,
            dataframe=output_df,
            dataset_name="provider_performance_summary",
            required_columns=[numerator, denominator],
            source_layer="Gold Layer",
            source_dataset=source_dataset,
        )

        output_df[metric_name] = safe_divide(
            output_df[numerator],
            output_df[denominator],
        )

        add_rule_record(
            runtime=runtime,
            rule_group="provider_performance_summary",
            rule_name=metric_name,
            rule_type="ratio",
            description=metric_config.get("description", ""),
            source_dataset=source_dataset,
            source_layer="Gold Layer",
            rule_config=metric_config,
        )

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "provider_performance_summary"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = source_dataset
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_performance_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider performance summary built successfully.",
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build provider performance summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Network and Specialty Summaries
###############################################################################

def build_provider_network_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider network summary.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Provider network summary.

    Raises
    ------
    ValidationError
        Raised when configured source dataset is missing.

    Notes
    -----
    Output grain is one row per configured network grouping.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_network_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider network summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance")

    if source_dataset not in datasets:
        raise ValidationError(
            f"Provider network source dataset missing: {source_dataset}"
        )

    output_df = build_group_summary(
        runtime=runtime,
        dataframe=datasets[source_dataset],
        summary_config=config,
        output_name="provider_network_summary",
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_network_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider network summary built successfully.",
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build provider network summary | Rows: %s", len(output_df))

    return output_df


def build_provider_specialty_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider specialty summary.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Provider specialty summary.

    Raises
    ------
    ValidationError
        Raised when configured source dataset is missing.

    Notes
    -----
    Output grain is one row per configured specialty grouping.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_specialty_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider specialty summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance")

    if source_dataset not in datasets:
        raise ValidationError(
            f"Provider specialty source dataset missing: {source_dataset}"
        )

    output_df = build_group_summary(
        runtime=runtime,
        dataframe=datasets[source_dataset],
        summary_config=config,
        output_name="provider_specialty_summary",
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_specialty_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider specialty summary built successfully.",
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build provider specialty summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Cost and Utilization Summaries
###############################################################################

def build_provider_cost_summary(
    runtime: AnalyticsDomainRuntime,
    provider_performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider cost summary.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    provider_performance_summary:
        Provider-level performance summary.

    Returns
    -------
    pandas.DataFrame
        Provider cost summary.

    Raises
    ------
    ValidationError
        Raised when required cost columns are missing.

    Notes
    -----
    Output grain is one row per provider.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_cost_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider cost summary disabled")
        return pd.DataFrame()

    required_columns = config.get("required_columns", [])

    require_columns(
        runtime=runtime,
        dataframe=provider_performance_summary,
        dataset_name="provider_performance_summary",
        required_columns=required_columns,
        source_layer=runtime.layer_name,
        source_dataset="provider_performance_summary",
    )

    output_df = provider_performance_summary[required_columns].copy()

    for metric_name, metric_config in config.get("cost_metrics", {}).items():
        numerator = metric_config.get("numerator")
        denominator = metric_config.get("denominator")

        require_columns(
            runtime=runtime,
            dataframe=output_df,
            dataset_name="provider_cost_summary",
            required_columns=[numerator, denominator],
            source_layer=runtime.layer_name,
            source_dataset="provider_performance_summary",
        )

        output_df[metric_name] = safe_divide(
            output_df[numerator],
            output_df[denominator],
        )

        add_rule_record(
            runtime=runtime,
            rule_group="provider_cost_summary",
            rule_name=metric_name,
            rule_type="ratio",
            description=metric_config.get(
                "description",
                f"Provider cost metric: {metric_name}",
            ),
            source_dataset="provider_performance_summary",
            source_layer=runtime.layer_name,
            rule_config=metric_config,
        )

    ranking = config.get("ranking", {})
    rank_column = ranking.get("rank_column")

    if rank_column:
        require_columns(
            runtime=runtime,
            dataframe=output_df,
            dataset_name="provider_cost_summary",
            required_columns=[rank_column],
            source_layer=runtime.layer_name,
            source_dataset="provider_performance_summary",
        )

        rank_method = ranking.get("rank_method", "descending")
        output_rank_column = ranking.get("output_rank_column", "provider_cost_rank")
        ascending = rank_method != "descending"

        output_df[output_rank_column] = (
            output_df[rank_column]
            .rank(method="dense", ascending=ascending)
            .astype(int)
        )

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "provider_cost_summary"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = "provider_performance_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_cost_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider cost summary built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="provider_performance_summary",
    )

    logger.info("COMPLETE: Build provider cost summary | Rows: %s", len(output_df))

    return output_df


def build_provider_utilization_summary(
    runtime: AnalyticsDomainRuntime,
    provider_performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider utilization summary.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    provider_performance_summary:
        Provider-level performance summary.

    Returns
    -------
    pandas.DataFrame
        Provider utilization summary.

    Raises
    ------
    ValidationError
        Raised when required utilization columns are missing.

    Notes
    -----
    Output grain is one row per provider.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_utilization_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider utilization summary disabled")
        return pd.DataFrame()

    required_columns = config.get("required_columns", [])

    require_columns(
        runtime=runtime,
        dataframe=provider_performance_summary,
        dataset_name="provider_performance_summary",
        required_columns=required_columns,
        source_layer=runtime.layer_name,
        source_dataset="provider_performance_summary",
    )

    output_df = provider_performance_summary[required_columns].copy()

    for metric_name, metric_config in config.get("utilization_metrics", {}).items():
        numerator = metric_config.get("numerator")
        denominator = metric_config.get("denominator")

        require_columns(
            runtime=runtime,
            dataframe=output_df,
            dataset_name="provider_utilization_summary",
            required_columns=[numerator, denominator],
            source_layer=runtime.layer_name,
            source_dataset="provider_performance_summary",
        )

        output_df[metric_name] = safe_divide(
            output_df[numerator],
            output_df[denominator],
        )

        add_rule_record(
            runtime=runtime,
            rule_group="provider_utilization_summary",
            rule_name=metric_name,
            rule_type="ratio",
            description=metric_config.get(
                "description",
                f"Provider utilization metric: {metric_name}",
            ),
            source_dataset="provider_performance_summary",
            source_layer=runtime.layer_name,
            rule_config=metric_config,
        )

    ranking = config.get("ranking", {})
    rank_column = ranking.get("rank_column")

    if rank_column:
        require_columns(
            runtime=runtime,
            dataframe=output_df,
            dataset_name="provider_utilization_summary",
            required_columns=[rank_column],
            source_layer=runtime.layer_name,
            source_dataset="provider_performance_summary",
        )

        rank_method = ranking.get("rank_method", "descending")
        output_rank_column = ranking.get(
            "output_rank_column",
            "provider_utilization_rank",
        )
        ascending = rank_method != "descending"

        output_df[output_rank_column] = (
            output_df[rank_column]
            .rank(method="dense", ascending=ascending)
            .astype(int)
        )

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "provider_utilization_summary"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = "provider_performance_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_utilization_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider utilization summary built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="provider_performance_summary",
    )

    logger.info("COMPLETE: Build provider utilization summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# High Performing Provider Registry
###############################################################################

def build_high_performing_providers(
    runtime: AnalyticsDomainRuntime,
    provider_performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build high-performing provider registry.

    Parameters
    ----------
    runtime:
        Provider Analytics runtime.

    provider_performance_summary:
        Provider-level performance summary.

    Returns
    -------
    pandas.DataFrame
        High-performing provider registry.

    Raises
    ------
    ValidationError
        Raised when configured rule columns are missing or rule operators are
        unsupported.

    Notes
    -----
    Output grain is one row per provider per matched performance rule.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("high_performing_providers", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: High performing providers disabled")
        return pd.DataFrame()

    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build high performing providers")

    for rule_key, rule_config in config.get("performance_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            logger.info("SKIP provider performance rule disabled: %s", rule_key)
            continue

        label = rule_config.get("label", rule_key)
        priority_rank = rule_config.get("priority_rank")
        description = rule_config.get("description", "")

        mask = pd.Series(True, index=provider_performance_summary.index)

        for condition in rule_config.get("conditions", []):
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            require_columns(
                runtime=runtime,
                dataframe=provider_performance_summary,
                dataset_name="provider_performance_summary",
                required_columns=[column],
                source_layer=runtime.layer_name,
                source_dataset="provider_performance_summary",
            )

            mask = mask & apply_operator(
                provider_performance_summary[column],
                operator,
                value,
            )

        selected = provider_performance_summary.loc[mask].copy()

        if selected.empty:
            continue

        selected["performance_rule_key"] = rule_key
        selected["performance_label"] = label
        selected["performance_priority_rank"] = priority_rank
        selected["performance_description"] = description

        add_rule_record(
            runtime=runtime,
            rule_group="high_performing_providers",
            rule_name=rule_key,
            rule_type="provider_selection",
            description=description,
            source_dataset="provider_performance_summary",
            source_layer=runtime.layer_name,
            rule_config=rule_config,
        )

        output_frames.append(selected)

        logger.info(
            "Applied high-performing provider rule: %s | Providers: %s",
            rule_key,
            len(selected),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                "performance_rule_key",
                "performance_label",
                "performance_priority_rank",
                "performance_description",
            ]
        )

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "high_performing_providers"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = "provider_performance_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="high_performing_providers",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="High performing providers built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="provider_performance_summary",
    )

    logger.info("COMPLETE: Build high performing providers | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_provider_analytics_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Provider Analytics layer.

    Parameters
    ----------
    config_path:
        Provider Analytics configuration path.

    Returns
    -------
    AnalyticsBuildResult
        Standard build result containing status, message, row count, and column
        count.

    Raises
    ------
    None
        Exceptions are captured and returned as a failed AnalyticsBuildResult.

    Notes
    -----
    This layer consumes configured Provider Performance / Gold Layer datasets
    and produces Provider Analytics summaries and registries.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Provider Analytics started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        provider_key = runtime.config.get("join_keys", {}).get(
            "provider_key",
            "provider_id",
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "provider_performance": "Gold Layer",
                "provider_attribution": "Gold Layer",
                "provider_network": "Gold Layer",
                "provider_specialty": "Gold Layer",
            },
            key_column_map={
                "provider_performance": provider_key,
                "provider_attribution": provider_key,
                "provider_network": provider_key,
                "provider_specialty": provider_key,
            },
        )

        provider_performance_summary = build_provider_performance_summary(
            runtime=runtime,
            datasets=datasets,
        )

        provider_network_summary = build_provider_network_summary(
            runtime=runtime,
            datasets=datasets,
        )

        provider_specialty_summary = build_provider_specialty_summary(
            runtime=runtime,
            datasets=datasets,
        )

        provider_cost_summary = build_provider_cost_summary(
            runtime=runtime,
            provider_performance_summary=provider_performance_summary,
        )

        provider_utilization_summary = build_provider_utilization_summary(
            runtime=runtime,
            provider_performance_summary=provider_performance_summary,
        )

        high_performing_providers = build_high_performing_providers(
            runtime=runtime,
            provider_performance_summary=provider_performance_summary,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "provider_performance_summary": provider_performance_summary,
            "provider_network_summary": provider_network_summary,
            "provider_specialty_summary": provider_specialty_summary,
            "provider_cost_summary": provider_cost_summary,
            "provider_utilization_summary": provider_utilization_summary,
            "high_performing_providers": high_performing_providers,
        }

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_provider_analytics_layer",
            message="Provider Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2F - Provider Analytics",
            source_dataset="provider_analytics_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="provider_analytics_dataset_inventory",
            column_dictionary_name="provider_analytics_column_dictionary",
            rule_catalog_name="provider_analytics_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="provider_analytics_audit_records",
            validation_results_name="provider_analytics_validation_results",
            execution_summary_name="provider_analytics_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Provider Analytics completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name=DOMAIN_SECTION,
            status=STATUS_SUCCESS,
            message="Provider Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(
                len(dataframe.columns) for dataframe in output_assets.values()
            ),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Provider Analytics failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_provider_analytics_layer",
                message=str(exc),
                source_layer="Layer 2F - Provider Analytics",
                source_dataset="provider_analytics",
            )

            try:
                output_format = get_output_format(
                    runtime=runtime,
                    domain_section=DOMAIN_SECTION,
                )

                write_audit_outputs(
                    runtime=runtime,
                    output_assets={},
                    output_format=output_format,
                    audit_records_name="provider_analytics_audit_records",
                    validation_results_name="provider_analytics_validation_results",
                    execution_summary_name="provider_analytics_execution_summary",
                )

            except Exception as audit_exc:
                logger.error(
                    "Failed to write Provider Analytics audit outputs: %s",
                    audit_exc,
                )

        return AnalyticsBuildResult(
            name=DOMAIN_SECTION,
            status=STATUS_FAILED,
            message=str(exc),
        )

    finally:
        if runtime is not None:
            runtime.context.logging.close()


###############################################################################
# CLI Entry Point
###############################################################################

def main() -> None:
    """
    Purpose
    -------
    Command-line entry point for Provider Analytics.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Raised with exit code 1 when execution fails.
    """

    config_path = os.environ.get(
        "MEDFABRIC_PROVIDER_ANALYTICS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_provider_analytics_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Provider Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()