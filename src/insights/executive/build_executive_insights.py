###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/executive/build_executive_insights.py
#
# Layer:
#     Layer 3 - Insights
#
# Domain:
#     Executive Insights
#
# Purpose:
#     Builds executive-ready KPI scorecards, domain status summaries, and
#     consolidated executive reporting summaries from Layer 2 Analytics Platform
#     outputs.
#
# Business Context:
#     Executive users need a concise view of the MedFabric platform covering:
#
#         - population health
#         - clinical analytics
#         - quality analytics
#         - predictive analytics
#         - provider analytics
#         - care management
#         - value-based care
#
#     This module converts Layer 2 outputs into dashboard-ready summary datasets.
#
# Architectural Rule:
#     Executive Insights consumes Layer 2 Analytics Platform outputs only.
#
#     This file does NOT:
#         - read raw, bronze, silver, gold, feature store, or modeling data
#         - train models
#         - score members
#         - generate healthcare business logic already owned by Layer 2
#
# Inputs:
#     config/insights/insights.yaml
#     data/analytics_platform/*/*.parquet
#
# Outputs:
#     data/insights/executive/executive_kpi_scorecard.parquet
#     data/insights/executive/executive_summary.parquet
#     data/insights/executive/executive_domain_status.parquet
#     data/insights/metadata/insights_dataset_inventory.parquet
#     data/insights/metadata/insights_column_dictionary.parquet
#     data/insights/metadata/insights_rule_catalog.parquet
#     data/insights/audit/insights_audit_records.parquet
#     data/insights/audit/insights_validation_results.parquet
#     data/insights/audit/insights_execution_summary.parquet
#
# Run:
#     python -m src.insights.executive.build_executive_insights
#
###############################################################################

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.exception_manager import PipelineError, ValidationError
from src.common.pipeline_context import create_pipeline_context
from src.insights.common.audit import (
    add_failed_audit,
    add_success_audit,
    write_audit_outputs,
)
from src.insights.common.io import (
    get_output_format,
    load_input_datasets,
    write_output_assets,
)
from src.insights.common.metadata import (
    add_metric_rule_record,
    add_rule_record,
    register_output_asset,
    write_metadata_outputs,
)
from src.insights.common.runtime import (
    InsightsBuildResult,
    InsightsDomainRuntime,
    STATUS_FAILED,
    STATUS_SUCCESS,
    create_domain_runtime,
    normalize_config_file,
    utc_now,
)
from src.insights.common.validation import (
    require_columns,
    validate_output_assets,
    warn_dataset_missing,
)


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "insights/insights.yaml"

DEFAULT_LAYER_NAME = "Layer 3 - Insights"
DEFAULT_DOMAIN_NAME = "Executive Insights"
DOMAIN_SECTION = "executive_insights"

PLATFORM_SECTION = "insights"

LOGGER_NAME = "medfabric.insights.executive"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Executive Insights configuration sections.

    Parameters
    ----------
    config:
        Loaded Insights YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required sections are missing.

    Notes
    -----
    This validates only the configuration contract needed by Executive Insights.
    Generic YAML loading is handled by PipelineContext.
    """

    required_sections = [
        "insights",
        "paths",
        "join_keys",
        "executive_insights",
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

    executive_config = config.get("executive_insights", {})

    for subsection in ["kpi_scorecard", "executive_summary", "domain_status"]:
        if subsection not in executive_config:
            missing_sections.append(f"executive_insights.{subsection}")

    if missing_sections:
        raise PipelineError(
            "Executive Insights configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> InsightsDomainRuntime:
    """
    Purpose
    -------
    Initialize Executive Insights runtime.

    Parameters
    ----------
    config_path:
        Path to config/insights/insights.yaml.

    Returns
    -------
    InsightsDomainRuntime
        Initialized runtime for Executive Insights.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This follows the same MedFabric shared runtime pattern used by the Analytics
    Platform and other Insights modules.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 3 - Executive Insights"
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
        message="Executive Insights runtime initialized successfully.",
        source_layer="Layer 3 - Insights",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# KPI Calculation Helpers
###############################################################################

def calculate_metric_value(
    dataframe: pd.DataFrame,
    metric_name: str,
    metric_config: Dict[str, Any],
) -> Any:
    """
    Purpose
    -------
    Calculate a single configured executive metric value.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    metric_name:
        Metric name.

    metric_config:
        Metric configuration from insights.yaml.

    Returns
    -------
    Any
        Calculated metric value.

    Raises
    ------
    ValidationError
        Raised when the calculation type is unsupported or required columns are
        missing.

    Notes
    -----
    Supported calculation types:
        - count_rows
        - count_distinct
        - sum
        - mean
        - min
        - max
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

    if calculation_type == "count_rows":
        return int(len(dataframe))

    if not column:
        raise ValidationError(
            f"Metric '{metric_name}' requires a configured column for "
            f"calculation_type '{calculation_type}'."
        )

    if column not in dataframe.columns:
        raise ValidationError(
            f"Metric '{metric_name}' source column missing: {column}"
        )

    if calculation_type == "count_distinct":
        return int(dataframe[column].nunique(dropna=True))

    if calculation_type == "sum":
        return float(pd.to_numeric(dataframe[column], errors="coerce").fillna(0).sum())

    if calculation_type == "mean":
        return float(pd.to_numeric(dataframe[column], errors="coerce").mean())

    if calculation_type == "min":
        return float(pd.to_numeric(dataframe[column], errors="coerce").min())

    if calculation_type == "max":
        return float(pd.to_numeric(dataframe[column], errors="coerce").max())

    raise ValidationError(
        f"Unsupported executive metric calculation_type: {calculation_type}"
    )


def build_metric_row(
    runtime: InsightsDomainRuntime,
    metric_name: str,
    metric_config: Dict[str, Any],
    metric_value: Any,
    source_row_count: int,
) -> Dict[str, Any]:
    """
    Purpose
    -------
    Build one standardized Executive KPI scorecard row.

    Parameters
    ----------
    runtime:
        Executive Insights runtime.

    metric_name:
        Metric key from configuration.

    metric_config:
        Metric configuration.

    metric_value:
        Calculated metric value.

    source_row_count:
        Number of source rows used for the calculation.

    Returns
    -------
    dict
        Executive KPI row.

    Raises
    ------
    None

    Notes
    -----
    The output row is intentionally business-friendly so it can be consumed by
    dashboards, notebooks, or downstream reporting tools.
    """

    return {
        "kpi_key": metric_name,
        "kpi_label": metric_config.get("label", metric_name),
        "kpi_category": metric_config.get("category", "Uncategorized"),
        "calculation_type": metric_config.get("calculation_type"),
        "source_dataset": metric_config.get("source_dataset"),
        "source_column": metric_config.get("column"),
        "metric_value": metric_value,
        "source_row_count": int(source_row_count),
        "analytics_layer_run_id": runtime.context.run_id,
        "insights_layer_run_id": runtime.context.run_id,
        "insights_domain": runtime.domain_name,
        "insights_asset_name": "executive_kpi_scorecard",
        "built_at_utc": utc_now().isoformat(),
    }


###############################################################################
# Executive KPI Scorecard
###############################################################################

def build_executive_kpi_scorecard(
    runtime: InsightsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the executive KPI scorecard.

    Parameters
    ----------
    runtime:
        Executive Insights runtime.

    datasets:
        Loaded Layer 2 Analytics Platform datasets.

    Returns
    -------
    pandas.DataFrame
        Executive KPI scorecard.

    Raises
    ------
    ValidationError
        Raised when a configured KPI references an unavailable required column.

    Notes
    -----
    Optional missing source datasets are recorded as warning KPI rows instead of
    failing the whole executive build. This keeps the executive scorecard useful
    when one optional reporting domain is unavailable.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("executive_insights", {}).get("kpi_scorecard", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Executive KPI scorecard disabled")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    logger.info("START: Build executive KPI scorecard")

    for metric_name, metric_config in config.get("kpis", {}).items():
        source_dataset = metric_config.get("source_dataset")

        add_metric_rule_record(
            runtime=runtime,
            reporting_domain="executive_kpi_scorecard",
            metric_name=metric_name,
            metric_config=metric_config,
            description=f"Executive KPI: {metric_config.get('label', metric_name)}",
        )

        if source_dataset not in datasets:
            warn_dataset_missing(
                runtime=runtime,
                datasets=datasets,
                dataset_name=source_dataset,
                source_layer="Layer 2 - Analytics Platform",
            )

            rows.append(
                {
                    "kpi_key": metric_name,
                    "kpi_label": metric_config.get("label", metric_name),
                    "kpi_category": metric_config.get("category", "Uncategorized"),
                    "calculation_type": metric_config.get("calculation_type"),
                    "source_dataset": source_dataset,
                    "source_column": metric_config.get("column"),
                    "metric_value": None,
                    "source_row_count": 0,
                    "kpi_status": "MISSING_SOURCE_DATASET",
                    "analytics_layer_run_id": None,
                    "insights_layer_run_id": runtime.context.run_id,
                    "insights_domain": runtime.domain_name,
                    "insights_asset_name": "executive_kpi_scorecard",
                    "built_at_utc": utc_now().isoformat(),
                }
            )

            continue

        source_df = datasets[source_dataset]
        calculation_type = metric_config.get("calculation_type")
        metric_column = metric_config.get("column")

        if calculation_type != "count_rows":
            require_columns(
                runtime=runtime,
                dataframe=source_df,
                dataset_name=source_dataset,
                required_columns=[metric_column],
                source_layer="Layer 2 - Analytics Platform",
                source_dataset=source_dataset,
            )

        metric_value = calculate_metric_value(
            dataframe=source_df,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        row = build_metric_row(
            runtime=runtime,
            metric_name=metric_name,
            metric_config=metric_config,
            metric_value=metric_value,
            source_row_count=len(source_df),
        )

        row["kpi_status"] = "AVAILABLE"
        rows.append(row)

        logger.info(
            "Built executive KPI: %s | Value: %s | Source: %s",
            metric_name,
            metric_value,
            source_dataset,
        )

    output_df = pd.DataFrame(rows)

    register_output_asset(
        runtime=runtime,
        dataset_name="executive_kpi_scorecard",
        dataframe=output_df,
        dataset_type="executive_insights_output",
        message="Executive KPI scorecard built successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset="multiple_layer_2_outputs",
    )

    logger.info("COMPLETE: Build executive KPI scorecard | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Executive Domain Status
###############################################################################

def build_executive_domain_status(
    runtime: InsightsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build executive domain availability status.

    Parameters
    ----------
    runtime:
        Executive Insights runtime.

    datasets:
        Loaded Layer 2 datasets.

    Returns
    -------
    pandas.DataFrame
        Domain-level availability status.

    Raises
    ------
    None

    Notes
    -----
    This output helps executives and developers quickly see whether each Layer 2
    Analytics Platform domain produced expected assets for reporting.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("executive_insights", {}).get("domain_status", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Executive domain status disabled")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    logger.info("START: Build executive domain status")

    for domain_key, domain_config in config.get("domains", {}).items():
        required_datasets = domain_config.get("required_datasets", [])
        available_datasets = [
            dataset_name for dataset_name in required_datasets if dataset_name in datasets
        ]
        missing_datasets = [
            dataset_name for dataset_name in required_datasets if dataset_name not in datasets
        ]

        total_rows = sum(
            len(datasets[dataset_name])
            for dataset_name in available_datasets
        )

        if missing_datasets:
            domain_status = "INCOMPLETE"
        else:
            domain_status = "AVAILABLE"

        rows.append(
            {
                "domain_key": domain_key,
                "domain_display_name": domain_config.get("display_name", domain_key),
                "domain_status": domain_status,
                "required_dataset_count": len(required_datasets),
                "available_dataset_count": len(available_datasets),
                "missing_dataset_count": len(missing_datasets),
                "available_datasets": ", ".join(available_datasets),
                "missing_datasets": ", ".join(missing_datasets),
                "total_available_rows": int(total_rows),
                "insights_layer_run_id": runtime.context.run_id,
                "insights_domain": runtime.domain_name,
                "insights_asset_name": "executive_domain_status",
                "built_at_utc": utc_now().isoformat(),
            }
        )

        add_rule_record(
            runtime=runtime,
            rule_group="executive_domain_status",
            rule_name=domain_key,
            rule_type="domain_availability_check",
            description=(
                "Checks whether required Layer 2 reporting datasets are "
                f"available for {domain_key}."
            ),
            source_dataset="multiple_layer_2_outputs",
            source_layer="Layer 2 - Analytics Platform",
            rule_config=domain_config,
        )

        logger.info(
            "Domain status: %s | Status: %s | Missing: %s",
            domain_key,
            domain_status,
            missing_datasets,
        )

    output_df = pd.DataFrame(rows)

    register_output_asset(
        runtime=runtime,
        dataset_name="executive_domain_status",
        dataframe=output_df,
        dataset_type="executive_insights_output",
        message="Executive domain status built successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset="multiple_layer_2_outputs",
    )

    logger.info("COMPLETE: Build executive domain status | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Executive Summary
###############################################################################

def build_executive_summary(
    runtime: InsightsDomainRuntime,
    executive_kpi_scorecard: pd.DataFrame,
    executive_domain_status: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build consolidated executive summary.

    Parameters
    ----------
    runtime:
        Executive Insights runtime.

    executive_kpi_scorecard:
        KPI scorecard dataframe.

    executive_domain_status:
        Domain status dataframe.

    Returns
    -------
    pandas.DataFrame
        Executive summary dataframe.

    Raises
    ------
    None

    Notes
    -----
    This output provides a one-row summary suitable for top-level dashboard
    landing pages and final project demonstrations.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("executive_insights", {}).get("executive_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Executive summary disabled")
        return pd.DataFrame()

    available_kpis = 0
    missing_kpis = 0

    if not executive_kpi_scorecard.empty and "kpi_status" in executive_kpi_scorecard.columns:
        available_kpis = int((executive_kpi_scorecard["kpi_status"] == "AVAILABLE").sum())
        missing_kpis = int(
            (executive_kpi_scorecard["kpi_status"] == "MISSING_SOURCE_DATASET").sum()
        )

    available_domains = 0
    incomplete_domains = 0

    if not executive_domain_status.empty and "domain_status" in executive_domain_status.columns:
        available_domains = int(
            (executive_domain_status["domain_status"] == "AVAILABLE").sum()
        )
        incomplete_domains = int(
            (executive_domain_status["domain_status"] == "INCOMPLETE").sum()
        )

    rows = [
        {
            "summary_name": "MedFabric Executive Insights Summary",
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "available_kpi_count": available_kpis,
            "missing_kpi_count": missing_kpis,
            "available_domain_count": available_domains,
            "incomplete_domain_count": incomplete_domains,
            "kpi_scorecard_row_count": int(len(executive_kpi_scorecard)),
            "domain_status_row_count": int(len(executive_domain_status)),
            "overall_status": (
                "AVAILABLE"
                if missing_kpis == 0 and incomplete_domains == 0
                else "PARTIAL"
            ),
            "insights_layer_run_id": runtime.context.run_id,
            "insights_domain": runtime.domain_name,
            "insights_asset_name": "executive_summary",
            "built_at_utc": utc_now().isoformat(),
        }
    ]

    output_df = pd.DataFrame(rows)

    add_rule_record(
        runtime=runtime,
        rule_group="executive_summary",
        rule_name="executive_summary_assembly",
        rule_type="summary_assembly",
        description=(
            "Builds a one-row executive summary from KPI scorecard and domain "
            "availability outputs."
        ),
        source_dataset="executive_kpi_scorecard, executive_domain_status",
        source_layer="Layer 3 - Insights",
        rule_config=config,
    )

    register_output_asset(
        runtime=runtime,
        dataset_name="executive_summary",
        dataframe=output_df,
        dataset_type="executive_insights_output",
        message="Executive summary built successfully.",
        source_layer="Layer 3 - Insights",
        source_dataset="executive_kpi_scorecard, executive_domain_status",
    )

    logger.info("COMPLETE: Build executive summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_executive_insights(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> InsightsBuildResult:
    """
    Purpose
    -------
    Build the Executive Insights domain.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration.

    Returns
    -------
    InsightsBuildResult
        Standard build result for Executive Insights.

    Raises
    ------
    None
        Exceptions are captured and returned as a failed InsightsBuildResult.

    Notes
    -----
    This function performs the full Executive Insights workflow:
        - initialize runtime
        - load Layer 2 inputs
        - build KPI scorecard
        - build domain status
        - build executive summary
        - write outputs
        - write metadata
        - write audit outputs
    """

    runtime: Optional[InsightsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Executive Insights started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=PLATFORM_SECTION,
        )

        member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
        provider_key = runtime.config.get("join_keys", {}).get(
            "provider_key",
            "provider_id",
        )
        model_key = runtime.config.get("join_keys", {}).get("model_key", "model_key")

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "population_health_summary": "Layer 2 - Population Health",
                "population_segment_registry": "Layer 2 - Population Health",
                "clinical_condition_summary": "Layer 2 - Clinical Analytics",
                "condition_registry": "Layer 2 - Clinical Analytics",
                "quality_gap_summary": "Layer 2 - Quality Analytics",
                "quality_gap_registry": "Layer 2 - Quality Analytics",
                "unified_prediction_registry": "Layer 2 - Predictive Analytics",
                "member_prediction_summary": "Layer 2 - Predictive Analytics",
                "high_priority_member_registry": "Layer 2 - Predictive Analytics",
                "model_risk_distribution": "Layer 2 - Predictive Analytics",
                "prediction_model_summary": "Layer 2 - Predictive Analytics",
                "provider_performance_summary": "Layer 2 - Provider Analytics",
                "provider_network_summary": "Layer 2 - Provider Analytics",
                "provider_specialty_summary": "Layer 2 - Provider Analytics",
                "provider_cost_summary": "Layer 2 - Provider Analytics",
                "provider_utilization_summary": "Layer 2 - Provider Analytics",
                "high_performing_providers": "Layer 2 - Provider Analytics",
                "care_programs": "Layer 2 - Care Management",
                "case_management_worklist": "Layer 2 - Care Management",
                "transitions_of_care": "Layer 2 - Care Management",
                "disease_management": "Layer 2 - Care Management",
                "outreach_tracking": "Layer 2 - Care Management",
                "program_effectiveness": "Layer 2 - Care Management",
                "value_based_contract_summary": "Layer 2 - Value-Based Care",
                "provider_incentive_summary": "Layer 2 - Value-Based Care",
                "shared_savings_summary": "Layer 2 - Value-Based Care",
                "risk_adjustment_summary": "Layer 2 - Value-Based Care",
                "bundle_opportunity_summary": "Layer 2 - Value-Based Care",
                "vbc_executive_scorecard": "Layer 2 - Value-Based Care",
            },
            key_column_map={
                "population_segment_registry": member_key,
                "condition_registry": member_key,
                "quality_gap_registry": member_key,
                "unified_prediction_registry": member_key,
                "member_prediction_summary": member_key,
                "high_priority_member_registry": member_key,
                "prediction_model_summary": model_key,
                "provider_performance_summary": provider_key,
                "provider_network_summary": provider_key,
                "provider_specialty_summary": provider_key,
                "provider_cost_summary": provider_key,
                "provider_utilization_summary": provider_key,
                "high_performing_providers": provider_key,
                "care_programs": member_key,
                "case_management_worklist": member_key,
                "transitions_of_care": member_key,
                "disease_management": member_key,
                "outreach_tracking": member_key,
            },
        )

        executive_kpi_scorecard = build_executive_kpi_scorecard(
            runtime=runtime,
            datasets=datasets,
        )

        executive_domain_status = build_executive_domain_status(
            runtime=runtime,
            datasets=datasets,
        )

        executive_summary = build_executive_summary(
            runtime=runtime,
            executive_kpi_scorecard=executive_kpi_scorecard,
            executive_domain_status=executive_domain_status,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "executive_kpi_scorecard": executive_kpi_scorecard,
            "executive_summary": executive_summary,
            "executive_domain_status": executive_domain_status,
        }

        validate_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            allow_empty=runtime.config.get("validation", {}).get(
                "fail_on_empty_required_outputs",
                False,
            ) is False,
        )

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_executive_insights",
            message="Executive Insights completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 3 - Insights",
            source_dataset="executive_insights_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="insights_dataset_inventory",
            column_dictionary_name="insights_column_dictionary",
            rule_catalog_name="insights_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="insights_audit_records",
            validation_results_name="insights_validation_results",
            execution_summary_name="insights_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Executive Insights completed successfully")
        logger.info("=" * 80)

        return InsightsBuildResult(
            name="executive_insights",
            status=STATUS_SUCCESS,
            message="Executive Insights completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(
                len(dataframe.columns) for dataframe in output_assets.values()
            ),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Executive Insights failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_executive_insights",
                message=str(exc),
                source_layer="Layer 3 - Insights",
                source_dataset="executive_insights",
            )

            try:
                output_format = get_output_format(
                    runtime=runtime,
                    domain_section=PLATFORM_SECTION,
                )

                write_audit_outputs(
                    runtime=runtime,
                    output_assets={},
                    output_format=output_format,
                    audit_records_name="insights_audit_records",
                    validation_results_name="insights_validation_results",
                    execution_summary_name="insights_execution_summary",
                )

            except Exception as audit_exc:
                logger.error(
                    "Failed to write Executive Insights audit outputs: %s",
                    audit_exc,
                )

        return InsightsBuildResult(
            name="executive_insights",
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
    Command-line entry point for Executive Insights.

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
        "MEDFABRIC_INSIGHTS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_executive_insights(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Executive Insights failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()