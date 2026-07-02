###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/quality/build_quality_reporting.py
#
# Layer:
#     Layer 3 - Insights
#
# Domain:
#     Quality Reporting
#
# Purpose:
#     Builds quality reporting summaries from Layer 2 Quality Analytics outputs.
#
# Business Context:
#     Quality reporting helps executives and analysts understand:
#
#         - members with quality gaps
#         - total quality gap records
#         - quality reporting availability
#         - quality gap registry readiness
#
# Architectural Rule:
#     Quality Reporting consumes Layer 2 Analytics Platform outputs only.
#
#     This file does NOT:
#         - read raw, bronze, silver, gold, feature store, or modeling data
#         - calculate quality measure logic
#         - create quality gap registries
#         - train or score predictive models
#
# Inputs:
#     config/insights/insights.yaml
#     data/analytics_platform/quality_analytics/*.parquet
#
# Outputs:
#     data/insights/quality/quality_reporting_summary.parquet
#
# Run:
#     python -m src.insights.quality.build_quality_reporting
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
DEFAULT_DOMAIN_NAME = "Quality Reporting"
DOMAIN_SECTION = "quality_reporting"

PLATFORM_SECTION = "insights"

LOGGER_NAME = "medfabric.insights.quality"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Quality Reporting configuration sections.

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
    This function validates only the Quality Reporting domain contract.
    Shared YAML loading is handled by PipelineContext.
    """

    required_sections = [
        "insights",
        "paths",
        "join_keys",
        "quality_reporting",
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

    quality_config = config.get("quality_reporting", {})

    for subsection in ["source_datasets", "metrics"]:
        if subsection not in quality_config:
            missing_sections.append(f"quality_reporting.{subsection}")

    if missing_sections:
        raise PipelineError(
            "Quality Reporting configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> InsightsDomainRuntime:
    """
    Purpose
    -------
    Initialize Quality Reporting runtime.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration.

    Returns
    -------
    InsightsDomainRuntime
        Initialized runtime for Quality Reporting.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This follows the same MedFabric shared runtime pattern as the other Layer 3
    reporting domains.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 3 - Quality Reporting"
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
        message="Quality Reporting runtime initialized successfully.",
        source_layer="Layer 3 - Insights",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Metric Calculation Helpers
###############################################################################

def calculate_metric_value(
    dataframe: pd.DataFrame,
    metric_name: str,
    metric_config: Dict[str, Any],
) -> Any:
    """
    Purpose
    -------
    Calculate a single configured quality reporting metric.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    metric_name:
        Metric name from insights.yaml.

    metric_config:
        Metric configuration.

    Returns
    -------
    Any
        Calculated metric value.

    Raises
    ------
    ValidationError
        Raised when calculation type or required columns are invalid.

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
            f"Quality metric '{metric_name}' requires a configured column "
            f"for calculation_type '{calculation_type}'."
        )

    if column not in dataframe.columns:
        raise ValidationError(
            f"Quality metric '{metric_name}' source column missing: {column}"
        )

    numeric_series = pd.to_numeric(dataframe[column], errors="coerce")

    if calculation_type == "count_distinct":
        return int(dataframe[column].nunique(dropna=True))

    if calculation_type == "sum":
        return float(numeric_series.fillna(0).sum())

    if calculation_type == "mean":
        return float(numeric_series.mean())

    if calculation_type == "min":
        return float(numeric_series.min())

    if calculation_type == "max":
        return float(numeric_series.max())

    raise ValidationError(
        f"Unsupported quality metric calculation_type: {calculation_type}"
    )


def build_quality_metric_row(
    runtime: InsightsDomainRuntime,
    metric_name: str,
    metric_config: Dict[str, Any],
    metric_value: Any,
    source_row_count: int,
) -> Dict[str, Any]:
    """
    Purpose
    -------
    Build one standardized quality reporting metric row.

    Parameters
    ----------
    runtime:
        Quality Reporting runtime.

    metric_name:
        Metric key.

    metric_config:
        Metric configuration.

    metric_value:
        Calculated metric value.

    source_row_count:
        Number of rows in the source dataset.

    Returns
    -------
    dict
        Quality reporting metric row.

    Raises
    ------
    None

    Notes
    -----
    The output schema aligns with the other Insights reporting summaries.
    """

    return {
        "reporting_domain": "quality_reporting",
        "metric_key": metric_name,
        "metric_label": metric_config.get("label", metric_name),
        "metric_category": metric_config.get("category", "Quality"),
        "calculation_type": metric_config.get("calculation_type"),
        "source_dataset": metric_config.get("source_dataset"),
        "source_column": metric_config.get("column"),
        "metric_value": metric_value,
        "source_row_count": int(source_row_count),
        "metric_status": "AVAILABLE",
        "insights_layer_run_id": runtime.context.run_id,
        "insights_domain": runtime.domain_name,
        "insights_asset_name": "quality_reporting_summary",
        "built_at_utc": utc_now().isoformat(),
    }


###############################################################################
# Quality Reporting Summary
###############################################################################

def build_quality_reporting_summary(
    runtime: InsightsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build quality reporting summary.

    Parameters
    ----------
    runtime:
        Quality Reporting runtime.

    datasets:
        Loaded Layer 2 Quality Analytics datasets.

    Returns
    -------
    pandas.DataFrame
        Quality reporting summary.

    Raises
    ------
    ValidationError
        Raised when configured metrics reference invalid source columns.

    Notes
    -----
    Missing optional datasets are represented as metric rows with
    MISSING_SOURCE_DATASET status so the reporting layer still runs.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("quality_reporting", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Quality Reporting disabled")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    logger.info("START: Build quality reporting summary")

    for metric_name, metric_config in config.get("metrics", {}).items():
        source_dataset = metric_config.get("source_dataset")
        calculation_type = metric_config.get("calculation_type")
        metric_column = metric_config.get("column")

        add_metric_rule_record(
            runtime=runtime,
            reporting_domain="quality_reporting",
            metric_name=metric_name,
            metric_config=metric_config,
            description=f"Quality reporting metric: {metric_name}",
        )

        if source_dataset not in datasets:
            warn_dataset_missing(
                runtime=runtime,
                datasets=datasets,
                dataset_name=source_dataset,
                source_layer="Layer 2 - Quality Analytics",
            )

            rows.append(
                {
                    "reporting_domain": "quality_reporting",
                    "metric_key": metric_name,
                    "metric_label": metric_config.get("label", metric_name),
                    "metric_category": metric_config.get("category", "Quality"),
                    "calculation_type": calculation_type,
                    "source_dataset": source_dataset,
                    "source_column": metric_column,
                    "metric_value": None,
                    "source_row_count": 0,
                    "metric_status": "MISSING_SOURCE_DATASET",
                    "insights_layer_run_id": runtime.context.run_id,
                    "insights_domain": runtime.domain_name,
                    "insights_asset_name": "quality_reporting_summary",
                    "built_at_utc": utc_now().isoformat(),
                }
            )

            continue

        source_df = datasets[source_dataset]

        if calculation_type != "count_rows":
            require_columns(
                runtime=runtime,
                dataframe=source_df,
                dataset_name=source_dataset,
                required_columns=[metric_column],
                source_layer="Layer 2 - Quality Analytics",
                source_dataset=source_dataset,
            )

        metric_value = calculate_metric_value(
            dataframe=source_df,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        rows.append(
            build_quality_metric_row(
                runtime=runtime,
                metric_name=metric_name,
                metric_config=metric_config,
                metric_value=metric_value,
                source_row_count=len(source_df),
            )
        )

        logger.info(
            "Built quality metric: %s | Value: %s | Source: %s",
            metric_name,
            metric_value,
            source_dataset,
        )

    output_df = pd.DataFrame(rows)

    add_rule_record(
        runtime=runtime,
        rule_group="quality_reporting",
        rule_name="quality_reporting_summary_assembly",
        rule_type="reporting_summary",
        description="Builds quality reporting summary from Quality Analytics outputs.",
        source_dataset="multiple_quality_analytics_outputs",
        source_layer="Layer 2 - Quality Analytics",
        rule_config=config,
    )

    register_output_asset(
        runtime=runtime,
        dataset_name="quality_reporting_summary",
        dataframe=output_df,
        dataset_type="quality_reporting_output",
        message="Quality reporting summary built successfully.",
        source_layer="Layer 2 - Quality Analytics",
        source_dataset="multiple_quality_analytics_outputs",
    )

    logger.info("COMPLETE: Build quality reporting summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_quality_reporting(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> InsightsBuildResult:
    """
    Purpose
    -------
    Build the Quality Reporting domain.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration.

    Returns
    -------
    InsightsBuildResult
        Standard build result for Quality Reporting.

    Raises
    ------
    None
        Exceptions are captured and returned as a failed InsightsBuildResult.

    Notes
    -----
    This function performs the complete Quality Reporting workflow:
        - initialize runtime
        - load configured Layer 2 Quality Analytics inputs
        - calculate quality reporting metrics
        - write output
        - write metadata
        - write audit outputs
    """

    runtime: Optional[InsightsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Quality Reporting started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=PLATFORM_SECTION,
        )

        member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "quality_gap_registry": "Layer 2 - Quality Analytics",
                "quality_gap_summary": "Layer 2 - Quality Analytics",
            },
            key_column_map={
                "quality_gap_registry": member_key,
            },
        )

        quality_reporting_summary = build_quality_reporting_summary(
            runtime=runtime,
            datasets=datasets,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "quality_reporting_summary": quality_reporting_summary,
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
            step_name="build_quality_reporting",
            message="Quality Reporting completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 3 - Insights",
            source_dataset="quality_reporting_outputs",
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
        logger.info("MedFabric Quality Reporting completed successfully")
        logger.info("=" * 80)

        return InsightsBuildResult(
            name="quality_reporting",
            status=STATUS_SUCCESS,
            message="Quality Reporting completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(
                len(dataframe.columns) for dataframe in output_assets.values()
            ),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Quality Reporting failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_quality_reporting",
                message=str(exc),
                source_layer="Layer 3 - Insights",
                source_dataset="quality_reporting",
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
                    "Failed to write Quality Reporting audit outputs: %s",
                    audit_exc,
                )

        return InsightsBuildResult(
            name="quality_reporting",
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
    Command-line entry point for Quality Reporting.

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

    result = build_quality_reporting(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Quality Reporting failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()