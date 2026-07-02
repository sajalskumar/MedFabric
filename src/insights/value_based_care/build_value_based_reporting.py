###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/value_based_care/build_value_based_reporting.py
#
# Layer:
#     Layer 3 - Insights
#
# Domain:
#     Value-Based Care Reporting
#
# Purpose:
#     Builds value-based care reporting summaries from Layer 2 Value-Based Care
#     analytics outputs.
#
# Business Context:
#     Value-Based Care Reporting helps executives and analysts understand:
#
#         - value-based contract count
#         - provider incentive count
#         - provider shared savings
#         - risk adjustment band count
#         - bundle opportunity count
#         - value-based care reporting readiness
#
# Architectural Rule:
#     Value-Based Care Reporting consumes Layer 2 Analytics Platform outputs only.
#
#     This file does NOT:
#         - read raw, bronze, silver, gold, feature store, or modeling data
#         - calculate value-based care contract logic
#         - assign provider incentives
#         - calculate risk adjustment bands
#         - build bundle opportunity rules
#
# Inputs:
#     config/insights/insights.yaml
#     data/analytics_platform/value_based_care/*.parquet
#
# Outputs:
#     data/insights/value_based_care/value_based_care_reporting_summary.parquet
#
# Run:
#     python -m src.insights.value_based_care.build_value_based_reporting
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
DEFAULT_DOMAIN_NAME = "Value-Based Care Reporting"
DOMAIN_SECTION = "value_based_care_reporting"

PLATFORM_SECTION = "insights"

LOGGER_NAME = "medfabric.insights.value_based_care"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Value-Based Care Reporting configuration sections.

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
        Raised when required configuration sections are missing.

    Notes
    -----
    This function validates only the Value-Based Care Reporting domain contract.
    Shared YAML loading is handled by PipelineContext.
    """

    required_sections = [
        "insights",
        "paths",
        "join_keys",
        "value_based_care_reporting",
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

    value_based_config = config.get("value_based_care_reporting", {})

    for subsection in ["source_datasets", "metrics"]:
        if subsection not in value_based_config:
            missing_sections.append(f"value_based_care_reporting.{subsection}")

    if missing_sections:
        raise PipelineError(
            "Value-Based Care Reporting configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> InsightsDomainRuntime:
    """
    Purpose
    -------
    Initialize Value-Based Care Reporting runtime.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration.

    Returns
    -------
    InsightsDomainRuntime
        Initialized runtime for Value-Based Care Reporting.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This follows the shared MedFabric Insights runtime pattern used by all
    Layer 3 reporting domains.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 3 - Value-Based Care Reporting"
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
        message="Value-Based Care Reporting runtime initialized successfully.",
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
    Calculate a single configured value-based care reporting metric.

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
            f"Value-Based Care metric '{metric_name}' requires a configured "
            f"column for calculation_type '{calculation_type}'."
        )

    if column not in dataframe.columns:
        raise ValidationError(
            f"Value-Based Care metric '{metric_name}' source column missing: {column}"
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
        f"Unsupported Value-Based Care metric calculation_type: {calculation_type}"
    )


def build_value_based_metric_row(
    runtime: InsightsDomainRuntime,
    metric_name: str,
    metric_config: Dict[str, Any],
    metric_value: Any,
    source_row_count: int,
) -> Dict[str, Any]:
    """
    Purpose
    -------
    Build one standardized value-based care reporting metric row.

    Parameters
    ----------
    runtime:
        Value-Based Care Reporting runtime.

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
        Value-Based Care reporting metric row.

    Raises
    ------
    None

    Notes
    -----
    The output schema aligns with all other Insights reporting summaries so
    dashboards can combine reporting domains consistently.
    """

    return {
        "reporting_domain": "value_based_care_reporting",
        "metric_key": metric_name,
        "metric_label": metric_config.get("label", metric_name),
        "metric_category": metric_config.get("category", "Value-Based Care"),
        "calculation_type": metric_config.get("calculation_type"),
        "source_dataset": metric_config.get("source_dataset"),
        "source_column": metric_config.get("column"),
        "metric_value": metric_value,
        "source_row_count": int(source_row_count),
        "metric_status": "AVAILABLE",
        "insights_layer_run_id": runtime.context.run_id,
        "insights_domain": runtime.domain_name,
        "insights_asset_name": "value_based_care_reporting_summary",
        "built_at_utc": utc_now().isoformat(),
    }


###############################################################################
# Value-Based Care Reporting Summary
###############################################################################

def build_value_based_care_reporting_summary(
    runtime: InsightsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build value-based care reporting summary.

    Parameters
    ----------
    runtime:
        Value-Based Care Reporting runtime.

    datasets:
        Loaded Layer 2 Value-Based Care datasets.

    Returns
    -------
    pandas.DataFrame
        Value-Based Care reporting summary.

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
    config = runtime.config.get("value_based_care_reporting", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Value-Based Care Reporting disabled")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    logger.info("START: Build value-based care reporting summary")

    for metric_name, metric_config in config.get("metrics", {}).items():
        source_dataset = metric_config.get("source_dataset")
        calculation_type = metric_config.get("calculation_type")
        metric_column = metric_config.get("column")

        add_metric_rule_record(
            runtime=runtime,
            reporting_domain="value_based_care_reporting",
            metric_name=metric_name,
            metric_config=metric_config,
            description=f"Value-Based Care reporting metric: {metric_name}",
        )

        if source_dataset not in datasets:
            warn_dataset_missing(
                runtime=runtime,
                datasets=datasets,
                dataset_name=source_dataset,
                source_layer="Layer 2 - Value-Based Care",
            )

            rows.append(
                {
                    "reporting_domain": "value_based_care_reporting",
                    "metric_key": metric_name,
                    "metric_label": metric_config.get("label", metric_name),
                    "metric_category": metric_config.get(
                        "category",
                        "Value-Based Care",
                    ),
                    "calculation_type": calculation_type,
                    "source_dataset": source_dataset,
                    "source_column": metric_column,
                    "metric_value": None,
                    "source_row_count": 0,
                    "metric_status": "MISSING_SOURCE_DATASET",
                    "insights_layer_run_id": runtime.context.run_id,
                    "insights_domain": runtime.domain_name,
                    "insights_asset_name": "value_based_care_reporting_summary",
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
                source_layer="Layer 2 - Value-Based Care",
                source_dataset=source_dataset,
            )

        metric_value = calculate_metric_value(
            dataframe=source_df,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        rows.append(
            build_value_based_metric_row(
                runtime=runtime,
                metric_name=metric_name,
                metric_config=metric_config,
                metric_value=metric_value,
                source_row_count=len(source_df),
            )
        )

        logger.info(
            "Built value-based care metric: %s | Value: %s | Source: %s",
            metric_name,
            metric_value,
            source_dataset,
        )

    output_df = pd.DataFrame(rows)

    add_rule_record(
        runtime=runtime,
        rule_group="value_based_care_reporting",
        rule_name="value_based_care_reporting_summary_assembly",
        rule_type="reporting_summary",
        description=(
            "Builds value-based care reporting summary from Value-Based Care "
            "Analytics Platform outputs."
        ),
        source_dataset="multiple_value_based_care_outputs",
        source_layer="Layer 2 - Value-Based Care",
        rule_config=config,
    )

    register_output_asset(
        runtime=runtime,
        dataset_name="value_based_care_reporting_summary",
        dataframe=output_df,
        dataset_type="value_based_care_reporting_output",
        message="Value-Based Care reporting summary built successfully.",
        source_layer="Layer 2 - Value-Based Care",
        source_dataset="multiple_value_based_care_outputs",
    )

    logger.info(
        "COMPLETE: Build value-based care reporting summary | Rows: %s",
        len(output_df),
    )

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_value_based_reporting(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> InsightsBuildResult:
    """
    Purpose
    -------
    Build the Value-Based Care Reporting domain.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration.

    Returns
    -------
    InsightsBuildResult
        Standard build result for Value-Based Care Reporting.

    Raises
    ------
    None
        Exceptions are captured and returned as a failed InsightsBuildResult.

    Notes
    -----
    This function performs the complete Value-Based Care Reporting workflow:
        - initialize runtime
        - load configured Layer 2 Value-Based Care inputs
        - calculate value-based care reporting metrics
        - write output
        - write metadata
        - write audit outputs
    """

    runtime: Optional[InsightsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Value-Based Care Reporting started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=PLATFORM_SECTION,
        )

        provider_key = runtime.config.get("join_keys", {}).get(
            "provider_key",
            "provider_id",
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "value_based_contract_summary": "Layer 2 - Value-Based Care",
                "provider_incentive_summary": "Layer 2 - Value-Based Care",
                "shared_savings_summary": "Layer 2 - Value-Based Care",
                "risk_adjustment_summary": "Layer 2 - Value-Based Care",
                "bundle_opportunity_summary": "Layer 2 - Value-Based Care",
                "vbc_executive_scorecard": "Layer 2 - Value-Based Care",
            },
            key_column_map={
                "provider_incentive_summary": provider_key,
            },
        )

        value_based_care_reporting_summary = build_value_based_care_reporting_summary(
            runtime=runtime,
            datasets=datasets,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "value_based_care_reporting_summary": value_based_care_reporting_summary,
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
            step_name="build_value_based_reporting",
            message="Value-Based Care Reporting completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 3 - Insights",
            source_dataset="value_based_care_reporting_outputs",
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
        logger.info("MedFabric Value-Based Care Reporting completed successfully")
        logger.info("=" * 80)

        return InsightsBuildResult(
            name="value_based_care_reporting",
            status=STATUS_SUCCESS,
            message="Value-Based Care Reporting completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(
                len(dataframe.columns) for dataframe in output_assets.values()
            ),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Value-Based Care Reporting failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_value_based_reporting",
                message=str(exc),
                source_layer="Layer 3 - Insights",
                source_dataset="value_based_care_reporting",
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
                    "Failed to write Value-Based Care Reporting audit outputs: %s",
                    audit_exc,
                )

        return InsightsBuildResult(
            name="value_based_care_reporting",
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
    Command-line entry point for Value-Based Care Reporting.

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

    result = build_value_based_reporting(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Value-Based Care Reporting failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()