###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/insights/build_insights_platform.py
#
# Layer:
#     Layer 3 - Insights
#
# Purpose:
#     Orchestrates the complete MedFabric Insights Platform build.
#
# Business Context:
#     The Insights Platform is the final project layer for MedFabric. It converts
#     Layer 2 Analytics Platform outputs into reporting-ready and executive-ready
#     datasets used for dashboards, scorecards, operational reporting, and final
#     portfolio demonstration.
#
#     The Insights Platform coordinates the following reporting domains:
#
#         - Executive Insights
#         - Financial Reporting
#         - Clinical Reporting
#         - Population Reporting
#         - Provider Reporting
#         - Quality Reporting
#         - Care Management Reporting
#         - Value-Based Care Reporting
#
# Architectural Rule:
#     Layer 3 consumes Layer 2 Analytics Platform outputs only.
#
#     This orchestrator does NOT:
#         - read raw, bronze, silver, gold, feature store, or modeling data
#         - train models
#         - score members
#         - calculate Layer 2 business analytics
#         - duplicate reporting-domain business logic
#
# Inputs:
#     config/insights/insights.yaml
#     data/analytics_platform/*/*.parquet
#
# Outputs:
#     data/insights/executive/
#     data/insights/financial/
#     data/insights/clinical/
#     data/insights/population/
#     data/insights/provider/
#     data/insights/quality/
#     data/insights/care_management/
#     data/insights/value_based_care/
#     data/insights/metadata/
#     data/insights/audit/
#
# Run:
#     python -m src.insights.build_insights_platform
#
###############################################################################

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context
from src.insights.care_management.build_care_management_reporting import (
    build_care_management_reporting,
)
from src.insights.clinical.build_clinical_reporting import build_clinical_reporting
from src.insights.common.audit import (
    add_failed_audit,
    add_success_audit,
    add_warning_audit,
    write_audit_outputs,
)
from src.insights.common.io import get_output_format
from src.insights.common.metadata import (
    add_dataset_record,
    add_rule_record,
    write_metadata_outputs,
)
from src.insights.common.runtime import (
    InsightsBuildResult,
    InsightsDomainRuntime,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    create_domain_runtime,
    normalize_config_file,
    utc_now,
)
from src.insights.executive.build_executive_insights import build_executive_insights
from src.insights.financial.build_financial_reporting import build_financial_reporting
from src.insights.population.build_population_reporting import build_population_reporting
from src.insights.provider.build_provider_reporting import build_provider_reporting
from src.insights.quality.build_quality_reporting import build_quality_reporting
from src.insights.value_based_care.build_value_based_reporting import (
    build_value_based_reporting,
)


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "insights/insights.yaml"

DEFAULT_LAYER_NAME = "Layer 3 - Insights"
DEFAULT_DOMAIN_NAME = "Insights Platform"
DOMAIN_SECTION = "insights"

LOGGER_NAME = "medfabric.insights.platform"


###############################################################################
# Type Aliases
###############################################################################

DomainBuilder = Callable[[str], InsightsBuildResult]


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict) -> None:
    """
    Purpose
    -------
    Validate required Insights Platform configuration sections.

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
    This validation checks the platform-level contract. Each reporting-domain
    module also validates its own domain-specific configuration.
    """

    required_sections = [
        "insights",
        "paths",
        "join_keys",
        "executive_insights",
        "financial_reporting",
        "clinical_reporting",
        "population_reporting",
        "provider_reporting",
        "quality_reporting",
        "care_management_reporting",
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

    if missing_sections:
        raise PipelineError(
            "Insights Platform configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> InsightsDomainRuntime:
    """
    Purpose
    -------
    Initialize the Insights Platform runtime.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration relative to the config folder.

    Returns
    -------
    InsightsDomainRuntime
        Initialized runtime for the full Insights Platform orchestrator.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This platform runtime records orchestration-level audit, metadata, and rule
    catalog entries. Individual reporting domains create their own runtimes when
    they are executed.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 3 - Insights Platform"
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
        message="Insights Platform runtime initialized successfully.",
        source_layer="Layer 3 - Insights",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Domain Registry
###############################################################################

def get_domain_builders() -> List[Dict[str, object]]:
    """
    Purpose
    -------
    Return the ordered Insights reporting-domain build registry.

    Parameters
    ----------
    None

    Returns
    -------
    list[dict]
        Ordered reporting-domain metadata and builder functions.

    Raises
    ------
    None

    Notes
    -----
    Execution order is intentionally explicit. Executive Insights runs first so
    an initial executive scorecard can be generated directly from Layer 2. The
    remaining reporting domains then build their reporting summaries.
    """

    return [
        {
            "domain_key": "executive_insights",
            "domain_name": "Executive Insights",
            "builder": build_executive_insights,
        },
        {
            "domain_key": "financial_reporting",
            "domain_name": "Financial Reporting",
            "builder": build_financial_reporting,
        },
        {
            "domain_key": "clinical_reporting",
            "domain_name": "Clinical Reporting",
            "builder": build_clinical_reporting,
        },
        {
            "domain_key": "population_reporting",
            "domain_name": "Population Reporting",
            "builder": build_population_reporting,
        },
        {
            "domain_key": "provider_reporting",
            "domain_name": "Provider Reporting",
            "builder": build_provider_reporting,
        },
        {
            "domain_key": "quality_reporting",
            "domain_name": "Quality Reporting",
            "builder": build_quality_reporting,
        },
        {
            "domain_key": "care_management_reporting",
            "domain_name": "Care Management Reporting",
            "builder": build_care_management_reporting,
        },
        {
            "domain_key": "value_based_care_reporting",
            "domain_name": "Value-Based Care Reporting",
            "builder": build_value_based_reporting,
        },
    ]


def is_domain_enabled(
    runtime: InsightsDomainRuntime,
    domain_key: str,
) -> bool:
    """
    Purpose
    -------
    Determine whether a reporting domain is enabled in configuration.

    Parameters
    ----------
    runtime:
        Insights Platform runtime.

    domain_key:
        Reporting domain configuration key.

    Returns
    -------
    bool
        True when enabled, otherwise False.

    Raises
    ------
    None
    """

    return bool(runtime.config.get(domain_key, {}).get("enabled", True))


###############################################################################
# Platform Execution Records
###############################################################################

def build_domain_execution_records(
    runtime: InsightsDomainRuntime,
    domain_results: List[InsightsBuildResult],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build domain execution records for the Insights Platform.

    Parameters
    ----------
    runtime:
        Insights Platform runtime.

    domain_results:
        Results returned by domain builders.

    Returns
    -------
    pandas.DataFrame
        One row per reporting-domain execution result.

    Raises
    ------
    None

    Notes
    -----
    This output is used for platform-level metadata and troubleshooting. It is
    not currently written as a business output because the Insights Platform
    only writes metadata and audit at the orchestrator level.
    """

    rows = []

    for result in domain_results:
        rows.append(
            {
                "run_id": runtime.context.run_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "child_domain_name": result.name,
                "child_domain_status": result.status,
                "child_domain_message": result.message,
                "child_domain_row_count": int(result.row_count),
                "child_domain_column_count": int(result.column_count),
                "event_timestamp_utc": utc_now().isoformat(),
            }
        )

    return pd.DataFrame(rows)


def register_platform_metadata(
    runtime: InsightsDomainRuntime,
    domain_results: List[InsightsBuildResult],
    domain_execution_df: pd.DataFrame,
) -> None:
    """
    Purpose
    -------
    Register platform-level metadata and rule records.

    Parameters
    ----------
    runtime:
        Insights Platform runtime.

    domain_results:
        Results returned by reporting-domain builders.

    domain_execution_df:
        Domain execution summary dataframe.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This gives the Insights Platform a catalog record describing the orchestrator
    run and the reporting domains it executed.
    """

    add_dataset_record(
        runtime=runtime,
        dataset_name="insights_platform_domain_execution",
        dataset_type="insights_platform_metadata",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(domain_execution_df),
        column_count=len(domain_execution_df.columns),
        message="Insights Platform domain execution records built successfully.",
        source_layer="Layer 3 - Insights",
        source_dataset="reporting_domain_results",
    )

    for result in domain_results:
        add_rule_record(
            runtime=runtime,
            rule_group="insights_platform_orchestration",
            rule_name=result.name,
            rule_type="domain_execution",
            description=f"Executed Insights reporting domain: {result.name}",
            source_dataset="insights_reporting_domain",
            source_layer="Layer 3 - Insights",
            rule_config={
                "status": result.status,
                "message": result.message,
                "row_count": result.row_count,
                "column_count": result.column_count,
            },
        )


###############################################################################
# Domain Orchestration
###############################################################################

def run_reporting_domains(
    runtime: InsightsDomainRuntime,
    config_path: str,
) -> List[InsightsBuildResult]:
    """
    Purpose
    -------
    Run all enabled Insights reporting domains.

    Parameters
    ----------
    runtime:
        Insights Platform runtime.

    config_path:
        Configuration path passed to each domain builder.

    Returns
    -------
    list[InsightsBuildResult]
        Reporting-domain build results.

    Raises
    ------
    None

    Notes
    -----
    This function does not raise when a reporting domain fails. Instead, it
    records the failure and returns a failed result so the orchestrator can write
    platform-level audit outputs.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    domain_results: List[InsightsBuildResult] = []

    for domain_entry in get_domain_builders():
        domain_key = str(domain_entry["domain_key"])
        domain_name = str(domain_entry["domain_name"])
        builder = domain_entry["builder"]

        if not is_domain_enabled(runtime, domain_key):
            message = f"Insights reporting domain disabled: {domain_name}"

            logger.info("SKIP: %s", message)

            add_warning_audit(
                runtime=runtime,
                step_name=f"run_domain:{domain_key}",
                message=message,
                source_layer="Layer 3 - Insights",
                source_dataset=domain_key,
            )

            domain_results.append(
                InsightsBuildResult(
                    name=domain_key,
                    status=STATUS_SKIPPED,
                    message=message,
                    row_count=0,
                    column_count=0,
                )
            )

            continue

        logger.info("=" * 80)
        logger.info("START Insights reporting domain: %s", domain_name)
        logger.info("=" * 80)

        try:
            result = builder(config_path)

            domain_results.append(result)

            if result.status == STATUS_SUCCESS:
                add_success_audit(
                    runtime=runtime,
                    step_name=f"run_domain:{domain_key}",
                    message=result.message,
                    row_count=result.row_count,
                    source_layer="Layer 3 - Insights",
                    source_dataset=domain_key,
                )
            else:
                add_failed_audit(
                    runtime=runtime,
                    step_name=f"run_domain:{domain_key}",
                    message=result.message,
                    row_count=result.row_count,
                    source_layer="Layer 3 - Insights",
                    source_dataset=domain_key,
                )

            logger.info(
                "COMPLETE Insights reporting domain: %s | Status: %s | Rows: %s",
                domain_name,
                result.status,
                result.row_count,
            )

        except Exception as exc:
            error_message = f"{domain_name} failed unexpectedly: {exc}"

            logger.error(error_message)
            logger.error("Traceback:\n%s", traceback.format_exc())

            add_failed_audit(
                runtime=runtime,
                step_name=f"run_domain:{domain_key}",
                message=error_message,
                source_layer="Layer 3 - Insights",
                source_dataset=domain_key,
            )

            domain_results.append(
                InsightsBuildResult(
                    name=domain_key,
                    status=STATUS_FAILED,
                    message=error_message,
                    row_count=0,
                    column_count=0,
                )
            )

    return domain_results


###############################################################################
# Platform Result Helpers
###############################################################################

def determine_platform_status(domain_results: List[InsightsBuildResult]) -> str:
    """
    Purpose
    -------
    Determine the overall Insights Platform status from domain results.

    Parameters
    ----------
    domain_results:
        Reporting-domain build results.

    Returns
    -------
    str
        SUCCESS, WARNING, or FAILED.

    Raises
    ------
    None

    Notes
    -----
    Any failed reporting domain makes the platform FAILED. Skipped domains are
    treated as WARNING because the platform ran but not all domains produced
    outputs.
    """

    statuses = [result.status for result in domain_results]

    if any(status == STATUS_FAILED for status in statuses):
        return STATUS_FAILED

    if any(status == STATUS_SKIPPED for status in statuses):
        return STATUS_WARNING

    return STATUS_SUCCESS


def build_platform_message(platform_status: str) -> str:
    """
    Purpose
    -------
    Build a standard platform message from the final status.

    Parameters
    ----------
    platform_status:
        Final platform status.

    Returns
    -------
    str
        Human-readable platform message.

    Raises
    ------
    None
    """

    if platform_status == STATUS_SUCCESS:
        return "Insights Platform completed successfully."

    if platform_status == STATUS_WARNING:
        return "Insights Platform completed with warnings."

    return "Insights Platform failed."


###############################################################################
# Main Orchestration
###############################################################################

def build_insights_platform(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> InsightsBuildResult:
    """
    Purpose
    -------
    Build the full MedFabric Insights Platform.

    Parameters
    ----------
    config_path:
        Path to the Insights YAML configuration relative to the config folder.

    Returns
    -------
    InsightsBuildResult
        Standard build result for the full Insights Platform.

    Raises
    ------
    None
        Exceptions are captured and returned as failed build results.

    Notes
    -----
    This is the Layer 3 orchestrator and should be the primary command used to
    rebuild all Insights outputs from Layer 2 Analytics Platform outputs.
    """

    runtime: Optional[InsightsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Insights Platform started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        domain_results = run_reporting_domains(
            runtime=runtime,
            config_path=config_path,
        )

        domain_execution_df = build_domain_execution_records(
            runtime=runtime,
            domain_results=domain_results,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "insights_platform_domain_execution": domain_execution_df,
        }

        register_platform_metadata(
            runtime=runtime,
            domain_results=domain_results,
            domain_execution_df=domain_execution_df,
        )

        platform_status = determine_platform_status(domain_results)
        platform_message = build_platform_message(platform_status)

        if platform_status == STATUS_SUCCESS:
            add_success_audit(
                runtime=runtime,
                step_name="build_insights_platform",
                message=platform_message,
                row_count=sum(result.row_count for result in domain_results),
                source_layer="Layer 3 - Insights",
                source_dataset="insights_reporting_domains",
            )
        elif platform_status == STATUS_WARNING:
            add_warning_audit(
                runtime=runtime,
                step_name="build_insights_platform",
                message=platform_message,
                row_count=sum(result.row_count for result in domain_results),
                source_layer="Layer 3 - Insights",
                source_dataset="insights_reporting_domains",
            )
        else:
            add_failed_audit(
                runtime=runtime,
                step_name="build_insights_platform",
                message=platform_message,
                row_count=sum(result.row_count for result in domain_results),
                source_layer="Layer 3 - Insights",
                source_dataset="insights_reporting_domains",
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
        logger.info("MedFabric Insights Platform completed")
        logger.info("Status: %s", platform_status)
        logger.info("=" * 80)

        return InsightsBuildResult(
            name="insights_platform",
            status=platform_status,
            message=platform_message,
            row_count=sum(result.row_count for result in domain_results),
            column_count=sum(result.column_count for result in domain_results),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Insights Platform failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_insights_platform",
                message=str(exc),
                source_layer="Layer 3 - Insights",
                source_dataset="insights_platform",
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
                    audit_records_name="insights_audit_records",
                    validation_results_name="insights_validation_results",
                    execution_summary_name="insights_execution_summary",
                )

            except Exception as audit_exc:
                logger.error(
                    "Failed to write Insights Platform audit outputs: %s",
                    audit_exc,
                )

        return InsightsBuildResult(
            name="insights_platform",
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
    Command-line entry point for the Insights Platform.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Raised with exit code 1 when the Insights Platform fails.
    """

    config_path = os.environ.get(
        "MEDFABRIC_INSIGHTS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_insights_platform(config_path=config_path)

    if result.status in {STATUS_SUCCESS, STATUS_WARNING}:
        print(result.message)
        return

    print(f"Insights Platform failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()