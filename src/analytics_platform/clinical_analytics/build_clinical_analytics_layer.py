###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/clinical_analytics/build_clinical_analytics_layer.py
#
# Layer:
#     Layer 2B - Clinical Analytics
#
# Purpose:
#     Builds Clinical Analytics outputs using the configured Clinical Analytics
#     YAML contract:
#
#         config/analytics_platform/clinical_analytics.yaml
#
# Business Context:
#     Clinical Analytics identifies clinically relevant member populations using
#     available enterprise data assets.
#
#     The current MedFabric Feature Store provides aggregate member-level
#     clinical signals, not diagnosis-line detail. Therefore, this builder
#     creates signal-based clinical registries. These registries are intentionally
#     designed so they can later be upgraded to ICD-based registries when
#     diagnosis-level evidence is exposed from Silver, Gold, or a future clinical
#     feature group.
#
# Architecture:
#     This file contains Clinical Analytics business logic only.
#
#     Shared Analytics Platform concerns are handled by:
#         - src.analytics_platform.common.runtime
#         - src.analytics_platform.common.io
#         - src.analytics_platform.common.audit
#         - src.analytics_platform.common.validation
#         - src.analytics_platform.common.metadata
#         - src.analytics_platform.common.rules
#
# Inputs:
#     config/analytics_platform/clinical_analytics.yaml
#
# Outputs:
#     data/analytics_platform/clinical_analytics/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.clinical_analytics.build_clinical_analytics_layer
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
    add_warning_audit,
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
from src.analytics_platform.common.validation import require_columns
from src.common.exception_manager import PipelineError, ValidationError
from src.common.pipeline_context import create_pipeline_context


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/analytics_platform/clinical_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2B - Clinical Analytics"
DEFAULT_DOMAIN_NAME = "Clinical Analytics"
DOMAIN_SECTION = "clinical_analytics"

LOGGER_NAME = "medfabric.analytics_platform.clinical_analytics"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Clinical Analytics configuration sections.

    Parameters
    ----------
    config:
        Loaded Clinical Analytics YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required configuration sections are missing.

    Notes
    -----
    Generic YAML loading is handled by ConfigurationManager. This function only
    validates the Clinical Analytics domain contract.
    """

    required_sections = [
        "clinical_analytics",
        "paths",
        "join_keys",
        "registry_framework",
        "condition_registry",
        "disease_registries",
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
            "Clinical Analytics configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Clinical Analytics runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Clinical Analytics configuration path.

    Returns
    -------
    AnalyticsDomainRuntime
        Initialized domain runtime.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This replaces local YAML loading, local logging setup, local path handling,
    and local runtime objects with shared Layer 0 and Layer 2 common services.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2B - Clinical Analytics"
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
        message="Clinical Analytics runtime initialized successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Business Helper Functions
###############################################################################

def build_base_member_universe(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build base member universe for Clinical Analytics.

    Parameters
    ----------
    runtime:
        Clinical Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Member-level universe dataframe.

    Raises
    ------
    ValidationError
        Raised when configured base member dataset or member key is missing.

    Notes
    -----
    This function is available for future registry rules that need a full member
    universe before applying clinical evidence filters.
    """

    framework_config = runtime.config.get("registry_framework", {})
    base_dataset_name = framework_config.get("base_member_dataset", "demographic_features")
    member_key = framework_config.get("member_key", "member_id")

    if base_dataset_name not in datasets:
        raise ValidationError(f"Base member dataset missing: {base_dataset_name}")

    base_df = datasets[base_dataset_name]

    require_columns(
        runtime=runtime,
        dataframe=base_df,
        dataset_name=base_dataset_name,
        required_columns=[member_key],
        source_layer="Layer 1F - Feature Store",
        source_dataset=base_dataset_name,
    )

    universe_df = base_df[[member_key]].drop_duplicates().copy()

    for enrichment_dataset_name in framework_config.get("enrichment_datasets", []):
        if enrichment_dataset_name not in datasets:
            continue

        enrichment_df = datasets[enrichment_dataset_name]

        if member_key not in enrichment_df.columns:
            continue

        enrichment_deduped = enrichment_df.drop_duplicates(subset=[member_key]).copy()
        universe_df = universe_df.merge(enrichment_deduped, on=member_key, how="left")

    return universe_df


###############################################################################
# Condition Registry
###############################################################################

def build_condition_registry(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build signal-based condition registry.

    Parameters
    ----------
    runtime:
        Clinical Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Signal-based condition registry.

    Raises
    ------
    ValidationError
        Raised when required source columns are missing.

    Notes
    -----
    This registry is intentionally signal-based because current Layer 1 assets
    expose aggregate clinical signals rather than diagnosis-line detail.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("condition_registry", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Condition registry disabled")
        return pd.DataFrame()

    member_key = config.get("member_key", "member_id")
    condition_frames: List[pd.DataFrame] = []

    logger.info("START: Build signal-based condition registry")

    for signal_name, signal_config in config.get("condition_signals", {}).items():
        condition_name = signal_config.get("condition_name", signal_name)
        condition_category = signal_config.get("condition_category", "")
        source_dataset_name = signal_config.get("source_dataset")
        signal_column = signal_config.get("signal_column")
        operator = signal_config.get("operator")
        value = signal_config.get("value")
        evidence_column = signal_config.get("evidence_column", signal_column)

        add_rule_record(
            runtime=runtime,
            rule_group="condition_registry",
            rule_name=signal_name,
            rule_type="signal_based_condition",
            description=f"Signal-based condition rule for {condition_name}.",
            source_dataset=str(source_dataset_name),
            source_layer="Layer 1F - Feature Store",
            rule_config=signal_config,
        )

        if source_dataset_name not in datasets:
            add_warning_audit(
                runtime=runtime,
                step_name=f"condition_signal:{signal_name}",
                message=(
                    f"Skipping condition signal because source dataset is missing: "
                    f"{source_dataset_name}"
                ),
                source_layer="Layer 1F - Feature Store",
                source_dataset=str(source_dataset_name),
            )
            continue

        source_df = datasets[source_dataset_name]

        required_columns = [member_key, signal_column]
        if evidence_column and evidence_column not in required_columns:
            required_columns.append(evidence_column)

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=required_columns,
            source_layer="Layer 1F - Feature Store",
            source_dataset=source_dataset_name,
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched_df = source_df.loc[mask, [member_key, evidence_column]].copy()
        matched_df = matched_df.rename(
            columns={evidence_column: "condition_evidence_value"}
        )

        matched_df["condition_name"] = condition_name
        matched_df["condition_category"] = condition_category
        matched_df["condition_source"] = source_dataset_name
        matched_df["condition_rule"] = signal_name
        matched_df["condition_evidence_count"] = 1
        matched_df["analytics_layer_run_id"] = runtime.context.run_id
        matched_df["analytics_domain"] = runtime.domain_name
        matched_df["analytics_asset_name"] = "condition_registry"
        matched_df["source_layer"] = "Layer 1F - Feature Store"
        matched_df["source_dataset"] = source_dataset_name
        matched_df["built_at_utc"] = utc_now().isoformat()

        condition_frames.append(matched_df)

        logger.info(
            "Built condition signal: %s | Members: %s",
            signal_name,
            len(matched_df),
        )

    if condition_frames:
        output_df = pd.concat(condition_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "condition_evidence_value",
                "condition_name",
                "condition_category",
                "condition_source",
                "condition_rule",
                "condition_evidence_count",
                "analytics_layer_run_id",
                "analytics_domain",
                "analytics_asset_name",
                "source_layer",
                "source_dataset",
                "built_at_utc",
            ]
        )

    add_dataset_record(
        runtime=runtime,
        dataset_name="condition_registry",
        dataset_type="clinical_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Signal-based condition registry built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset="configured_condition_signals",
    )

    logger.info("COMPLETE: Build condition registry | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Disease Registries
###############################################################################

def build_single_disease_registry(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
    registry_key: str,
    registry_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build one signal-based disease registry.

    Parameters
    ----------
    runtime:
        Clinical Analytics runtime.

    datasets:
        Loaded input datasets.

    registry_key:
        Registry key from configuration.

    registry_config:
        Registry rule configuration.

    Returns
    -------
    pandas.DataFrame
        Disease registry output.

    Raises
    ------
    ValidationError
        Raised when source dataset or required source columns are missing.

    Notes
    -----
    Disease registries currently use aggregate signals. The output schema keeps
    source and evidence columns explicit so the logic can later be upgraded to
    diagnosis-code evidence without breaking downstream consumers.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    source_dataset_name = registry_config.get("source_dataset")
    signal_column = registry_config.get("signal_column")
    operator = registry_config.get("operator")
    value = registry_config.get("value")
    evidence_column = registry_config.get("evidence_column", signal_column)

    registry_name = registry_config.get("registry_name", registry_key)
    condition_name = registry_config.get("condition_name", registry_name)
    condition_category = registry_config.get("condition_category", "")
    registry_method = registry_config.get("registry_method", "signal_based_placeholder")
    note = registry_config.get("note", "")

    add_rule_record(
        runtime=runtime,
        rule_group="disease_registries",
        rule_name=registry_key,
        rule_type=registry_method,
        description=note,
        source_dataset=str(source_dataset_name),
        source_layer="Layer 1F - Feature Store",
        rule_config=registry_config,
    )

    if source_dataset_name not in datasets:
        raise ValidationError(
            f"Disease registry source dataset missing for {registry_key}: "
            f"{source_dataset_name}"
        )

    source_df = datasets[source_dataset_name]

    required_columns = [member_key, signal_column]
    if evidence_column and evidence_column not in required_columns:
        required_columns.append(evidence_column)

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=required_columns,
        source_layer="Layer 1F - Feature Store",
        source_dataset=source_dataset_name,
    )

    mask = apply_operator(source_df[signal_column], operator, value)

    registry_df = source_df.loc[mask, [member_key, evidence_column]].copy()
    registry_df = registry_df.rename(
        columns={evidence_column: "registry_evidence_value"}
    )

    registry_df["registry_key"] = registry_key
    registry_df["registry_name"] = registry_name
    registry_df["condition_name"] = condition_name
    registry_df["condition_category"] = condition_category
    registry_df["registry_method"] = registry_method
    registry_df["source_dataset"] = source_dataset_name
    registry_df["signal_column"] = signal_column
    registry_df["registry_note"] = note
    registry_df["analytics_layer_run_id"] = runtime.context.run_id
    registry_df["analytics_domain"] = runtime.domain_name
    registry_df["analytics_asset_name"] = registry_config.get("output_name", registry_key)
    registry_df["source_layer"] = "Layer 1F - Feature Store"
    registry_df["built_at_utc"] = utc_now().isoformat()

    return registry_df


def build_disease_registries(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Purpose
    -------
    Build all configured signal-based disease registries.

    Parameters
    ----------
    runtime:
        Clinical Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Disease registry outputs keyed by output asset name.

    Raises
    ------
    ValidationError
        Raised when any enabled disease registry has invalid required inputs.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    registries_config = runtime.config.get("disease_registries", {})

    registry_outputs: Dict[str, pd.DataFrame] = {}

    logger.info("START: Build signal-based disease registries")

    for registry_key, registry_config in registries_config.items():
        if not bool(registry_config.get("enabled", True)):
            logger.info("SKIP registry disabled: %s", registry_key)
            continue

        output_name = registry_config.get("output_name", f"{registry_key}_registry")

        registry_df = build_single_disease_registry(
            runtime=runtime,
            datasets=datasets,
            registry_key=registry_key,
            registry_config=registry_config,
        )

        registry_outputs[output_name] = registry_df

        add_dataset_record(
            runtime=runtime,
            dataset_name=output_name,
            dataset_type="clinical_analytics_output",
            status=STATUS_SUCCESS,
            path=None,
            row_count=len(registry_df),
            column_count=len(registry_df.columns),
            message=f"Signal-based disease registry built successfully: {output_name}",
            source_layer="Layer 1F - Feature Store",
            source_dataset=registry_config.get("source_dataset"),
        )

        logger.info(
            "Built disease registry: %s | Rows: %s",
            output_name,
            len(registry_df),
        )

    logger.info("COMPLETE: Build disease registries | Count: %s", len(registry_outputs))

    return registry_outputs


###############################################################################
# Main Orchestration
###############################################################################

def build_clinical_analytics_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Clinical Analytics layer.

    Parameters
    ----------
    config_path:
        Clinical Analytics configuration path.

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
    This is the public build entry point used by the Analytics Platform master
    orchestrator.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Clinical Analytics started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "demographic_features": "Layer 1F - Feature Store",
                "claims_features": "Layer 1F - Feature Store",
                "laboratory_features": "Layer 1F - Feature Store",
                "pharmacy_features": "Layer 1F - Feature Store",
                "risk_features": "Layer 1F - Feature Store",
                "member_360": "Layer 1E - Gold",
                "member_360_semantic_view": "Layer 1G - Semantic Layer",
            },
            key_column_map={
                "demographic_features": "member_id",
                "claims_features": "member_id",
                "laboratory_features": "member_id",
                "pharmacy_features": "member_id",
                "risk_features": "member_id",
                "member_360": "member_id",
                "member_360_semantic_view": "member_id",
            },
        )

        condition_registry = build_condition_registry(runtime, datasets)
        disease_registry_outputs = build_disease_registries(runtime, datasets)

        output_assets: Dict[str, pd.DataFrame] = {
            "condition_registry": condition_registry,
        }
        output_assets.update(disease_registry_outputs)

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_clinical_analytics_layer",
            message="Clinical Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2B - Clinical Analytics",
            source_dataset="clinical_analytics_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="clinical_analytics_dataset_inventory",
            column_dictionary_name="clinical_analytics_column_dictionary",
            rule_catalog_name="clinical_analytics_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="clinical_analytics_audit_records",
            validation_results_name="clinical_analytics_validation_results",
            execution_summary_name="clinical_analytics_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Clinical Analytics completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name="clinical_analytics",
            status=STATUS_SUCCESS,
            message="Clinical Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(len(dataframe.columns) for dataframe in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Clinical Analytics failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_clinical_analytics_layer",
                message=str(exc),
                source_layer="Layer 2B - Clinical Analytics",
                source_dataset="clinical_analytics",
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
                    audit_records_name="clinical_analytics_audit_records",
                    validation_results_name="clinical_analytics_validation_results",
                    execution_summary_name="clinical_analytics_execution_summary",
                )
            except Exception as audit_exc:
                logger.error("Failed to write audit outputs: %s", audit_exc)

        return AnalyticsBuildResult(
            name="clinical_analytics",
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
    Command-line entry point for Clinical Analytics.

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
        "MEDFABRIC_CLINICAL_ANALYTICS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_clinical_analytics_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Clinical Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()