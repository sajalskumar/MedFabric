###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/population_health/build_population_health_layer.py
#
# Layer:
#     Layer 2A - Population Health Analytics
#
# Purpose:
#     Builds the Population Health Analytics layer from MedFabric Layer 1 outputs.
#
#     This module consumes configured Gold, Feature Store, and Semantic Layer
#     datasets and produces Population Health analytics assets including:
#
#         - Population cohorts
#         - Risk stratification
#         - Member segmentation
#         - Provider attribution analytics
#         - Dataset inventory
#         - Column dictionary
#         - Rule catalog
#         - Validation results
#         - Audit records
#         - Execution summary
#
# Business Context:
#     Population Health Analytics is the first domain in Layer 2 because it
#     establishes reusable analytics assets for downstream clinical analytics,
#     quality analytics, predictive analytics, care management, provider
#     analytics, and value-based care.
#
# Architecture:
#     This file now contains Population Health business logic only.
#
#     Reusable Analytics Platform concerns are handled by:
#         - src.analytics_platform.common.runtime
#         - src.analytics_platform.common.io
#         - src.analytics_platform.common.audit
#         - src.analytics_platform.common.validation
#         - src.analytics_platform.common.metadata
#
#     Platform-wide services are still handled by:
#         - src.common.pipeline_context
#         - src.common.configuration_manager
#         - src.common.path_manager
#         - src.common.storage_manager
#         - src.common.logging_manager
#         - src.common.validation_manager
#         - src.common.metadata_manager
#
# Inputs:
#     config/analytics_platform/population_health.yaml
#
# Outputs:
#     data/analytics_platform/population_health/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.population_health.build_population_health_layer
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
)
from src.common.exception_manager import PipelineError, ValidationError
from src.common.pipeline_context import create_pipeline_context

from src.analytics_platform.common.rules import apply_operator


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/analytics_platform/population_health.yaml"

DEFAULT_LAYER_NAME = "Layer 2A - Population Health Analytics"
DEFAULT_DOMAIN_NAME = "Population Health"
DOMAIN_SECTION = "population_health"

LOGGER_NAME = "medfabric.analytics_platform.population_health"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Population Health configuration sections.

    Parameters
    ----------
    config:
        Loaded Population Health YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required sections are missing.

    Notes
    -----
    This validation is intentionally domain-specific. Generic YAML loading is
    handled by ConfigurationManager through PipelineContext.
    """

    required_sections = [
        "population_health",
        "paths",
        "join_keys",
        "population_cohorts",
        "risk_stratification",
        "member_segmentation",
        "provider_attribution_analytics",
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
            "Population Health configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Population Health runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Population Health configuration path.

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
    This replaces the old local runtime dataclass, local YAML loading, local
    logging setup, and local path setup. Those concerns now come from shared
    MedFabric foundation services.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2A - Population Health Analytics"
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
        message="Population Health runtime initialized successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset=config_file,
    )

    return runtime


def resolve_existing_column(
    dataframe: pd.DataFrame,
    preferred_column: Optional[str],
    fallback_columns: List[str],
) -> Optional[str]:
    """
    Purpose
    -------
    Resolve a usable column from a preferred column and fallback columns.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    preferred_column:
        Preferred configured column.

    fallback_columns:
        Fallback columns allowed by the Population Health domain.

    Returns
    -------
    str | None
        Existing column name if found.

    Raises
    ------
    None

    Notes
    -----
    This prevents brittle failures when upstream Layer 1 data uses an available
    equivalent feature, such as latest_sdoh_risk_score instead of
    composite_risk_score.
    """

    if preferred_column and preferred_column in dataframe.columns:
        return preferred_column

    for column in fallback_columns:
        if column in dataframe.columns:
            return column

    return None


def build_member_universe(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
    preferred_dataset_name: str,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build the base member universe for Population Health outputs.

    Parameters
    ----------
    runtime:
        Population Health runtime.

    datasets:
        Loaded input datasets.

    preferred_dataset_name:
        Preferred dataset for deriving member universe.

    Returns
    -------
    pandas.DataFrame
        Distinct member universe.

    Raises
    ------
    ValidationError
        Raised when no valid member-level source exists.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    if preferred_dataset_name in datasets:
        source_dataset_name = preferred_dataset_name
    elif "member_360_semantic_view" in datasets:
        source_dataset_name = "member_360_semantic_view"
    elif "demographic_features" in datasets:
        source_dataset_name = "demographic_features"
    else:
        raise ValidationError(
            "Unable to build member universe. No member-level dataset found."
        )

    source_df = datasets[source_dataset_name]

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=[member_key],
        source_layer="Layer 1",
        source_dataset=source_dataset_name,
    )

    universe_df = source_df[[member_key]].drop_duplicates().copy()

    universe_df["analytics_layer_run_id"] = runtime.context.run_id
    universe_df["analytics_domain"] = runtime.domain_name
    universe_df["built_at_utc"] = utc_now().isoformat()

    return universe_df


###############################################################################
# Population Cohorts
###############################################################################

def build_population_cohorts(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build configured Population Health cohorts.

    Parameters
    ----------
    runtime:
        Population Health runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Population cohort output.

    Raises
    ------
    ValidationError
        Raised when required member-level columns are missing.

    Notes
    -----
    Cohorts are built from YAML rules. The all_members rule uses the configured
    member universe. Numeric threshold cohorts use configured feature datasets.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("population_cohorts", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Population cohorts disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    source_dataset_name = config.get("source_dataset", "member_360_semantic_view")

    member_universe_df = build_member_universe(
        runtime=runtime,
        datasets=datasets,
        preferred_dataset_name=source_dataset_name,
    )

    cohort_frames: List[pd.DataFrame] = []

    logger.info("START: Build population cohorts")

    for cohort_name, cohort_config in config.get("cohorts", {}).items():
        if not bool(cohort_config.get("enabled", True)):
            continue

        rule_type = cohort_config.get("rule_type")
        description = cohort_config.get("description", "")
        rule_source_dataset = cohort_config.get("source_dataset", source_dataset_name)

        add_rule_record(
            runtime=runtime,
            rule_group="population_cohorts",
            rule_name=cohort_name,
            rule_type=str(rule_type),
            description=description,
            source_dataset=rule_source_dataset,
            source_layer="Layer 1",
            rule_config=cohort_config,
        )

        if rule_type == "all_members":
            cohort_df = member_universe_df[[member_key]].copy()

        elif rule_type == "numeric_threshold":
            if rule_source_dataset not in datasets:
                add_warning_audit(
                    runtime=runtime,
                    step_name=f"build_population_cohort:{cohort_name}",
                    message=(
                        f"Skipping cohort because source dataset is missing: "
                        f"{rule_source_dataset}"
                    ),
                    source_layer="Layer 1",
                    source_dataset=rule_source_dataset,
                )
                continue

            source_df = datasets[rule_source_dataset]
            column = cohort_config.get("column")
            operator = cohort_config.get("operator")
            threshold = cohort_config.get("threshold")

            resolved_column = resolve_existing_column(
                dataframe=source_df,
                preferred_column=column,
                fallback_columns=[
                    "latest_sdoh_risk_score",
                    "total_paid_amount",
                    "total_claims",
                    "total_lab_results",
                    "total_pharmacy_claims",
                ],
            )

            if resolved_column is None:
                add_warning_audit(
                    runtime=runtime,
                    step_name=f"build_population_cohort:{cohort_name}",
                    message=(
                        f"Skipping cohort because configured column was not "
                        f"available: {column}"
                    ),
                    source_layer="Layer 1",
                    source_dataset=rule_source_dataset,
                )
                continue

            require_columns(
                runtime=runtime,
                dataframe=source_df,
                dataset_name=rule_source_dataset,
                required_columns=[member_key, resolved_column],
                source_layer="Layer 1",
                source_dataset=rule_source_dataset,
            )

            mask = apply_operator(source_df[resolved_column], operator, threshold)
            cohort_df = source_df.loc[mask, [member_key]].drop_duplicates().copy()

        else:
            add_warning_audit(
                runtime=runtime,
                step_name=f"build_population_cohort:{cohort_name}",
                message=f"Unsupported cohort rule_type: {rule_type}",
                source_layer="Layer 2A - Population Health Analytics",
                source_dataset="population_cohorts",
            )
            continue

        cohort_df["cohort_name"] = cohort_name
        cohort_df["cohort_description"] = description
        cohort_df["analytics_layer_run_id"] = runtime.context.run_id
        cohort_df["analytics_domain"] = runtime.domain_name
        cohort_df["analytics_asset_name"] = "population_cohorts"
        cohort_df["source_layer"] = "Layer 1"
        cohort_df["source_dataset"] = rule_source_dataset
        cohort_df["built_at_utc"] = utc_now().isoformat()

        cohort_frames.append(cohort_df)

        logger.info(
            "Built cohort: %s | Members: %s",
            cohort_name,
            len(cohort_df),
        )

    if cohort_frames:
        output_df = pd.concat(cohort_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "cohort_name",
                "cohort_description",
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
        dataset_name="population_cohorts",
        dataset_type="population_health_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Population cohorts built successfully.",
        source_layer="Layer 1",
        source_dataset=source_dataset_name,
    )

    logger.info("COMPLETE: Build population cohorts | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Risk Stratification
###############################################################################

def build_risk_stratification(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build member-level risk stratification.

    Parameters
    ----------
    runtime:
        Population Health runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Risk stratification output.

    Raises
    ------
    ValidationError
        Raised when the configured source dataset or member key is invalid.

    Notes
    -----
    The preferred risk score column is read from configuration. If unavailable,
    this function falls back to risk-like columns that exist in the current
    Feature Store output.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("risk_stratification", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Risk stratification disabled")
        return pd.DataFrame()

    source_dataset_name = config.get("source_dataset", "risk_features")
    member_key = config.get(
        "member_key",
        runtime.config.get("join_keys", {}).get("member_key", "member_id"),
    )

    configured_risk_score_column = config.get(
        "risk_score_column",
        "composite_risk_score",
    )

    if source_dataset_name not in datasets:
        raise ValidationError(
            f"Risk stratification source dataset missing: {source_dataset_name}"
        )

    source_df = datasets[source_dataset_name]

    risk_score_column = resolve_existing_column(
        dataframe=source_df,
        preferred_column=configured_risk_score_column,
        fallback_columns=[
            "composite_risk_score",
            "latest_sdoh_risk_score",
            "high_cost_risk_signal",
            "sdoh_risk_signal",
            "clinical_risk_signal",
            "total_paid_amount",
        ],
    )

    if risk_score_column is None:
        raise ValidationError(
            "Risk stratification could not find a usable risk score column. "
            f"Configured column: {configured_risk_score_column}"
        )

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=[member_key, risk_score_column],
        source_layer="Layer 1F - Feature Store",
        source_dataset=source_dataset_name,
    )

    output_df = (
        source_df[[member_key, risk_score_column]]
        .drop_duplicates(subset=[member_key])
        .copy()
    )

    output_df["risk_tier"] = "Unassigned"
    output_df["risk_priority_rank"] = None

    for tier_name, tier_config in config.get("tiers", {}).items():
        min_value = tier_config.get("min_value")
        max_value = tier_config.get("max_value")
        label = tier_config.get("label", tier_name)
        priority_rank = tier_config.get("priority_rank")

        add_rule_record(
            runtime=runtime,
            rule_group="risk_stratification",
            rule_name=tier_name,
            rule_type="range",
            description=f"Risk tier assignment for {label}.",
            source_dataset=source_dataset_name,
            source_layer="Layer 1F - Feature Store",
            rule_config=tier_config,
        )

        mask = (
            (output_df[risk_score_column] >= min_value)
            & (output_df[risk_score_column] <= max_value)
        )

        output_df.loc[mask, "risk_tier"] = label
        output_df.loc[mask, "risk_priority_rank"] = priority_rank

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "risk_stratification"
    output_df["source_layer"] = "Layer 1F - Feature Store"
    output_df["source_dataset"] = source_dataset_name
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="risk_stratification",
        dataset_type="population_health_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Risk stratification built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset=source_dataset_name,
    )

    logger.info("COMPLETE: Build risk stratification | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Member Segmentation
###############################################################################

def prepare_segmentation_base(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build a member-level dataframe used for segmentation rules.

    Parameters
    ----------
    runtime:
        Population Health runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Joined member-level segmentation dataframe.

    Raises
    ------
    ValidationError
        Raised when required segmentation datasets or keys are missing.

    Notes
    -----
    The segmentation base uses one member-level base dataset and left-joins
    additional configured member-level feature datasets.
    """

    config = runtime.config.get("member_segmentation", {})
    member_key = config.get(
        "member_key",
        runtime.config.get("join_keys", {}).get("member_key", "member_id"),
    )
    base_dataset_name = config.get("base_dataset", "demographic_features")

    if base_dataset_name not in datasets:
        raise ValidationError(
            f"Segmentation base dataset missing: {base_dataset_name}"
        )

    base_df = datasets[base_dataset_name]

    require_columns(
        runtime=runtime,
        dataframe=base_df,
        dataset_name=base_dataset_name,
        required_columns=[member_key],
        source_layer="Layer 1F - Feature Store",
        source_dataset=base_dataset_name,
    )

    joined_df = base_df[[member_key]].drop_duplicates().copy()

    for dataset_name in config.get("required_datasets", []):
        if dataset_name not in datasets:
            raise ValidationError(
                f"Required segmentation dataset missing: {dataset_name}"
            )

        source_df = datasets[dataset_name]

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=dataset_name,
            required_columns=[member_key],
            source_layer="Layer 1F - Feature Store",
            source_dataset=dataset_name,
        )

        source_deduped = source_df.drop_duplicates(subset=[member_key]).copy()

        overlapping_columns = [
            column
            for column in source_deduped.columns
            if column != member_key and column in joined_df.columns
        ]

        rename_map = {
            column: f"{dataset_name}__{column}"
            for column in overlapping_columns
        }

        source_deduped = source_deduped.rename(columns=rename_map)
        joined_df = joined_df.merge(source_deduped, on=member_key, how="left")

    return joined_df


def find_condition_column(
    dataframe: pd.DataFrame,
    dataset_name: str,
    column: str,
) -> str:
    """
    Purpose
    -------
    Locate a condition column in the prepared segmentation dataframe.

    Parameters
    ----------
    dataframe:
        Segmentation dataframe.

    dataset_name:
        Source dataset name from rule configuration.

    column:
        Configured condition column.

    Returns
    -------
    str
        Resolved dataframe column name.

    Raises
    ------
    ValidationError
        Raised when the condition column cannot be found.
    """

    if column in dataframe.columns:
        return column

    prefixed_column = f"{dataset_name}__{column}"

    if prefixed_column in dataframe.columns:
        return prefixed_column

    fallback_column = resolve_existing_column(
        dataframe=dataframe,
        preferred_column=column,
        fallback_columns=[
            "latest_sdoh_risk_score",
            "total_paid_amount",
            "total_claims",
            "total_lab_results",
            "total_pharmacy_claims",
            "high_cost_risk_signal",
            "sdoh_risk_signal",
            "clinical_risk_signal",
        ],
    )

    if fallback_column:
        return fallback_column

    raise ValidationError(
        "Segmentation condition column not found. "
        f"Dataset: {dataset_name}, Column: {column}"
    )


def build_member_segmentation(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build member segmentation output.

    Parameters
    ----------
    runtime:
        Population Health runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Member segmentation output.

    Raises
    ------
    ValidationError
        Raised when segmentation rules reference invalid datasets or columns.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("member_segmentation", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Member segmentation disabled")
        return pd.DataFrame()

    member_key = config.get(
        "member_key",
        runtime.config.get("join_keys", {}).get("member_key", "member_id"),
    )

    segmentation_df = prepare_segmentation_base(runtime, datasets)

    segmentation_df["member_segment"] = "Unassigned"
    segmentation_df["segment_priority_rank"] = None

    logger.info("START: Build member segmentation")

    ordered_segments = sorted(
        config.get("segments", {}).items(),
        key=lambda item: item[1].get("priority_rank", 999),
    )

    for segment_name, segment_config in ordered_segments:
        description = segment_config.get("description", "")
        priority_rank = segment_config.get("priority_rank")

        add_rule_record(
            runtime=runtime,
            rule_group="member_segmentation",
            rule_name=segment_name,
            rule_type="multi_condition",
            description=description,
            source_dataset=config.get("base_dataset", "demographic_features"),
            source_layer="Layer 1F - Feature Store",
            rule_config=segment_config,
        )

        segment_mask = pd.Series(True, index=segmentation_df.index)

        for condition in segment_config.get("conditions", []):
            dataset_name = condition.get("dataset")
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            actual_column = find_condition_column(
                dataframe=segmentation_df,
                dataset_name=dataset_name,
                column=column,
            )

            condition_mask = apply_operator(
                series=segmentation_df[actual_column],
                operator=operator,
                value=value,
            )

            segment_mask = segment_mask & condition_mask

        unassigned_mask = segmentation_df["member_segment"] == "Unassigned"
        final_mask = segment_mask & unassigned_mask

        segmentation_df.loc[final_mask, "member_segment"] = segment_name
        segmentation_df.loc[final_mask, "segment_priority_rank"] = priority_rank

        logger.info(
            "Applied segment: %s | Members assigned: %s",
            segment_name,
            int(final_mask.sum()),
        )

    output_df = segmentation_df[
        [
            member_key,
            "member_segment",
            "segment_priority_rank",
        ]
    ].copy()

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "member_segmentation"
    output_df["source_layer"] = "Layer 1F - Feature Store"
    output_df["source_dataset"] = config.get("base_dataset", "demographic_features")
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="member_segmentation",
        dataset_type="population_health_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Member segmentation built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset=config.get("base_dataset", "demographic_features"),
    )

    logger.info("COMPLETE: Build member segmentation | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Attribution Analytics
###############################################################################

def calculate_provider_metric(
    dataframe: pd.DataFrame,
    provider_key: str,
    metric_name: str,
    metric_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Calculate one provider attribution metric.

    Parameters
    ----------
    dataframe:
        Provider attribution source dataframe.

    provider_key:
        Provider key column.

    metric_name:
        Output metric name.

    metric_config:
        Metric configuration from YAML.

    Returns
    -------
    pandas.DataFrame
        Provider-level metric dataframe.

    Raises
    ------
    ValidationError
        Raised when a metric references a missing required column.
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

    if calculation_type == "count_rows":
        return (
            dataframe.groupby(provider_key)
            .size()
            .reset_index(name=metric_name)
        )

    if calculation_type == "count_distinct":
        if not column or column not in dataframe.columns:
            return (
                dataframe.groupby(provider_key)
                .size()
                .reset_index(name=metric_name)
            )

        return (
            dataframe.groupby(provider_key)[column]
            .nunique(dropna=True)
            .reset_index(name=metric_name)
        )

    if calculation_type == "sum":
        if not column or column not in dataframe.columns:
            raise ValidationError(
                f"Metric {metric_name} requires available column: {column}"
            )

        return (
            dataframe.groupby(provider_key)[column]
            .sum()
            .reset_index(name=metric_name)
        )

    if calculation_type == "mean":
        if not column or column not in dataframe.columns:
            raise ValidationError(
                f"Metric {metric_name} requires available column: {column}"
            )

        return (
            dataframe.groupby(provider_key)[column]
            .mean()
            .reset_index(name=metric_name)
        )

    raise ValidationError(
        f"Unsupported provider metric calculation_type: {calculation_type}"
    )


def build_provider_attribution_analytics(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider attribution analytics.

    Parameters
    ----------
    runtime:
        Population Health runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Provider-level attribution analytics output.

    Raises
    ------
    ValidationError
        Raised when the provider source dataset or provider key is missing.

    Notes
    -----
    Current MedFabric provider_attribution_features is provider-grain, not
    member-provider-grain. Therefore member_id is not required for this output.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_attribution_analytics", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider attribution analytics disabled")
        return pd.DataFrame()

    source_dataset_name = config.get("source_dataset", "provider_attribution_features")
    provider_key = config.get("provider_key", "provider_id")

    if source_dataset_name not in datasets:
        raise ValidationError(
            f"Provider attribution source dataset missing: {source_dataset_name}"
        )

    source_df = datasets[source_dataset_name]

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=[provider_key],
        source_layer="Layer 1F - Feature Store",
        source_dataset=source_dataset_name,
    )

    analytics_df = source_df[[provider_key]].drop_duplicates().copy()

    for metric_name, metric_config in config.get("metrics", {}).items():
        add_rule_record(
            runtime=runtime,
            rule_group="provider_attribution_analytics",
            rule_name=metric_name,
            rule_type=metric_config.get("calculation_type", ""),
            description=metric_config.get("description", ""),
            source_dataset=source_dataset_name,
            source_layer="Layer 1F - Feature Store",
            rule_config=metric_config,
        )

        metric_df = calculate_provider_metric(
            dataframe=source_df,
            provider_key=provider_key,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        analytics_df = analytics_df.merge(metric_df, on=provider_key, how="left")

    analytics_df["analytics_layer_run_id"] = runtime.context.run_id
    analytics_df["analytics_domain"] = runtime.domain_name
    analytics_df["analytics_asset_name"] = "provider_attribution_analytics"
    analytics_df["source_layer"] = "Layer 1F - Feature Store"
    analytics_df["source_dataset"] = source_dataset_name
    analytics_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_attribution_analytics",
        dataset_type="population_health_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(analytics_df),
        column_count=len(analytics_df.columns),
        message="Provider attribution analytics built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset=source_dataset_name,
    )

    logger.info(
        "COMPLETE: Build provider attribution analytics | Rows: %s",
        len(analytics_df),
    )

    return analytics_df


###############################################################################
# Main Orchestration
###############################################################################

def build_population_health_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Population Health Analytics layer.

    Parameters
    ----------
    config_path:
        Population Health configuration path.

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
    This function is intentionally the only public build entry point for the
    Population Health domain.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Population Health Analytics started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "member_360": "Layer 1E - Gold",
                "demographic_features": "Layer 1F - Feature Store",
                "enrollment_features": "Layer 1F - Feature Store",
                "cost_features": "Layer 1F - Feature Store",
                "utilization_features": "Layer 1F - Feature Store",
                "risk_features": "Layer 1F - Feature Store",
                "provider_attribution_features": "Layer 1F - Feature Store",
                "sdoh_features": "Layer 1F - Feature Store",
                "member_360_semantic_view": "Layer 1G - Semantic Layer",
                "provider_attribution_semantic_view": "Layer 1G - Semantic Layer",
            },
            key_column_map={
                "member_360": "member_id",
                "demographic_features": "member_id",
                "enrollment_features": "member_id",
                "cost_features": "member_id",
                "utilization_features": "member_id",
                "risk_features": "member_id",
                "sdoh_features": "member_id",
                "member_360_semantic_view": "member_id",
                "provider_attribution_features": "provider_id",
                "provider_attribution_semantic_view": "provider_id",
            },
        )

        population_cohorts = build_population_cohorts(runtime, datasets)
        risk_stratification = build_risk_stratification(runtime, datasets)
        member_segmentation = build_member_segmentation(runtime, datasets)
        provider_attribution_analytics = build_provider_attribution_analytics(
            runtime,
            datasets,
        )

        output_assets = {
            "population_cohorts": population_cohorts,
            "risk_stratification": risk_stratification,
            "member_segmentation": member_segmentation,
            "provider_attribution_analytics": provider_attribution_analytics,
        }

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_population_health_layer",
            message="Population Health Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2A - Population Health Analytics",
            source_dataset="population_health_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="population_health_dataset_inventory",
            column_dictionary_name="population_health_column_dictionary",
            rule_catalog_name="population_health_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="population_health_audit_records",
            validation_results_name="population_health_validation_results",
            execution_summary_name="population_health_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Population Health Analytics completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name="population_health",
            status=STATUS_SUCCESS,
            message="Population Health Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(len(dataframe.columns) for dataframe in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Population Health Analytics failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_population_health_layer",
                message=str(exc),
                source_layer="Layer 2A - Population Health Analytics",
                source_dataset="population_health",
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
                    audit_records_name="population_health_audit_records",
                    validation_results_name="population_health_validation_results",
                    execution_summary_name="population_health_execution_summary",
                )
            except Exception as audit_exc:
                logger.error("Failed to write audit outputs: %s", audit_exc)

        return AnalyticsBuildResult(
            name="population_health",
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
    Command-line entry point for Population Health Analytics.

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
        "MEDFABRIC_POPULATION_HEALTH_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_population_health_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Population Health Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()