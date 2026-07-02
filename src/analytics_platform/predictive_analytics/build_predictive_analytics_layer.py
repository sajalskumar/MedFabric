###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/predictive_analytics/build_predictive_analytics_layer.py
#
# Layer:
#     Layer 2D - Predictive Analytics
#
# Purpose:
#     Builds Predictive Analytics outputs from Model Training & Scoring outputs.
#
# Architectural Rule:
#     Predictive Analytics consumes scored model outputs.
#
#     This file does NOT train models.
#     This file does NOT create model pickle artifacts.
#     This file does NOT read Feature Store datasets directly.
#
# Dependency Flow:
#     Feature Store
#         ↓
#     Modeling Layer - Model Training & Scoring
#         ↓
#     data/scoring/
#         ↓
#     Layer 2D - Predictive Analytics
#
# Architecture:
#     This file contains Predictive Analytics business logic only.
#
#     Shared Analytics Platform concerns are handled by:
#         - src.analytics_platform.common.runtime
#         - src.analytics_platform.common.io
#         - src.analytics_platform.common.audit
#         - src.analytics_platform.common.validation
#         - src.analytics_platform.common.metadata
#
# Inputs:
#     config/analytics_platform/predictive_analytics.yaml
#     data/scoring/*.parquet
#     data/metadata/modeling_model_registry.parquet
#
# Outputs:
#     data/analytics_platform/predictive_analytics/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.predictive_analytics.build_predictive_analytics_layer
#
###############################################################################

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Optional

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

DEFAULT_CONFIG_PATH = "config/analytics_platform/predictive_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2D - Predictive Analytics"
DEFAULT_DOMAIN_NAME = "Predictive Analytics"
DOMAIN_SECTION = "predictive_analytics"

LOGGER_NAME = "medfabric.analytics_platform.predictive_analytics"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict) -> None:
    """
    Purpose
    -------
    Validate required Predictive Analytics configuration sections.

    Parameters
    ----------
    config:
        Loaded Predictive Analytics YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required sections are missing.

    Notes
    -----
    Generic YAML loading is handled by ConfigurationManager through
    PipelineContext. This function only validates the Predictive Analytics
    domain contract.
    """

    required_sections = [
        "predictive_analytics",
        "paths",
        "join_keys",
        "model_score_registry",
        "unified_prediction_registry",
        "member_prediction_summary",
        "high_priority_member_registry",
        "model_risk_distribution",
        "prediction_model_summary",
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
            "Predictive Analytics configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Predictive Analytics runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Predictive Analytics configuration path.

    Returns
    -------
    AnalyticsDomainRuntime
        Initialized domain runtime.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2D - Predictive Analytics"
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
        message="Predictive Analytics runtime initialized successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Input Validation
###############################################################################

def validate_model_score_inputs(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Purpose
    -------
    Validate configured model scoring input datasets.

    Parameters
    ----------
    runtime:
        Predictive Analytics runtime.

    datasets:
        Loaded scoring and model metadata datasets.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        Raised when required scoring datasets or scoring columns are missing.

    Notes
    -----
    Predictive Analytics depends on scoring outputs already produced by the
    Modeling Layer. This function enforces the boundary that scoring assets must
    already exist and must contain the configured score, prediction, and tier
    columns.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    for model_key, model_config in runtime.config.get("model_score_registry", {}).items():
        if not bool(model_config.get("enabled", True)):
            continue

        source_dataset = model_config.get("source_dataset")

        if source_dataset not in datasets:
            raise ValidationError(
                f"Model scoring dataset not loaded for model '{model_key}': "
                f"{source_dataset}"
            )

        score_df = datasets[source_dataset]

        required_columns = [
            member_key,
            model_config.get("score_column"),
            model_config.get("prediction_column"),
            model_config.get("tier_column"),
            "model_key",
            "model_name",
            "modeling_layer_run_id",
            "scored_at_utc",
        ]

        require_columns(
            runtime=runtime,
            dataframe=score_df,
            dataset_name=source_dataset,
            required_columns=required_columns,
            source_layer="Modeling Layer - Scoring",
            source_dataset=source_dataset,
        )

    add_success_audit(
        runtime=runtime,
        step_name="validate_model_score_inputs",
        message="Predictive Analytics scoring inputs validated successfully.",
        source_layer="Modeling Layer - Scoring",
        source_dataset="data/scoring",
    )


###############################################################################
# Unified Prediction Registry
###############################################################################

def build_unified_prediction_registry(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build unified long-format prediction registry.

    Parameters
    ----------
    runtime:
        Predictive Analytics runtime.

    datasets:
        Loaded scoring datasets.

    Returns
    -------
    pandas.DataFrame
        Unified prediction registry.

    Raises
    ------
    ValidationError
        Raised when an enabled scoring source is missing required columns.

    Notes
    -----
    Output grain is one row per member per model.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("unified_prediction_registry", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Unified prediction registry disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    registry_frames: List[pd.DataFrame] = []

    logger.info("START: Build unified prediction registry")

    for model_key, model_config in runtime.config.get("model_score_registry", {}).items():
        if not bool(model_config.get("enabled", True)):
            logger.info("SKIP model score disabled: %s", model_key)
            continue

        source_dataset = model_config.get("source_dataset")
        score_column = model_config.get("score_column")
        prediction_column = model_config.get("prediction_column")
        tier_column = model_config.get("tier_column")
        model_name = model_config.get("model_name", model_key)

        add_rule_record(
            runtime=runtime,
            rule_group="model_score_registry",
            rule_name=model_key,
            rule_type="scoring_output_standardization",
            description=f"Standardize scoring output for {model_name}.",
            source_dataset=source_dataset,
            source_layer="Modeling Layer - Scoring",
            rule_config=model_config,
        )

        if source_dataset not in datasets:
            raise ValidationError(f"Scoring source dataset missing: {source_dataset}")

        source_df = datasets[source_dataset]

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[
                member_key,
                score_column,
                prediction_column,
                tier_column,
                "modeling_layer_run_id",
                "scored_at_utc",
            ],
            source_layer="Modeling Layer - Scoring",
            source_dataset=source_dataset,
        )

        standardized_df = source_df[
            [
                member_key,
                score_column,
                prediction_column,
                tier_column,
                "modeling_layer_run_id",
                "scored_at_utc",
            ]
        ].copy()

        standardized_df = standardized_df.rename(
            columns={
                score_column: "prediction_score",
                prediction_column: "prediction_flag",
                tier_column: "prediction_tier",
            }
        )

        standardized_df["model_key"] = model_key
        standardized_df["model_name"] = model_name
        standardized_df["analytics_layer_run_id"] = runtime.context.run_id
        standardized_df["analytics_domain"] = runtime.domain_name
        standardized_df["analytics_asset_name"] = "unified_prediction_registry"
        standardized_df["source_layer"] = "Modeling Layer - Scoring"
        standardized_df["source_dataset"] = source_dataset
        standardized_df["built_at_utc"] = utc_now().isoformat()

        registry_frames.append(standardized_df)

        logger.info(
            "Standardized scoring output: %s | Rows: %s",
            model_key,
            len(standardized_df),
        )

    if registry_frames:
        output_df = pd.concat(registry_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "prediction_score",
                "prediction_flag",
                "prediction_tier",
                "modeling_layer_run_id",
                "scored_at_utc",
                "model_key",
                "model_name",
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
        dataset_name="unified_prediction_registry",
        dataset_type="predictive_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Unified prediction registry built successfully.",
        source_layer="Modeling Layer - Scoring",
        source_dataset="data/scoring",
    )

    logger.info("COMPLETE: Build unified prediction registry | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Member Prediction Summary
###############################################################################

def build_member_prediction_summary(
    runtime: AnalyticsDomainRuntime,
    unified_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build member-level prediction summary.

    Parameters
    ----------
    runtime:
        Predictive Analytics runtime.

    unified_registry:
        Unified long-format prediction registry.

    Returns
    -------
    pandas.DataFrame
        Member-level prediction summary.

    Raises
    ------
    None

    Notes
    -----
    Output grain is one row per member.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("member_prediction_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Member prediction summary disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    if unified_registry.empty:
        return pd.DataFrame()

    logger.info("START: Build member prediction summary")

    grouped = unified_registry.groupby(member_key)

    output_df = grouped.agg(
        max_prediction_score=("prediction_score", "max"),
        average_prediction_score=("prediction_score", "mean"),
        positive_prediction_count=("prediction_flag", "sum"),
        model_count=("model_key", "nunique"),
    ).reset_index()

    top_model_df = (
        unified_registry.sort_values(["prediction_score"], ascending=False)
        .drop_duplicates(subset=[member_key])
        [[member_key, "model_key", "model_name", "prediction_score", "prediction_tier"]]
        .rename(
            columns={
                "model_key": "top_risk_model_key",
                "model_name": "top_risk_model_name",
                "prediction_score": "top_risk_score",
                "prediction_tier": "top_risk_tier",
            }
        )
    )

    output_df = output_df.merge(top_model_df, on=member_key, how="left")

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "member_prediction_summary"
    output_df["source_layer"] = "Layer 2D - Predictive Analytics"
    output_df["source_dataset"] = "unified_prediction_registry"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="member_prediction_summary",
        rule_name="member_level_prediction_summary",
        rule_type="aggregation",
        description="Aggregates model scores to one row per member.",
        source_dataset="unified_prediction_registry",
        source_layer="Layer 2D - Predictive Analytics",
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="member_prediction_summary",
        dataset_type="predictive_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Member prediction summary built successfully.",
        source_layer="Layer 2D - Predictive Analytics",
        source_dataset="unified_prediction_registry",
    )

    logger.info("COMPLETE: Build member prediction summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# High Priority Member Registry
###############################################################################

def build_high_priority_member_registry(
    runtime: AnalyticsDomainRuntime,
    member_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build high-priority member registry using configured priority rules.

    Parameters
    ----------
    runtime:
        Predictive Analytics runtime.

    member_summary:
        Member-level prediction summary.

    Returns
    -------
    pandas.DataFrame
        High-priority member registry.

    Raises
    ------
    None

    Notes
    -----
    Output grain is one row per priority-qualified member.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("high_priority_member_registry", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: High priority member registry disabled")
        return pd.DataFrame()

    if member_summary.empty:
        return pd.DataFrame()

    output_df = member_summary.copy()
    output_df["priority_label"] = "Unassigned"
    output_df["priority_rank"] = None

    logger.info("START: Build high priority member registry")

    rules = sorted(
        config.get("priority_rules", {}).items(),
        key=lambda item: item[1].get("priority_rank", 999),
    )

    for rule_key, rule_config in rules:
        label = rule_config.get("label", rule_key)
        min_score = rule_config.get("min_score", 0.0)
        min_positive_predictions = rule_config.get("min_positive_predictions", 0)
        priority_rank = rule_config.get("priority_rank")

        add_rule_record(
            runtime=runtime,
            rule_group="high_priority_member_registry",
            rule_name=rule_key,
            rule_type="priority_assignment",
            description=f"Assigns {label} priority.",
            source_dataset="member_prediction_summary",
            source_layer="Layer 2D - Predictive Analytics",
            rule_config=rule_config,
        )

        mask = (
            (output_df["max_prediction_score"] >= min_score)
            & (output_df["positive_prediction_count"] >= min_positive_predictions)
            & (output_df["priority_label"] == "Unassigned")
        )

        output_df.loc[mask, "priority_label"] = label
        output_df.loc[mask, "priority_rank"] = priority_rank

        logger.info(
            "Applied priority rule: %s | Members assigned: %s",
            rule_key,
            int(mask.sum()),
        )

    output_df = output_df[output_df["priority_label"] != "Unassigned"].copy()

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "high_priority_member_registry"
    output_df["source_layer"] = "Layer 2D - Predictive Analytics"
    output_df["source_dataset"] = "member_prediction_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="high_priority_member_registry",
        dataset_type="predictive_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="High priority member registry built successfully.",
        source_layer="Layer 2D - Predictive Analytics",
        source_dataset="member_prediction_summary",
    )

    logger.info("COMPLETE: Build high priority member registry | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Model Risk Distribution
###############################################################################

def build_model_risk_distribution(
    runtime: AnalyticsDomainRuntime,
    unified_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build model-level risk distribution.

    Parameters
    ----------
    runtime:
        Predictive Analytics runtime.

    unified_registry:
        Unified long-format prediction registry.

    Returns
    -------
    pandas.DataFrame
        Model risk distribution.

    Raises
    ------
    None

    Notes
    -----
    Output grain is one row per configured grouping, typically model and tier.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("model_risk_distribution", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Model risk distribution disabled")
        return pd.DataFrame()

    if unified_registry.empty:
        return pd.DataFrame()

    group_by = config.get("group_by", ["model_key", "model_name", "prediction_tier"])

    logger.info("START: Build model risk distribution")

    output_df = unified_registry.groupby(group_by).agg(
        member_count=("member_id", "nunique"),
        average_prediction_score=("prediction_score", "mean"),
        positive_prediction_count=("prediction_flag", "sum"),
    ).reset_index()

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "model_risk_distribution"
    output_df["source_layer"] = "Layer 2D - Predictive Analytics"
    output_df["source_dataset"] = "unified_prediction_registry"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="model_risk_distribution",
        rule_name="model_tier_distribution",
        rule_type="aggregation",
        description="Aggregates prediction registry by model and tier.",
        source_dataset="unified_prediction_registry",
        source_layer="Layer 2D - Predictive Analytics",
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="model_risk_distribution",
        dataset_type="predictive_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Model risk distribution built successfully.",
        source_layer="Layer 2D - Predictive Analytics",
        source_dataset="unified_prediction_registry",
    )

    logger.info("COMPLETE: Build model risk distribution | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Prediction Model Summary
###############################################################################

def build_prediction_model_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
    unified_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build prediction model summary from model registry and scoring statistics.

    Parameters
    ----------
    runtime:
        Predictive Analytics runtime.

    datasets:
        Loaded input datasets, including optional model registry.

    unified_registry:
        Unified prediction registry.

    Returns
    -------
    pandas.DataFrame
        Prediction model summary.

    Raises
    ------
    None

    Notes
    -----
    The model registry is optional. If it is unavailable, scoring statistics are
    still produced from the unified prediction registry.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("prediction_model_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Prediction model summary disabled")
        return pd.DataFrame()

    model_registry_name = config.get("source_dataset", "modeling_model_registry")

    if model_registry_name in datasets:
        model_registry_df = datasets[model_registry_name].copy()
    else:
        model_registry_df = pd.DataFrame()

    if unified_registry.empty:
        score_stats_df = pd.DataFrame()
    else:
        score_stats_df = unified_registry.groupby(["model_key", "model_name"]).agg(
            scored_member_count=("member_id", "nunique"),
            average_prediction_score=("prediction_score", "mean"),
            max_prediction_score=("prediction_score", "max"),
            positive_prediction_count=("prediction_flag", "sum"),
        ).reset_index()

    if not model_registry_df.empty and "model_key" in model_registry_df.columns:
        output_df = model_registry_df.merge(score_stats_df, on="model_key", how="left")
    else:
        output_df = score_stats_df.copy()

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "prediction_model_summary"
    output_df["source_layer"] = "Modeling Layer + Layer 2D - Predictive Analytics"
    output_df["source_dataset"] = model_registry_name
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="prediction_model_summary",
        rule_name="model_registry_with_score_stats",
        rule_type="metadata_enrichment",
        description="Combines model registry metadata with scoring distribution.",
        source_dataset=model_registry_name,
        source_layer="Modeling Layer",
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="prediction_model_summary",
        dataset_type="predictive_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Prediction model summary built successfully.",
        source_layer="Modeling Layer + Layer 2D - Predictive Analytics",
        source_dataset=model_registry_name,
    )

    logger.info("COMPLETE: Build prediction model summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_predictive_analytics_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Predictive Analytics layer.

    Parameters
    ----------
    config_path:
        Predictive Analytics configuration path.

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
    This layer consumes scored model outputs only. It does not train models,
    score members, or create model artifacts.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Predictive Analytics started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)
        logger.info("Architectural check: Predictive Analytics consumes scoring outputs only.")

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "high_cost_scores": "Modeling Layer - Scoring",
                "readmission_scores": "Modeling Layer - Scoring",
                "er_utilization_scores": "Modeling Layer - Scoring",
                "rising_risk_scores": "Modeling Layer - Scoring",
                "chronic_progression_scores": "Modeling Layer - Scoring",
                "avoidable_admission_scores": "Modeling Layer - Scoring",
                "medication_non_adherence_scores": "Modeling Layer - Scoring",
                "care_gap_closure_scores": "Modeling Layer - Scoring",
                "modeling_model_registry": "Modeling Layer - Metadata",
            },
            key_column_map={
                "high_cost_scores": "member_id",
                "readmission_scores": "member_id",
                "er_utilization_scores": "member_id",
                "rising_risk_scores": "member_id",
                "chronic_progression_scores": "member_id",
                "avoidable_admission_scores": "member_id",
                "medication_non_adherence_scores": "member_id",
                "care_gap_closure_scores": "member_id",
                "modeling_model_registry": "model_key",
            },
        )

        validate_model_score_inputs(runtime, datasets)

        unified_prediction_registry = build_unified_prediction_registry(
            runtime=runtime,
            datasets=datasets,
        )

        member_prediction_summary = build_member_prediction_summary(
            runtime=runtime,
            unified_registry=unified_prediction_registry,
        )

        high_priority_member_registry = build_high_priority_member_registry(
            runtime=runtime,
            member_summary=member_prediction_summary,
        )

        model_risk_distribution = build_model_risk_distribution(
            runtime=runtime,
            unified_registry=unified_prediction_registry,
        )

        prediction_model_summary = build_prediction_model_summary(
            runtime=runtime,
            datasets=datasets,
            unified_registry=unified_prediction_registry,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "unified_prediction_registry": unified_prediction_registry,
            "member_prediction_summary": member_prediction_summary,
            "high_priority_member_registry": high_priority_member_registry,
            "model_risk_distribution": model_risk_distribution,
            "prediction_model_summary": prediction_model_summary,
        }

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_predictive_analytics_layer",
            message="Predictive Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2D - Predictive Analytics",
            source_dataset="predictive_analytics_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="predictive_analytics_dataset_inventory",
            column_dictionary_name="predictive_analytics_column_dictionary",
            rule_catalog_name="predictive_analytics_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="predictive_analytics_audit_records",
            validation_results_name="predictive_analytics_validation_results",
            execution_summary_name="predictive_analytics_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Predictive Analytics completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name="predictive_analytics",
            status=STATUS_SUCCESS,
            message="Predictive Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(len(dataframe.columns) for dataframe in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Predictive Analytics failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_predictive_analytics_layer",
                message=str(exc),
                source_layer="Layer 2D - Predictive Analytics",
                source_dataset="predictive_analytics",
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
                    audit_records_name="predictive_analytics_audit_records",
                    validation_results_name="predictive_analytics_validation_results",
                    execution_summary_name="predictive_analytics_execution_summary",
                )
            except Exception as audit_exc:
                logger.error("Failed to write audit outputs: %s", audit_exc)

        return AnalyticsBuildResult(
            name="predictive_analytics",
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
    Command-line entry point for Predictive Analytics.

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
        "MEDFABRIC_PREDICTIVE_ANALYTICS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_predictive_analytics_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Predictive Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()