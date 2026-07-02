###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/care_management/build_care_management_layer.py
#
# Layer:
#     Layer 2G - Care Management
#
# Purpose:
#     Builds Care Management analytics outputs from Population Health,
#     Clinical Analytics, Quality Analytics, Predictive Analytics, and Provider
#     Analytics outputs.
#
# Architectural Rule:
#     Care Management consumes already-built upstream analytics outputs.
#
#     This file does NOT generate raw data.
#     This file does NOT build Silver dimensional models.
#     This file does NOT train predictive models.
#     This file does NOT score members.
#
# Dependency Flow:
#     Population Health + Clinical Analytics + Quality Analytics
#         ↓
#     Predictive Analytics + Provider Analytics
#         ↓
#     Layer 2G - Care Management
#         ↓
#     data/analytics_platform/care_management/
#
# Architecture:
#     This file contains Care Management business logic only.
#
#     Shared Analytics Platform concerns are handled by:
#         - src.analytics_platform.common.runtime
#         - src.analytics_platform.common.io
#         - src.analytics_platform.common.audit
#         - src.analytics_platform.common.validation
#         - src.analytics_platform.common.metadata
#
# Inputs:
#     config/analytics_platform/care_management.yaml
#
# Outputs:
#     data/analytics_platform/care_management/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.care_management.build_care_management_layer
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
from src.analytics_platform.common.validation import require_columns
from src.common.exception_manager import PipelineError, ValidationError
from src.common.pipeline_context import create_pipeline_context


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/analytics_platform/care_management.yaml"

DEFAULT_LAYER_NAME = "Layer 2G - Care Management"
DEFAULT_DOMAIN_NAME = "Care Management"
DOMAIN_SECTION = "care_management"

LOGGER_NAME = "medfabric.analytics_platform.care_management"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Care Management configuration sections.

    Parameters
    ----------
    config:
        Loaded Care Management YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required configuration sections are missing.

    Notes
    -----
    Generic YAML loading is handled by ConfigurationManager through
    PipelineContext. This function validates only the Care Management domain
    contract.
    """

    required_sections = [
        "care_management",
        "paths",
        "join_keys",
        "care_management_framework",
        "care_programs",
        "case_management_worklist",
        "transitions_of_care",
        "disease_management",
        "outreach_tracking",
        "program_effectiveness",
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
            "Care Management configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Care Management runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Care Management configuration path.

    Returns
    -------
    AnalyticsDomainRuntime
        Initialized Care Management runtime.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This follows the same shared framework pattern as Predictive Analytics,
    Provider Analytics, and Value-Based Care.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2G - Care Management"
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
        message="Care Management runtime initialized successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Shared Business Helpers
###############################################################################

def select_available_columns(dataframe: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Purpose
    -------
    Return configured columns that exist in a dataframe.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    columns:
        Candidate column list.

    Returns
    -------
    list[str]
        Columns from the candidate list that exist in the dataframe.

    Raises
    ------
    None
    """

    return [column for column in columns if column in dataframe.columns]


def deduplicate_member_dataset(
    dataframe: pd.DataFrame,
    member_key: str,
    sort_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Deduplicate a member-level dataset.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    member_key:
        Member key column.

    sort_columns:
        Optional columns used before deduplication to keep the highest-priority
        member row.

    Returns
    -------
    pandas.DataFrame
        Deduplicated dataframe.

    Raises
    ------
    None

    Notes
    -----
    If the member key is missing, the dataframe is returned as-is because some
    upstream summary assets may not be member-grain.
    """

    if member_key not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()

    if sort_columns:
        existing_sort_columns = [
            column for column in sort_columns if column in result.columns
        ]

        if existing_sort_columns:
            result = result.sort_values(existing_sort_columns, ascending=True)

    return result.drop_duplicates(subset=[member_key]).copy()


def merge_member_enrichments(
    base_df: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    enrichment_dataset_names: List[str],
    member_key: str,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Merge member-level enrichment datasets onto a base dataframe.

    Parameters
    ----------
    base_df:
        Base member-level dataframe.

    datasets:
        Loaded input datasets.

    enrichment_dataset_names:
        Dataset names to merge.

    member_key:
        Member key column.

    Returns
    -------
    pandas.DataFrame
        Base dataframe enriched with available member-level datasets.

    Raises
    ------
    None

    Notes
    -----
    Duplicate column names are prefixed with the enrichment dataset name so that
    information is preserved instead of overwritten.
    """

    output_df = base_df.copy()

    for dataset_name in enrichment_dataset_names:
        if dataset_name not in datasets:
            continue

        enrichment_df = datasets[dataset_name]

        if member_key not in enrichment_df.columns:
            continue

        enrichment_deduped = deduplicate_member_dataset(
            dataframe=enrichment_df,
            member_key=member_key,
            sort_columns=[
                "program_priority_rank",
                "priority_rank",
                "segment_priority_rank",
                "risk_priority_rank",
            ],
        )

        columns_to_keep = [
            column for column in enrichment_deduped.columns if column != member_key
        ]

        rename_map: Dict[str, str] = {}

        for column in columns_to_keep:
            if column in output_df.columns:
                rename_map[column] = f"{dataset_name}__{column}"

        enrichment_deduped = enrichment_deduped.rename(columns=rename_map)

        output_df = output_df.merge(enrichment_deduped, on=member_key, how="left")

    return output_df


def calculate_group_metric(
    dataframe: pd.DataFrame,
    group_by: List[str],
    metric_name: str,
    metric_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Calculate a configured grouped metric.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    group_by:
        Grouping columns.

    metric_name:
        Output metric column name.

    metric_config:
        Metric configuration.

    Returns
    -------
    pandas.DataFrame
        Grouped metric dataframe.

    Raises
    ------
    ValidationError
        Raised when metric configuration is invalid.

    Notes
    -----
    Supported calculation types:
        - count_distinct
        - mean
        - sum
        - count_rows
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

    if calculation_type == "count_rows":
        return dataframe.groupby(group_by).size().reset_index(name=metric_name)

    if not column:
        raise ValidationError(
            f"Metric '{metric_name}' requires a configured column."
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

    if calculation_type == "mean":
        return (
            dataframe.groupby(group_by)[column]
            .mean()
            .reset_index(name=metric_name)
        )

    if calculation_type == "sum":
        return (
            dataframe.groupby(group_by)[column]
            .sum()
            .reset_index(name=metric_name)
        )

    raise ValidationError(f"Unsupported calculation_type: {calculation_type}")


###############################################################################
# Care Programs
###############################################################################

def build_care_programs(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build configured care program assignments.

    Parameters
    ----------
    runtime:
        Care Management runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Care program assignments.

    Raises
    ------
    ValidationError
        Raised when configured source datasets or rule columns are missing.

    Notes
    -----
    Output grain is one row per member per care program.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("care_programs", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Care programs disabled")
        return pd.DataFrame()

    framework_config = runtime.config.get("care_management_framework", {})
    member_key = framework_config.get("member_key", "member_id")
    enrichment_datasets = framework_config.get("enrichment_datasets", [])

    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build care programs")

    for rule_key, rule_config in config.get("program_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            logger.info("SKIP care program rule disabled: %s", rule_key)
            continue

        source_dataset = rule_config.get("source_dataset")
        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        care_program_name = rule_config.get("care_program_name", rule_key)
        care_program_category = rule_config.get("care_program_category", "")
        program_priority_rank = rule_config.get("program_priority_rank")
        description = rule_config.get("description", "")

        if source_dataset not in datasets:
            raise ValidationError(f"Care program source dataset missing: {source_dataset}")

        source_df = datasets[source_dataset]

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[member_key, rule_column],
            source_layer="Upstream Analytics",
            source_dataset=source_dataset,
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        program_df = source_df.loc[mask].copy()

        program_df = deduplicate_member_dataset(
            dataframe=program_df,
            member_key=member_key,
            sort_columns=[
                "priority_rank",
                "segment_priority_rank",
                "risk_priority_rank",
            ],
        )

        program_df = merge_member_enrichments(
            base_df=program_df,
            datasets=datasets,
            enrichment_dataset_names=enrichment_datasets,
            member_key=member_key,
        )

        program_df["care_program_key"] = rule_key
        program_df["care_program_name"] = care_program_name
        program_df["care_program_category"] = care_program_category
        program_df["program_priority_rank"] = program_priority_rank
        program_df["care_program_description"] = description
        program_df["analytics_layer_run_id"] = runtime.context.run_id
        program_df["analytics_domain"] = runtime.domain_name
        program_df["analytics_asset_name"] = "care_programs"
        program_df["source_layer"] = runtime.layer_name
        program_df["source_dataset"] = source_dataset
        program_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="care_programs",
            rule_name=rule_key,
            rule_type="care_program_assignment",
            description=description,
            source_dataset=source_dataset,
            source_layer="Upstream Analytics",
            rule_config=rule_config,
        )

        output_frames.append(program_df)

        logger.info(
            "Built care program: %s | Members: %s",
            care_program_name,
            len(program_df),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "care_program_key",
                "care_program_name",
                "care_program_category",
                "program_priority_rank",
                "care_program_description",
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
        dataset_name="care_programs",
        dataset_type="care_management_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Care programs built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="care_program_rules",
    )

    logger.info("COMPLETE: Build care programs | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Case Management Worklist
###############################################################################

def build_case_management_worklist(
    runtime: AnalyticsDomainRuntime,
    care_programs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build case management worklist.

    Parameters
    ----------
    runtime:
        Care Management runtime.

    care_programs:
        Care program assignment dataframe.

    Returns
    -------
    pandas.DataFrame
        Case management worklist.

    Raises
    ------
    None

    Notes
    -----
    Output grain is one row per member. The highest-priority program assignment
    becomes the case-management worklist driver.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("case_management_worklist", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Case management worklist disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    if care_programs.empty:
        return pd.DataFrame()

    priority_columns = select_available_columns(
        dataframe=care_programs,
        columns=config.get("priority_columns", []),
    )

    sort_columns = priority_columns if priority_columns else ["program_priority_rank"]

    output_df = (
        care_programs.sort_values(sort_columns, ascending=True)
        .drop_duplicates(subset=[member_key])
        .copy()
    )

    output_columns = select_available_columns(
        dataframe=output_df,
        columns=config.get("output_columns", []),
    )

    if output_columns:
        output_df = output_df[output_columns].copy()

    output_df["case_status"] = "Open"
    output_df["case_priority_rank"] = range(1, len(output_df) + 1)
    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "case_management_worklist"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = "care_programs"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="case_management_worklist",
        rule_name="member_case_prioritization",
        rule_type="dedupe_and_rank",
        description="Creates one prioritized care management row per member.",
        source_dataset="care_programs",
        source_layer=runtime.layer_name,
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="case_management_worklist",
        dataset_type="care_management_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Case management worklist built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="care_programs",
    )

    logger.info("COMPLETE: Build case management worklist | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Transitions of Care
###############################################################################

def build_transitions_of_care(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build transitions-of-care candidate registry.

    Parameters
    ----------
    runtime:
        Care Management runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Transitions-of-care candidate registry.

    Raises
    ------
    ValidationError
        Raised when configured source datasets or rule columns are missing.

    Notes
    -----
    Output grain is one row per member per transition rule.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("transitions_of_care", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Transitions of care disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    source_dataset = config.get("source_dataset", "high_priority_member_registry")

    if source_dataset not in datasets:
        raise ValidationError(f"Transitions of care source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build transitions of care")

    for rule_key, rule_config in config.get("transition_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            logger.info("SKIP transition rule disabled: %s", rule_key)
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[member_key, rule_column],
            source_layer="Layer 2D - Predictive Analytics",
            source_dataset=source_dataset,
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        transition_df = source_df.loc[mask].copy()
        transition_df = deduplicate_member_dataset(
            dataframe=transition_df,
            member_key=member_key,
            sort_columns=["priority_rank"],
        )

        transition_df["transition_key"] = rule_key
        transition_df["transition_name"] = rule_config.get("transition_name", rule_key)
        transition_df["transition_category"] = rule_config.get("transition_category", "")
        transition_df["transition_priority_rank"] = rule_config.get(
            "transition_priority_rank"
        )
        transition_df["transition_description"] = rule_config.get("description", "")
        transition_df["analytics_layer_run_id"] = runtime.context.run_id
        transition_df["analytics_domain"] = runtime.domain_name
        transition_df["analytics_asset_name"] = "transitions_of_care"
        transition_df["source_layer"] = "Layer 2D - Predictive Analytics"
        transition_df["source_dataset"] = source_dataset
        transition_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="transitions_of_care",
            rule_name=rule_key,
            rule_type="transition_candidate_selection",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
            source_layer="Layer 2D - Predictive Analytics",
            rule_config=rule_config,
        )

        output_frames.append(transition_df)

        logger.info(
            "Built transition rule: %s | Members: %s",
            rule_key,
            len(transition_df),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame()

    add_dataset_record(
        runtime=runtime,
        dataset_name="transitions_of_care",
        dataset_type="care_management_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Transitions of care built successfully.",
        source_layer="Layer 2D - Predictive Analytics",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build transitions of care | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Disease Management
###############################################################################

def build_disease_management(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build disease-management program candidates.

    Parameters
    ----------
    runtime:
        Care Management runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Disease-management program candidates.

    Raises
    ------
    ValidationError
        Raised when configured source datasets or rule columns are missing.

    Notes
    -----
    Output grain is one row per member per disease-management rule.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("disease_management", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Disease management disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    source_dataset = config.get("source_dataset", "condition_registry")

    if source_dataset not in datasets:
        raise ValidationError(f"Disease management source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build disease management")

    for rule_key, rule_config in config.get("disease_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            logger.info("SKIP disease management rule disabled: %s", rule_key)
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[member_key, rule_column],
            source_layer="Clinical Analytics",
            source_dataset=source_dataset,
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        disease_df = source_df.loc[mask].copy()
        disease_df = deduplicate_member_dataset(
            dataframe=disease_df,
            member_key=member_key,
        )

        disease_df["disease_rule_key"] = rule_key
        disease_df["disease_program_name"] = rule_config.get(
            "disease_program_name",
            rule_key,
        )
        disease_df["disease_category"] = rule_config.get("disease_category", "")
        disease_df["disease_priority_rank"] = rule_config.get("disease_priority_rank")
        disease_df["disease_program_description"] = rule_config.get("description", "")
        disease_df["analytics_layer_run_id"] = runtime.context.run_id
        disease_df["analytics_domain"] = runtime.domain_name
        disease_df["analytics_asset_name"] = "disease_management"
        disease_df["source_layer"] = "Clinical Analytics"
        disease_df["source_dataset"] = source_dataset
        disease_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="disease_management",
            rule_name=rule_key,
            rule_type="disease_program_assignment",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
            source_layer="Clinical Analytics",
            rule_config=rule_config,
        )

        output_frames.append(disease_df)

        logger.info(
            "Built disease management rule: %s | Members: %s",
            rule_key,
            len(disease_df),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame()

    add_dataset_record(
        runtime=runtime,
        dataset_name="disease_management",
        dataset_type="care_management_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Disease management built successfully.",
        source_layer="Clinical Analytics",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build disease management | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Outreach Tracking
###############################################################################

def build_outreach_tracking(
    runtime: AnalyticsDomainRuntime,
    case_management_worklist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build outreach tracking records.

    Parameters
    ----------
    runtime:
        Care Management runtime.

    case_management_worklist:
        Case management worklist dataframe.

    Returns
    -------
    pandas.DataFrame
        Outreach tracking records.

    Raises
    ------
    None

    Notes
    -----
    This creates initialized outreach tracking records that can later be
    updated by care-management workflow systems.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("outreach_tracking", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Outreach tracking disabled")
        return pd.DataFrame()

    if case_management_worklist.empty:
        return pd.DataFrame()

    output_df = case_management_worklist.copy()

    output_df["outreach_status"] = config.get("default_outreach_status", "Pending")
    output_df["outreach_channel"] = config.get(
        "default_outreach_channel",
        "Care Manager Review",
    )
    output_df["owner_role"] = config.get(
        "default_owner_role",
        "Care Management Team",
    )
    output_df["outreach_attempt_count"] = 0
    output_df["last_outreach_date"] = None
    output_df["next_outreach_action"] = "Initial Review"
    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "outreach_tracking"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = "case_management_worklist"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="outreach_tracking",
        rule_name="default_outreach_tracking",
        rule_type="status_initialization",
        description=(
            "Initializes outreach tracking fields for case management worklist."
        ),
        source_dataset="case_management_worklist",
        source_layer=runtime.layer_name,
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="outreach_tracking",
        dataset_type="care_management_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Outreach tracking built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="case_management_worklist",
    )

    logger.info("COMPLETE: Build outreach tracking | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Program Effectiveness
###############################################################################

def build_program_effectiveness(
    runtime: AnalyticsDomainRuntime,
    care_programs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build program effectiveness summary.

    Parameters
    ----------
    runtime:
        Care Management runtime.

    care_programs:
        Care program assignment dataframe.

    Returns
    -------
    pandas.DataFrame
        Program effectiveness summary.

    Raises
    ------
    ValidationError
        Raised when configured grouping or metric columns are missing.

    Notes
    -----
    Current version summarizes assigned members by care program. Future versions
    can add outcome tracking once intervention and follow-up event data exists.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("program_effectiveness", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Program effectiveness disabled")
        return pd.DataFrame()

    if care_programs.empty:
        return pd.DataFrame()

    group_by = config.get("group_by", [])
    metrics = config.get("metrics", {})

    require_columns(
        runtime=runtime,
        dataframe=care_programs,
        dataset_name="care_programs",
        required_columns=group_by,
        source_layer=runtime.layer_name,
        source_dataset="care_programs",
    )

    output_df = care_programs[group_by].drop_duplicates().copy()

    for metric_name, metric_config in metrics.items():
        metric_column = metric_config.get("column")

        if metric_config.get("calculation_type") != "count_rows":
            require_columns(
                runtime=runtime,
                dataframe=care_programs,
                dataset_name="care_programs",
                required_columns=[metric_column],
                source_layer=runtime.layer_name,
                source_dataset="care_programs",
            )

        metric_df = calculate_group_metric(
            dataframe=care_programs,
            group_by=group_by,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        output_df = output_df.merge(metric_df, on=group_by, how="left")

        add_rule_record(
            runtime=runtime,
            rule_group="program_effectiveness",
            rule_name=metric_name,
            rule_type=metric_config.get("calculation_type", ""),
            description=f"Program effectiveness metric: {metric_name}",
            source_dataset="care_programs",
            source_layer=runtime.layer_name,
            rule_config=metric_config,
        )

    output_df["analytics_layer_run_id"] = runtime.context.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "program_effectiveness"
    output_df["source_layer"] = runtime.layer_name
    output_df["source_dataset"] = "care_programs"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="program_effectiveness",
        dataset_type="care_management_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Program effectiveness built successfully.",
        source_layer=runtime.layer_name,
        source_dataset="care_programs",
    )

    logger.info("COMPLETE: Build program effectiveness | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_care_management_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Care Management layer.

    Parameters
    ----------
    config_path:
        Care Management configuration path.

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
    This layer consumes configured upstream analytics outputs and produces care
    programs, worklists, transitions-of-care registries, disease management
    candidates, outreach tracking records, and program effectiveness summaries.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Care Management started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        member_key = runtime.config.get("join_keys", {}).get(
            "member_key",
            "member_id",
        )
        provider_key = runtime.config.get("join_keys", {}).get(
            "provider_key",
            "provider_id",
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "high_priority_member_registry": "Layer 2D - Predictive Analytics",
                "member_prediction_summary": "Layer 2D - Predictive Analytics",
                "population_segment_registry": "Population Health Analytics",
                "condition_registry": "Clinical Analytics",
                "quality_gap_registry": "Quality Analytics",
                "provider_performance_summary": "Layer 2F - Provider Analytics",
            },
            key_column_map={
                "high_priority_member_registry": member_key,
                "member_prediction_summary": member_key,
                "population_segment_registry": member_key,
                "condition_registry": member_key,
                "quality_gap_registry": member_key,
                "provider_performance_summary": provider_key,
            },
        )

        care_programs = build_care_programs(
            runtime=runtime,
            datasets=datasets,
        )

        case_management_worklist = build_case_management_worklist(
            runtime=runtime,
            care_programs=care_programs,
        )

        transitions_of_care = build_transitions_of_care(
            runtime=runtime,
            datasets=datasets,
        )

        disease_management = build_disease_management(
            runtime=runtime,
            datasets=datasets,
        )

        outreach_tracking = build_outreach_tracking(
            runtime=runtime,
            case_management_worklist=case_management_worklist,
        )

        program_effectiveness = build_program_effectiveness(
            runtime=runtime,
            care_programs=care_programs,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "care_programs": care_programs,
            "case_management_worklist": case_management_worklist,
            "transitions_of_care": transitions_of_care,
            "disease_management": disease_management,
            "outreach_tracking": outreach_tracking,
            "program_effectiveness": program_effectiveness,
        }

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_care_management_layer",
            message="Care Management completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2G - Care Management",
            source_dataset="care_management_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="care_management_dataset_inventory",
            column_dictionary_name="care_management_column_dictionary",
            rule_catalog_name="care_management_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="care_management_audit_records",
            validation_results_name="care_management_validation_results",
            execution_summary_name="care_management_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Care Management completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name=DOMAIN_SECTION,
            status=STATUS_SUCCESS,
            message="Care Management completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(
                len(dataframe.columns) for dataframe in output_assets.values()
            ),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Care Management failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_care_management_layer",
                message=str(exc),
                source_layer="Layer 2G - Care Management",
                source_dataset="care_management",
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
                    audit_records_name="care_management_audit_records",
                    validation_results_name="care_management_validation_results",
                    execution_summary_name="care_management_execution_summary",
                )

            except Exception as audit_exc:
                logger.error(
                    "Failed to write Care Management audit outputs: %s",
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
    Command-line entry point for Care Management.

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
        "MEDFABRIC_CARE_MANAGEMENT_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_care_management_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Care Management failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()