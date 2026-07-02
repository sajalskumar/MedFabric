###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/quality_analytics/build_quality_analytics_layer.py
#
# Layer:
#     Layer 2C - Quality Analytics
#
# Purpose:
#     Builds Quality Analytics outputs using the configured YAML contract:
#
#         config/analytics_platform/quality_analytics.yaml
#
# Business Context:
#     Quality Analytics identifies care gaps, preventive care signals,
#     medication adherence proxies, and measure-style summaries using the
#     currently available MedFabric aggregate feature assets.
#
# Architecture:
#     This file contains Quality Analytics business logic only.
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
#     config/analytics_platform/quality_analytics.yaml
#
# Outputs:
#     data/analytics_platform/quality_analytics/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.quality_analytics.build_quality_analytics_layer
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

DEFAULT_CONFIG_PATH = "config/analytics_platform/quality_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2C - Quality Analytics"
DEFAULT_DOMAIN_NAME = "Quality Analytics"
DOMAIN_SECTION = "quality_analytics"

LOGGER_NAME = "medfabric.analytics_platform.quality_analytics"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Quality Analytics configuration sections.

    Parameters
    ----------
    config:
        Loaded Quality Analytics YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required sections are missing.

    Notes
    -----
    Generic YAML loading is handled by ConfigurationManager. This function only
    validates the Quality Analytics domain contract.
    """

    required_sections = [
        "quality_analytics",
        "paths",
        "join_keys",
        "quality_framework",
        "quality_measures",
        "care_gaps",
        "preventive_care",
        "medication_adherence",
        "hedis_summary",
        "cms_summary",
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
            "Quality Analytics configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Quality Analytics runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Quality Analytics configuration path.

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
        pipeline_name="Layer 2C - Quality Analytics"
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
        message="Quality Analytics runtime initialized successfully.",
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
    Build base member universe for Quality Analytics.

    Parameters
    ----------
    runtime:
        Quality Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Member-level universe dataframe.

    Raises
    ------
    ValidationError
        Raised when the configured base dataset or member key is missing.

    Notes
    -----
    This function supports future quality logic that may need the full eligible
    member universe before applying numerator or gap rules.
    """

    framework_config = runtime.config.get("quality_framework", {})
    base_dataset_name = framework_config.get("base_member_dataset", "risk_features")
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
# Quality Measures
###############################################################################

def build_quality_measures(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build configured signal-based quality measures.

    Parameters
    ----------
    runtime:
        Quality Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Measure-level quality measure summary.

    Raises
    ------
    ValidationError
        Raised when configured measure inputs or columns are missing.

    Notes
    -----
    Output grain is one row per configured quality measure.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("quality_measures", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Quality measures disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    rows: List[Dict[str, Any]] = []

    logger.info("START: Build quality measures")

    for measure_key, measure_config in config.get("measures", {}).items():
        if not bool(measure_config.get("enabled", True)):
            logger.info("SKIP measure disabled: %s", measure_key)
            continue

        source_dataset_name = measure_config.get("source_dataset")
        denominator_rule = measure_config.get("denominator_rule", {})
        numerator_rule = measure_config.get("numerator_rule", {})

        measure_name = measure_config.get("measure_name", measure_key)
        measure_category = measure_config.get("measure_category", "")
        measure_type = measure_config.get("measure_type", "signal_based_quality_measure")
        description = measure_config.get("description", "")

        add_rule_record(
            runtime=runtime,
            rule_group="quality_measures",
            rule_name=measure_key,
            rule_type=measure_type,
            description=description,
            source_dataset=str(source_dataset_name),
            source_layer="Layer 1F - Feature Store",
            rule_config=measure_config,
        )

        if source_dataset_name not in datasets:
            raise ValidationError(
                f"Quality measure source dataset missing: {source_dataset_name}"
            )

        source_df = datasets[source_dataset_name]

        denominator_column = denominator_rule.get("column")
        numerator_column = numerator_rule.get("column")

        required_columns = [member_key, numerator_column]
        if denominator_column:
            required_columns.append(denominator_column)

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=required_columns,
            source_layer="Layer 1F - Feature Store",
            source_dataset=source_dataset_name,
        )

        denominator_operator = denominator_rule.get("operator")
        denominator_value = denominator_rule.get("value")
        if denominator_column and denominator_operator:
            denominator_mask = apply_operator(
                source_df[denominator_column],
                denominator_operator,
                denominator_value,
            )
            denominator_df = source_df.loc[denominator_mask].copy()
        else:
            denominator_df = source_df.copy()

        denominator_count = int(denominator_df[member_key].nunique(dropna=True))

        numerator_mask = apply_operator(
            denominator_df[numerator_column],
            numerator_rule.get("operator"),
            numerator_rule.get("value"),
        )

        numerator_count = int(
            denominator_df.loc[numerator_mask, member_key].nunique(dropna=True)
        )

        rate = numerator_count / denominator_count if denominator_count else 0.0

        rows.append(
            {
                "measure_key": measure_key,
                "measure_name": measure_name,
                "measure_category": measure_category,
                "measure_type": measure_type,
                "source_dataset": source_dataset_name,
                "denominator_count": denominator_count,
                "numerator_count": numerator_count,
                "measure_rate": rate,
                "description": description,
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "quality_measures",
                "source_layer": "Layer 1F - Feature Store",
                "built_at_utc": utc_now().isoformat(),
            }
        )

        logger.info(
            "Built quality measure: %s | Numerator: %s | Denominator: %s | Rate: %.4f",
            measure_name,
            numerator_count,
            denominator_count,
            rate,
        )

    output_df = pd.DataFrame(rows)

    add_dataset_record(
        runtime=runtime,
        dataset_name="quality_measures",
        dataset_type="quality_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Quality measures built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset="configured_quality_measures",
    )

    logger.info("COMPLETE: Build quality measures | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Care Gaps
###############################################################################

def build_care_gaps(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build configured care gap outputs.

    Parameters
    ----------
    runtime:
        Quality Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Member-level care gap output.

    Raises
    ------
    ValidationError
        Raised when configured care gap inputs or columns are missing.

    Notes
    -----
    Output grain is one row per member per care gap.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("care_gaps", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Care gaps disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    gap_frames: List[pd.DataFrame] = []

    logger.info("START: Build care gaps")

    for gap_key, gap_config in config.get("gaps", {}).items():
        if not bool(gap_config.get("enabled", True)):
            logger.info("SKIP care gap disabled: %s", gap_key)
            continue

        source_dataset_name = gap_config.get("source_dataset")
        gap_rule = gap_config.get("gap_rule", {})

        care_gap_name = gap_config.get("care_gap_name", gap_key)
        care_gap_category = gap_config.get("care_gap_category", "")
        severity = gap_config.get("severity", "Medium")
        description = gap_config.get("description", "")

        add_rule_record(
            runtime=runtime,
            rule_group="care_gaps",
            rule_name=gap_key,
            rule_type="signal_based_care_gap",
            description=description,
            source_dataset=str(source_dataset_name),
            source_layer="Layer 1F - Feature Store",
            rule_config=gap_config,
        )

        if source_dataset_name not in datasets:
            raise ValidationError(f"Care gap source dataset missing: {source_dataset_name}")

        source_df = datasets[source_dataset_name]

        signal_column = gap_rule.get("column")
        operator = gap_rule.get("operator")
        value = gap_rule.get("value")

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=[member_key, signal_column],
            source_layer="Layer 1F - Feature Store",
            source_dataset=source_dataset_name,
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched_df = source_df.loc[mask, [member_key, signal_column]].copy()
        matched_df = matched_df.rename(
            columns={signal_column: "care_gap_evidence_value"}
        )

        matched_df["care_gap_key"] = gap_key
        matched_df["care_gap_name"] = care_gap_name
        matched_df["care_gap_category"] = care_gap_category
        matched_df["care_gap_severity"] = severity
        matched_df["source_dataset"] = source_dataset_name
        matched_df["care_gap_description"] = description
        matched_df["analytics_layer_run_id"] = runtime.context.run_id
        matched_df["analytics_domain"] = runtime.domain_name
        matched_df["analytics_asset_name"] = "care_gaps"
        matched_df["source_layer"] = "Layer 1F - Feature Store"
        matched_df["built_at_utc"] = utc_now().isoformat()

        gap_frames.append(matched_df)

        logger.info(
            "Built care gap: %s | Members: %s",
            care_gap_name,
            len(matched_df),
        )

    if gap_frames:
        output_df = pd.concat(gap_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "care_gap_evidence_value",
                "care_gap_key",
                "care_gap_name",
                "care_gap_category",
                "care_gap_severity",
                "source_dataset",
                "care_gap_description",
                "analytics_layer_run_id",
                "analytics_domain",
                "analytics_asset_name",
                "source_layer",
                "built_at_utc",
            ]
        )

    add_dataset_record(
        runtime=runtime,
        dataset_name="care_gaps",
        dataset_type="quality_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Care gaps built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset="configured_care_gap_rules",
    )

    logger.info("COMPLETE: Build care gaps | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Preventive Care
###############################################################################

def build_preventive_care(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build configured preventive care signal outputs.

    Parameters
    ----------
    runtime:
        Quality Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Preventive care signal output.

    Raises
    ------
    ValidationError
        Raised when configured preventive care inputs or columns are missing.

    Notes
    -----
    Output grain is one row per member per preventive care signal.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("preventive_care", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Preventive care disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    signal_frames: List[pd.DataFrame] = []

    logger.info("START: Build preventive care signals")

    for signal_key, signal_config in config.get("preventive_signals", {}).items():
        if not bool(signal_config.get("enabled", True)):
            logger.info("SKIP preventive care signal disabled: %s", signal_key)
            continue

        source_dataset_name = signal_config.get("source_dataset")
        signal_rule = signal_config.get("signal_rule", {})

        preventive_care_name = signal_config.get("preventive_care_name", signal_key)
        preventive_care_category = signal_config.get("preventive_care_category", "")
        description = signal_config.get("description", "")

        add_rule_record(
            runtime=runtime,
            rule_group="preventive_care",
            rule_name=signal_key,
            rule_type="signal_based_preventive_care",
            description=description,
            source_dataset=str(source_dataset_name),
            source_layer="Layer 1F - Feature Store",
            rule_config=signal_config,
        )

        if source_dataset_name not in datasets:
            raise ValidationError(
                f"Preventive care source dataset missing: {source_dataset_name}"
            )

        source_df = datasets[source_dataset_name]

        signal_column = signal_rule.get("column")
        operator = signal_rule.get("operator")
        value = signal_rule.get("value")

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=[member_key, signal_column],
            source_layer="Layer 1F - Feature Store",
            source_dataset=source_dataset_name,
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched_df = source_df.loc[mask, [member_key, signal_column]].copy()
        matched_df = matched_df.rename(
            columns={signal_column: "preventive_evidence_value"}
        )

        matched_df["preventive_care_key"] = signal_key
        matched_df["preventive_care_name"] = preventive_care_name
        matched_df["preventive_care_category"] = preventive_care_category
        matched_df["source_dataset"] = source_dataset_name
        matched_df["preventive_care_description"] = description
        matched_df["analytics_layer_run_id"] = runtime.context.run_id
        matched_df["analytics_domain"] = runtime.domain_name
        matched_df["analytics_asset_name"] = "preventive_care"
        matched_df["source_layer"] = "Layer 1F - Feature Store"
        matched_df["built_at_utc"] = utc_now().isoformat()

        signal_frames.append(matched_df)

        logger.info(
            "Built preventive care signal: %s | Members: %s",
            preventive_care_name,
            len(matched_df),
        )

    if signal_frames:
        output_df = pd.concat(signal_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "preventive_evidence_value",
                "preventive_care_key",
                "preventive_care_name",
                "preventive_care_category",
                "source_dataset",
                "preventive_care_description",
                "analytics_layer_run_id",
                "analytics_domain",
                "analytics_asset_name",
                "source_layer",
                "built_at_utc",
            ]
        )

    add_dataset_record(
        runtime=runtime,
        dataset_name="preventive_care",
        dataset_type="quality_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Preventive care signals built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset="configured_preventive_care_rules",
    )

    logger.info("COMPLETE: Build preventive care | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Medication Adherence
###############################################################################

def build_medication_adherence(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build configured medication adherence signal outputs.

    Parameters
    ----------
    runtime:
        Quality Analytics runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Medication adherence signal output.

    Raises
    ------
    ValidationError
        Raised when configured adherence inputs or columns are missing.

    Notes
    -----
    Output grain is one row per member per adherence signal.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("medication_adherence", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Medication adherence disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    adherence_frames: List[pd.DataFrame] = []

    logger.info("START: Build medication adherence signals")

    for signal_key, signal_config in config.get("adherence_signals", {}).items():
        if not bool(signal_config.get("enabled", True)):
            logger.info("SKIP adherence signal disabled: %s", signal_key)
            continue

        source_dataset_name = signal_config.get("source_dataset")
        signal_rule = signal_config.get("signal_rule", {})

        adherence_name = signal_config.get("adherence_name", signal_key)
        adherence_category = signal_config.get("adherence_category", "")
        evidence_column = signal_config.get(
            "evidence_column",
            signal_rule.get("column"),
        )
        description = signal_config.get("description", "")

        add_rule_record(
            runtime=runtime,
            rule_group="medication_adherence",
            rule_name=signal_key,
            rule_type="signal_based_medication_adherence",
            description=description,
            source_dataset=str(source_dataset_name),
            source_layer="Layer 1F - Feature Store",
            rule_config=signal_config,
        )

        if source_dataset_name not in datasets:
            raise ValidationError(
                f"Medication adherence source dataset missing: {source_dataset_name}"
            )

        source_df = datasets[source_dataset_name]

        signal_column = signal_rule.get("column")
        operator = signal_rule.get("operator")
        value = signal_rule.get("value")

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=[member_key, signal_column, evidence_column],
            source_layer="Layer 1F - Feature Store",
            source_dataset=source_dataset_name,
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched_df = source_df.loc[mask, [member_key, evidence_column]].copy()
        matched_df = matched_df.rename(
            columns={evidence_column: "adherence_evidence_value"}
        )

        matched_df["adherence_key"] = signal_key
        matched_df["adherence_name"] = adherence_name
        matched_df["adherence_category"] = adherence_category
        matched_df["source_dataset"] = source_dataset_name
        matched_df["adherence_description"] = description
        matched_df["analytics_layer_run_id"] = runtime.context.run_id
        matched_df["analytics_domain"] = runtime.domain_name
        matched_df["analytics_asset_name"] = "medication_adherence"
        matched_df["source_layer"] = "Layer 1F - Feature Store"
        matched_df["built_at_utc"] = utc_now().isoformat()

        adherence_frames.append(matched_df)

        logger.info(
            "Built medication adherence signal: %s | Members: %s",
            adherence_name,
            len(matched_df),
        )

    if adherence_frames:
        output_df = pd.concat(adherence_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "adherence_evidence_value",
                "adherence_key",
                "adherence_name",
                "adherence_category",
                "source_dataset",
                "adherence_description",
                "analytics_layer_run_id",
                "analytics_domain",
                "analytics_asset_name",
                "source_layer",
                "built_at_utc",
            ]
        )

    add_dataset_record(
        runtime=runtime,
        dataset_name="medication_adherence",
        dataset_type="quality_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Medication adherence signals built successfully.",
        source_layer="Layer 1F - Feature Store",
        source_dataset="configured_medication_adherence_rules",
    )

    logger.info("COMPLETE: Build medication adherence | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Summary Outputs
###############################################################################

def build_measure_summary(
    runtime: AnalyticsDomainRuntime,
    quality_measures: pd.DataFrame,
    summary_config: Dict[str, Any],
    output_name: str,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build HEDIS or CMS signal-based summary from quality measures.

    Parameters
    ----------
    runtime:
        Quality Analytics runtime.

    quality_measures:
        Quality measure dataframe.

    summary_config:
        Summary configuration.

    output_name:
        Output asset name.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe.

    Raises
    ------
    None

    Notes
    -----
    These summaries are placeholders built from currently available aggregate
    signals. They can later be upgraded to event-level HEDIS/CMS logic.
    """

    if quality_measures.empty:
        return pd.DataFrame()

    included_measure_keys = summary_config.get("measures_included", [])
    summary_method = summary_config.get("summary_method", "signal_based_placeholder")
    note = summary_config.get("note", "")

    if included_measure_keys:
        summary_df = quality_measures[
            quality_measures["measure_key"].isin(included_measure_keys)
        ].copy()
    else:
        summary_df = quality_measures.copy()

    if summary_df.empty:
        summary_df = quality_measures.copy()

    summary_df["summary_name"] = output_name
    summary_df["summary_method"] = summary_method
    summary_df["summary_note"] = note
    summary_df["analytics_asset_name"] = output_name
    summary_df["analytics_layer_run_id"] = runtime.context.run_id
    summary_df["analytics_domain"] = runtime.domain_name
    summary_df["source_layer"] = "Layer 2C - Quality Analytics"
    summary_df["source_dataset"] = "quality_measures"
    summary_df["built_at_utc"] = utc_now().isoformat()

    return summary_df


###############################################################################
# Main Orchestration
###############################################################################

def build_quality_analytics_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Quality Analytics layer.

    Parameters
    ----------
    config_path:
        Quality Analytics configuration path.

    Returns
    -------
    AnalyticsBuildResult
        Standard build result containing status, message, row count, and column
        count.

    Raises
    ------
    None
        Exceptions are captured and returned as a failed AnalyticsBuildResult.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Quality Analytics started")
        logger.info("=" * 80)
        logger.info("Configuration file: %s", runtime.config_file)

        output_format = get_output_format(
            runtime=runtime,
            domain_section=DOMAIN_SECTION,
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "claims_features": "Layer 1F - Feature Store",
                "pharmacy_features": "Layer 1F - Feature Store",
                "laboratory_features": "Layer 1F - Feature Store",
                "risk_features": "Layer 1F - Feature Store",
                "member_segmentation": "Layer 2A - Population Health Analytics",
                "risk_stratification": "Layer 2A - Population Health Analytics",
                "condition_registry": "Layer 2B - Clinical Analytics",
            },
            key_column_map={
                "claims_features": "member_id",
                "pharmacy_features": "member_id",
                "laboratory_features": "member_id",
                "risk_features": "member_id",
                "member_segmentation": "member_id",
                "risk_stratification": "member_id",
                "condition_registry": "member_id",
            },
        )

        quality_measures = build_quality_measures(runtime, datasets)
        care_gaps = build_care_gaps(runtime, datasets)
        preventive_care = build_preventive_care(runtime, datasets)
        medication_adherence = build_medication_adherence(runtime, datasets)

        hedis_summary = build_measure_summary(
            runtime=runtime,
            quality_measures=quality_measures,
            summary_config=runtime.config.get("hedis_summary", {}),
            output_name="hedis_summary",
        )

        cms_summary = build_measure_summary(
            runtime=runtime,
            quality_measures=quality_measures,
            summary_config=runtime.config.get("cms_summary", {}),
            output_name="cms_summary",
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "quality_measures": quality_measures,
            "care_gaps": care_gaps,
            "preventive_care": preventive_care,
            "medication_adherence": medication_adherence,
            "hedis_summary": hedis_summary,
            "cms_summary": cms_summary,
        }

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_quality_analytics_layer",
            message="Quality Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2C - Quality Analytics",
            source_dataset="quality_analytics_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="quality_analytics_dataset_inventory",
            column_dictionary_name="quality_analytics_column_dictionary",
            rule_catalog_name="quality_analytics_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="quality_analytics_audit_records",
            validation_results_name="quality_analytics_validation_results",
            execution_summary_name="quality_analytics_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Quality Analytics completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name="quality_analytics",
            status=STATUS_SUCCESS,
            message="Quality Analytics completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(len(dataframe.columns) for dataframe in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Quality Analytics failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_quality_analytics_layer",
                message=str(exc),
                source_layer="Layer 2C - Quality Analytics",
                source_dataset="quality_analytics",
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
                    audit_records_name="quality_analytics_audit_records",
                    validation_results_name="quality_analytics_validation_results",
                    execution_summary_name="quality_analytics_execution_summary",
                )
            except Exception as audit_exc:
                logger.error("Failed to write audit outputs: %s", audit_exc)

        return AnalyticsBuildResult(
            name="quality_analytics",
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
    Command-line entry point for Quality Analytics.

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
        "MEDFABRIC_QUALITY_ANALYTICS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_quality_analytics_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Quality Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()