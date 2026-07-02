###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/value_based_care/build_value_based_care_layer.py
#
# Layer:
#     Layer 2H - Value-Based Care
#
# Purpose:
#     Builds Value-Based Care analytics outputs from Provider Analytics,
#     Predictive Analytics, Care Management, and Gold outputs.
#
# Architectural Rule:
#     Value-Based Care consumes already-built upstream analytics outputs.
#
#     This file does NOT generate raw data.
#     This file does NOT build Silver dimensional models.
#     This file does NOT train predictive models.
#     This file does NOT score members.
#
# Dependency Flow:
#     Gold Layer
#         ↓
#     Provider Analytics + Predictive Analytics + Care Management
#         ↓
#     Layer 2H - Value-Based Care
#         ↓
#     data/analytics_platform/value_based_care/
#
# Architecture:
#     This file contains Value-Based Care business logic only.
#
#     Shared Analytics Platform concerns are handled by:
#         - src.analytics_platform.common.runtime
#         - src.analytics_platform.common.io
#         - src.analytics_platform.common.audit
#         - src.analytics_platform.common.validation
#         - src.analytics_platform.common.metadata
#
# Inputs:
#     config/analytics_platform/value_based_care.yaml
#
# Outputs:
#     data/analytics_platform/value_based_care/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.value_based_care.build_value_based_care_layer
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

DEFAULT_CONFIG_PATH = "config/analytics_platform/value_based_care.yaml"

DEFAULT_LAYER_NAME = "Layer 2H - Value-Based Care"
DEFAULT_DOMAIN_NAME = "Value-Based Care"
DOMAIN_SECTION = "value_based_care"

LOGGER_NAME = "medfabric.analytics_platform.value_based_care"


###############################################################################
# Configuration and Runtime
###############################################################################

def validate_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate required Value-Based Care configuration sections.

    Parameters
    ----------
    config:
        Loaded Value-Based Care YAML configuration.

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
    PipelineContext. This function validates only the Value-Based Care domain
    contract.
    """

    required_sections = [
        "value_based_care",
        "paths",
        "join_keys",
        "value_based_care_framework",
        "value_based_contract_summary",
        "provider_incentive_summary",
        "shared_savings_summary",
        "risk_adjustment_summary",
        "bundle_opportunity_summary",
        "vbc_executive_scorecard",
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
            "Value-Based Care configuration validation failed. "
            f"Missing sections: {missing_sections}"
        )


def initialize_runtime(config_path: str) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Initialize Value-Based Care runtime using MedFabric PipelineContext.

    Parameters
    ----------
    config_path:
        Value-Based Care configuration path.

    Returns
    -------
    AnalyticsDomainRuntime
        Initialized Value-Based Care runtime.

    Raises
    ------
    PipelineError
        Raised when configuration loading or validation fails.

    Notes
    -----
    This follows the same shared framework pattern as the working Predictive
    Analytics and Provider Analytics layers.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2H - Value-Based Care"
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
        message="Value-Based Care runtime initialized successfully.",
        source_layer="Layer 2 - Analytics Platform",
        source_dataset=config_file,
    )

    return runtime


###############################################################################
# Calculation Helpers
###############################################################################

def safe_divide(numerator: float, denominator: float) -> float:
    """
    Purpose
    -------
    Safely divide two numeric values.

    Parameters
    ----------
    numerator:
        Numerator value.

    denominator:
        Denominator value.

    Returns
    -------
    float
        Division result. Returns zero when denominator is missing or zero.

    Raises
    ------
    None
    """

    if denominator is None or denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


###############################################################################
# Value-Based Contract Summary
###############################################################################

def build_value_based_contract_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build value-based contract summary.

    Parameters
    ----------
    runtime:
        Value-Based Care runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Value-based contract summary.

    Raises
    ------
    ValidationError
        Raised when required source datasets or columns are missing.

    Notes
    -----
    Output grain is one row per configured synthetic value-based contract.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("value_based_contract_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Value-based contract summary disabled")
        return pd.DataFrame()

    source_datasets = config.get("source_datasets", {})
    pmpm_dataset_name = source_datasets.get("pmpm_dataset", "pmpm_summary")
    cost_dataset_name = source_datasets.get("cost_dataset", "cost_summary")
    provider_dataset_name = source_datasets.get(
        "provider_dataset",
        "provider_performance_summary",
    )

    for dataset_name in [pmpm_dataset_name, cost_dataset_name, provider_dataset_name]:
        if dataset_name not in datasets:
            raise ValidationError(
                f"Value-Based Contract source dataset missing: {dataset_name}"
            )

    pmpm_df = datasets[pmpm_dataset_name]
    cost_df = datasets[cost_dataset_name]
    provider_df = datasets[provider_dataset_name]

    financial_columns = config.get("financial_columns", {})
    benchmark = config.get("benchmark", {})

    required_pmpm_columns = [
        financial_columns.get("member_month_count"),
        financial_columns.get("total_paid_amount"),
        financial_columns.get("total_allowed_amount"),
        financial_columns.get("pmpm_paid_amount"),
        financial_columns.get("pmpm_allowed_amount"),
    ]

    require_columns(
        runtime=runtime,
        dataframe=pmpm_df,
        dataset_name=pmpm_dataset_name,
        required_columns=required_pmpm_columns,
        source_layer="Gold Layer",
        source_dataset=pmpm_dataset_name,
    )

    if pmpm_df.empty:
        raise ValidationError(f"PMPM source dataset is empty: {pmpm_dataset_name}")

    pmpm_row = pmpm_df.iloc[0]

    member_month_count = float(pmpm_row[financial_columns.get("member_month_count")])
    total_paid_amount = float(pmpm_row[financial_columns.get("total_paid_amount")])
    total_allowed_amount = float(pmpm_row[financial_columns.get("total_allowed_amount")])
    pmpm_paid_amount = float(pmpm_row[financial_columns.get("pmpm_paid_amount")])
    pmpm_allowed_amount = float(pmpm_row[financial_columns.get("pmpm_allowed_amount")])

    benchmark_pmpm_paid_amount = float(
        benchmark.get("benchmark_pmpm_paid_amount", 0.0)
    )
    benchmark_pmpm_allowed_amount = float(
        benchmark.get("benchmark_pmpm_allowed_amount", 0.0)
    )
    minimum_savings_rate = float(benchmark.get("minimum_savings_rate", 0.0))
    provider_share_rate = float(benchmark.get("provider_share_rate", 0.0))

    benchmark_total_paid_amount = benchmark_pmpm_paid_amount * member_month_count
    gross_savings_amount = benchmark_total_paid_amount - total_paid_amount
    gross_savings_rate = safe_divide(
        gross_savings_amount,
        benchmark_total_paid_amount,
    )

    minimum_savings_met = gross_savings_rate >= minimum_savings_rate
    shared_savings_amount = (
        gross_savings_amount * provider_share_rate
        if minimum_savings_met
        else 0.0
    )

    provider_count = (
        int(provider_df["provider_id"].nunique())
        if "provider_id" in provider_df.columns
        else int(len(provider_df))
    )

    output_df = pd.DataFrame(
        [
            {
                "value_based_contract_name": config.get("contract_name"),
                "contract_type": config.get("contract_type"),
                "payment_model": config.get("payment_model"),
                "measurement_period": config.get("measurement_period"),
                "provider_count": provider_count,
                "cost_category_count": int(len(cost_df)),
                "member_month_count": member_month_count,
                "total_paid_amount": total_paid_amount,
                "total_allowed_amount": total_allowed_amount,
                "pmpm_paid_amount": pmpm_paid_amount,
                "pmpm_allowed_amount": pmpm_allowed_amount,
                "benchmark_pmpm_paid_amount": benchmark_pmpm_paid_amount,
                "benchmark_pmpm_allowed_amount": benchmark_pmpm_allowed_amount,
                "benchmark_total_paid_amount": benchmark_total_paid_amount,
                "minimum_savings_rate": minimum_savings_rate,
                "gross_savings_amount": gross_savings_amount,
                "gross_savings_rate": gross_savings_rate,
                "minimum_savings_met": minimum_savings_met,
                "provider_share_rate": provider_share_rate,
                "shared_savings_amount": shared_savings_amount,
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "value_based_contract_summary",
                "source_layer": "Layer 2H - Value-Based Care",
                "source_dataset": pmpm_dataset_name,
                "built_at_utc": utc_now().isoformat(),
            }
        ]
    )

    add_rule_record(
        runtime=runtime,
        rule_group="value_based_contract_summary",
        rule_name="synthetic_shared_savings_contract",
        rule_type="contract_financial_summary",
        description=(
            "Builds synthetic value-based contract summary using PMPM "
            "benchmark comparison."
        ),
        source_dataset=pmpm_dataset_name,
        source_layer="Gold Layer",
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="value_based_contract_summary",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Value-based contract summary built successfully.",
        source_layer="Layer 2H - Value-Based Care",
        source_dataset=pmpm_dataset_name,
    )

    logger.info("COMPLETE: Build value-based contract summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Incentive Summary
###############################################################################

def build_provider_incentive_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build provider incentive summary.

    Parameters
    ----------
    runtime:
        Value-Based Care runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Provider incentive summary.

    Raises
    ------
    ValidationError
        Raised when configured source dataset or rule columns are missing.

    Notes
    -----
    Output grain is one row per provider per earned incentive rule.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("provider_incentive_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider incentive summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance_summary")
    provider_key = config.get("provider_key", "provider_id")

    if source_dataset not in datasets:
        raise ValidationError(
            f"Provider incentive source dataset missing: {source_dataset}"
        )

    source_df = datasets[source_dataset]
    output_frames: List[pd.DataFrame] = []

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=[provider_key],
        source_layer="Layer 2F - Provider Analytics",
        source_dataset=source_dataset,
    )

    logger.info("START: Build provider incentive summary")

    for rule_key, rule_config in config.get("incentive_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            logger.info("SKIP provider incentive rule disabled: %s", rule_key)
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")
        incentive_amount = float(rule_config.get("incentive_amount", 0.0))

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[provider_key, rule_column],
            source_layer="Layer 2F - Provider Analytics",
            source_dataset=source_dataset,
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        incentive_df = source_df.loc[mask].copy()
        incentive_df["incentive_rule_key"] = rule_key
        incentive_df["incentive_name"] = rule_config.get("incentive_name", rule_key)
        incentive_df["incentive_amount"] = incentive_amount
        incentive_df["incentive_description"] = rule_config.get("description", "")
        incentive_df["analytics_layer_run_id"] = runtime.context.run_id
        incentive_df["analytics_domain"] = runtime.domain_name
        incentive_df["analytics_asset_name"] = "provider_incentive_summary"
        incentive_df["source_layer"] = "Layer 2F - Provider Analytics"
        incentive_df["source_dataset"] = source_dataset
        incentive_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="provider_incentive_summary",
            rule_name=rule_key,
            rule_type="provider_incentive_assignment",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
            source_layer="Layer 2F - Provider Analytics",
            rule_config=rule_config,
        )

        output_frames.append(incentive_df)

        logger.info(
            "Applied provider incentive rule: %s | Providers: %s",
            rule_key,
            len(incentive_df),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                provider_key,
                "incentive_rule_key",
                "incentive_name",
                "incentive_amount",
                "incentive_description",
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
        dataset_name="provider_incentive_summary",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider incentive summary built successfully.",
        source_layer="Layer 2F - Provider Analytics",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build provider incentive summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Shared Savings Summary
###############################################################################

def build_shared_savings_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build shared savings summary.

    Parameters
    ----------
    runtime:
        Value-Based Care runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Shared savings summary.

    Raises
    ------
    ValidationError
        Raised when required financial columns are missing.

    Notes
    -----
    This summary compares actual PMPM cost against configured benchmark PMPM
    cost and calculates provider shared savings when minimum savings is met.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("shared_savings_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Shared savings summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "pmpm_summary")

    if source_dataset not in datasets:
        raise ValidationError(f"Shared savings source dataset missing: {source_dataset}")

    pmpm_df = datasets[source_dataset]

    require_columns(
        runtime=runtime,
        dataframe=pmpm_df,
        dataset_name=source_dataset,
        required_columns=[
            "member_month_count",
            "total_paid_amount",
            "pmpm_paid_amount",
        ],
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    if pmpm_df.empty:
        raise ValidationError(f"Shared savings source dataset is empty: {source_dataset}")

    row = pmpm_df.iloc[0]
    benchmark = config.get("benchmark", {})

    benchmark_pmpm_paid_amount = float(
        benchmark.get("benchmark_pmpm_paid_amount", 0.0)
    )
    minimum_savings_rate = float(benchmark.get("minimum_savings_rate", 0.0))
    provider_share_rate = float(benchmark.get("provider_share_rate", 0.0))

    member_month_count = float(row["member_month_count"])
    actual_total_paid_amount = float(row["total_paid_amount"])
    actual_pmpm_paid_amount = float(row["pmpm_paid_amount"])

    benchmark_total_paid_amount = benchmark_pmpm_paid_amount * member_month_count
    gross_savings_amount = benchmark_total_paid_amount - actual_total_paid_amount
    gross_savings_rate = safe_divide(
        gross_savings_amount,
        benchmark_total_paid_amount,
    )
    minimum_savings_met = gross_savings_rate >= minimum_savings_rate

    provider_shared_savings_amount = (
        gross_savings_amount * provider_share_rate
        if minimum_savings_met
        else 0.0
    )
    payer_retained_savings_amount = (
        gross_savings_amount - provider_shared_savings_amount
        if minimum_savings_met
        else 0.0
    )

    output_df = pd.DataFrame(
        [
            {
                "savings_method": config.get("savings_method"),
                "member_month_count": member_month_count,
                "actual_pmpm_paid_amount": actual_pmpm_paid_amount,
                "benchmark_pmpm_paid_amount": benchmark_pmpm_paid_amount,
                "actual_total_paid_amount": actual_total_paid_amount,
                "benchmark_total_paid_amount": benchmark_total_paid_amount,
                "gross_savings_amount": gross_savings_amount,
                "gross_savings_rate": gross_savings_rate,
                "minimum_savings_rate": minimum_savings_rate,
                "minimum_savings_met": minimum_savings_met,
                "provider_share_rate": provider_share_rate,
                "provider_shared_savings_amount": provider_shared_savings_amount,
                "payer_retained_savings_amount": payer_retained_savings_amount,
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "shared_savings_summary",
                "source_layer": "Gold Layer",
                "source_dataset": source_dataset,
                "built_at_utc": utc_now().isoformat(),
            }
        ]
    )

    add_rule_record(
        runtime=runtime,
        rule_group="shared_savings_summary",
        rule_name="pmpm_benchmark_comparison",
        rule_type="shared_savings_calculation",
        description="Calculates shared savings using PMPM benchmark comparison.",
        source_dataset=source_dataset,
        source_layer="Gold Layer",
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="shared_savings_summary",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Shared savings summary built successfully.",
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build shared savings summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Risk Adjustment Summary
###############################################################################

def build_risk_adjustment_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build risk adjustment summary from member prediction summary.

    Parameters
    ----------
    runtime:
        Value-Based Care runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Risk adjustment summary.

    Raises
    ------
    ValidationError
        Raised when required risk score columns are missing.

    Notes
    -----
    Output grain is one row per configured risk band.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("risk_adjustment_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Risk adjustment summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "member_prediction_summary")

    if source_dataset not in datasets:
        raise ValidationError(
            f"Risk adjustment source dataset missing: {source_dataset}"
        )

    source_df = datasets[source_dataset]
    risk_columns = config.get("risk_columns", {})
    score_column = risk_columns.get("max_prediction_score", "max_prediction_score")

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=[score_column],
        source_layer="Layer 2D - Predictive Analytics",
        source_dataset=source_dataset,
    )

    output_rows: List[Dict[str, Any]] = []

    logger.info("START: Build risk adjustment summary")

    for band_key, band_config in config.get("risk_bands", {}).items():
        min_score = float(band_config.get("min_score", 0.0))
        max_score = float(band_config.get("max_score", 1.0))
        risk_weight = float(band_config.get("risk_weight", 1.0))
        label = band_config.get("label", band_key)

        mask = (
            (source_df[score_column] >= min_score)
            & (source_df[score_column] <= max_score)
        )

        member_count = int(mask.sum())
        weighted_member_count = member_count * risk_weight

        output_rows.append(
            {
                "risk_band_key": band_key,
                "risk_band_label": label,
                "min_score": min_score,
                "max_score": max_score,
                "risk_weight": risk_weight,
                "member_count": member_count,
                "weighted_member_count": weighted_member_count,
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "risk_adjustment_summary",
                "source_layer": "Layer 2D - Predictive Analytics",
                "source_dataset": source_dataset,
                "built_at_utc": utc_now().isoformat(),
            }
        )

        add_rule_record(
            runtime=runtime,
            rule_group="risk_adjustment_summary",
            rule_name=band_key,
            rule_type="risk_band_assignment",
            description=f"Risk adjustment band: {label}",
            source_dataset=source_dataset,
            source_layer="Layer 2D - Predictive Analytics",
            rule_config=band_config,
        )

    output_df = pd.DataFrame(output_rows)

    add_dataset_record(
        runtime=runtime,
        dataset_name="risk_adjustment_summary",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Risk adjustment summary built successfully.",
        source_layer="Layer 2D - Predictive Analytics",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build risk adjustment summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Bundle Opportunity Summary
###############################################################################

def build_bundle_opportunity_summary(
    runtime: AnalyticsDomainRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build bundle opportunity summary from cost summary.

    Parameters
    ----------
    runtime:
        Value-Based Care runtime.

    datasets:
        Loaded input datasets.

    Returns
    -------
    pandas.DataFrame
        Bundle opportunity summary.

    Raises
    ------
    ValidationError
        Raised when configured bundle opportunity columns are missing.

    Notes
    -----
    Output grain depends on the configured source cost summary and opportunity
    rules.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("bundle_opportunity_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Bundle opportunity summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "cost_summary")

    if source_dataset not in datasets:
        raise ValidationError(
            f"Bundle opportunity source dataset missing: {source_dataset}"
        )

    source_df = datasets[source_dataset]
    group_by = config.get("group_by", [])

    require_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=group_by,
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build bundle opportunity summary")

    for rule_key, rule_config in config.get("opportunity_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            logger.info("SKIP bundle opportunity rule disabled: %s", rule_key)
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        require_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=group_by + [rule_column],
            source_layer="Gold Layer",
            source_dataset=source_dataset,
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        opportunity_df = source_df.loc[mask].copy()
        opportunity_df["opportunity_rule_key"] = rule_key
        opportunity_df["opportunity_name"] = rule_config.get(
            "opportunity_name",
            rule_key,
        )
        opportunity_df["opportunity_priority_rank"] = rule_config.get(
            "opportunity_priority_rank"
        )
        opportunity_df["opportunity_description"] = rule_config.get("description", "")
        opportunity_df["analytics_layer_run_id"] = runtime.context.run_id
        opportunity_df["analytics_domain"] = runtime.domain_name
        opportunity_df["analytics_asset_name"] = "bundle_opportunity_summary"
        opportunity_df["source_layer"] = "Gold Layer"
        opportunity_df["source_dataset"] = source_dataset
        opportunity_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="bundle_opportunity_summary",
            rule_name=rule_key,
            rule_type="bundle_opportunity_selection",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
            source_layer="Gold Layer",
            rule_config=rule_config,
        )

        output_frames.append(opportunity_df)

        logger.info(
            "Applied bundle opportunity rule: %s | Rows: %s",
            rule_key,
            len(opportunity_df),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame()

    add_dataset_record(
        runtime=runtime,
        dataset_name="bundle_opportunity_summary",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Bundle opportunity summary built successfully.",
        source_layer="Gold Layer",
        source_dataset=source_dataset,
    )

    logger.info("COMPLETE: Build bundle opportunity summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# VBC Executive Scorecard
###############################################################################

def build_vbc_executive_scorecard(
    runtime: AnalyticsDomainRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Purpose
    -------
    Build executive scorecard from Value-Based Care output assets.

    Parameters
    ----------
    runtime:
        Value-Based Care runtime.

    output_assets:
        Value-Based Care output assets already built during this run.

    Returns
    -------
    pandas.DataFrame
        VBC executive scorecard.

    Raises
    ------
    None

    Notes
    -----
    This creates a compact executive-ready summary of Value-Based Care outputs.
    """

    logger = runtime.get_logger(LOGGER_NAME)
    config = runtime.config.get("vbc_executive_scorecard", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: VBC executive scorecard disabled")
        return pd.DataFrame()

    source_assets = config.get("source_assets", [])
    rows: List[Dict[str, Any]] = []

    for asset_name in source_assets:
        asset_df = output_assets.get(asset_name, pd.DataFrame())

        rows.append(
            {
                "scorecard_section": asset_name,
                "row_count": int(len(asset_df)),
                "column_count": int(len(asset_df.columns)),
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "vbc_executive_scorecard",
                "source_layer": "Layer 2H - Value-Based Care",
                "source_dataset": asset_name,
                "built_at_utc": utc_now().isoformat(),
            }
        )

    contract_df = output_assets.get("value_based_contract_summary", pd.DataFrame())

    if not contract_df.empty:
        contract_row = contract_df.iloc[0]

        rows.append(
            {
                "scorecard_section": "contract_financials",
                "row_count": 1,
                "column_count": len(contract_df.columns),
                "total_paid_amount": contract_row.get("total_paid_amount"),
                "pmpm_paid_amount": contract_row.get("pmpm_paid_amount"),
                "gross_savings_amount": contract_row.get("gross_savings_amount"),
                "gross_savings_rate": contract_row.get("gross_savings_rate"),
                "shared_savings_amount": contract_row.get("shared_savings_amount"),
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "vbc_executive_scorecard",
                "source_layer": "Layer 2H - Value-Based Care",
                "source_dataset": "value_based_contract_summary",
                "built_at_utc": utc_now().isoformat(),
            }
        )

    incentive_df = output_assets.get("provider_incentive_summary", pd.DataFrame())

    if not incentive_df.empty and "incentive_amount" in incentive_df.columns:
        rows.append(
            {
                "scorecard_section": "provider_incentives",
                "row_count": int(len(incentive_df)),
                "column_count": int(len(incentive_df.columns)),
                "total_incentive_amount": float(incentive_df["incentive_amount"].sum()),
                "analytics_layer_run_id": runtime.context.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "vbc_executive_scorecard",
                "source_layer": "Layer 2H - Value-Based Care",
                "source_dataset": "provider_incentive_summary",
                "built_at_utc": utc_now().isoformat(),
            }
        )

    output_df = pd.DataFrame(rows)

    add_rule_record(
        runtime=runtime,
        rule_group="vbc_executive_scorecard",
        rule_name="executive_scorecard_assembly",
        rule_type="scorecard_summary",
        description=(
            "Builds executive scorecard from Value-Based Care output assets."
        ),
        source_dataset="value_based_care_outputs",
        source_layer="Layer 2H - Value-Based Care",
        rule_config=config,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="vbc_executive_scorecard",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="VBC executive scorecard built successfully.",
        source_layer="Layer 2H - Value-Based Care",
        source_dataset="value_based_care_outputs",
    )

    logger.info("COMPLETE: Build VBC executive scorecard | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Main Orchestration
###############################################################################

def build_value_based_care_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AnalyticsBuildResult:
    """
    Purpose
    -------
    Build the complete Value-Based Care layer.

    Parameters
    ----------
    config_path:
        Value-Based Care configuration path.

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
    This layer consumes configured upstream Gold, Provider Analytics, Predictive
    Analytics, and Care Management outputs and produces Value-Based Care
    summaries and scorecards.
    """

    runtime: Optional[AnalyticsDomainRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.get_logger(LOGGER_NAME)

        logger.info("=" * 80)
        logger.info("MedFabric Value-Based Care started")
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
        member_key = runtime.config.get("join_keys", {}).get(
            "member_key",
            "member_id",
        )

        datasets = load_input_datasets(
            runtime=runtime,
            input_source_layer_map={
                "pmpm_summary": "Gold Layer",
                "cost_summary": "Gold Layer",
                "provider_performance_summary": "Layer 2F - Provider Analytics",
                "member_prediction_summary": "Layer 2D - Predictive Analytics",
                "care_management_summary": "Care Management",
            },
            key_column_map={
                "provider_performance_summary": provider_key,
                "member_prediction_summary": member_key,
                "care_management_summary": member_key,
            },
        )

        value_based_contract_summary = build_value_based_contract_summary(
            runtime=runtime,
            datasets=datasets,
        )

        provider_incentive_summary = build_provider_incentive_summary(
            runtime=runtime,
            datasets=datasets,
        )

        shared_savings_summary = build_shared_savings_summary(
            runtime=runtime,
            datasets=datasets,
        )

        risk_adjustment_summary = build_risk_adjustment_summary(
            runtime=runtime,
            datasets=datasets,
        )

        bundle_opportunity_summary = build_bundle_opportunity_summary(
            runtime=runtime,
            datasets=datasets,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "value_based_contract_summary": value_based_contract_summary,
            "provider_incentive_summary": provider_incentive_summary,
            "shared_savings_summary": shared_savings_summary,
            "risk_adjustment_summary": risk_adjustment_summary,
            "bundle_opportunity_summary": bundle_opportunity_summary,
        }

        vbc_executive_scorecard = build_vbc_executive_scorecard(
            runtime=runtime,
            output_assets=output_assets,
        )

        output_assets["vbc_executive_scorecard"] = vbc_executive_scorecard

        write_output_assets(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        add_success_audit(
            runtime=runtime,
            step_name="build_value_based_care_layer",
            message="Value-Based Care completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            source_layer="Layer 2H - Value-Based Care",
            source_dataset="value_based_care_outputs",
        )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="value_based_care_dataset_inventory",
            column_dictionary_name="value_based_care_column_dictionary",
            rule_catalog_name="value_based_care_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            audit_records_name="value_based_care_audit_records",
            validation_results_name="value_based_care_validation_results",
            execution_summary_name="value_based_care_execution_summary",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Value-Based Care completed successfully")
        logger.info("=" * 80)

        return AnalyticsBuildResult(
            name=DOMAIN_SECTION,
            status=STATUS_SUCCESS,
            message="Value-Based Care completed successfully.",
            row_count=sum(len(dataframe) for dataframe in output_assets.values()),
            column_count=sum(
                len(dataframe.columns) for dataframe in output_assets.values()
            ),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.get_logger(LOGGER_NAME)

            logger.error("=" * 80)
            logger.error("Value-Based Care failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_value_based_care_layer",
                message=str(exc),
                source_layer="Layer 2H - Value-Based Care",
                source_dataset="value_based_care",
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
                    audit_records_name="value_based_care_audit_records",
                    validation_results_name="value_based_care_validation_results",
                    execution_summary_name="value_based_care_execution_summary",
                )

            except Exception as audit_exc:
                logger.error(
                    "Failed to write Value-Based Care audit outputs: %s",
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
    Command-line entry point for Value-Based Care.

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
        "MEDFABRIC_VALUE_BASED_CARE_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_value_based_care_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Value-Based Care failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()