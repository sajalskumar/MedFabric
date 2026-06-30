###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/predictive_analytics/build_predictive_analytics_layer.py
#
# Layer:
#     Layer 2E - Predictive Analytics
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
#     Layer 2D - Model Training & Scoring
#         ↓
#     data/scoring/
#         ↓
#     Layer 2E - Predictive Analytics
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

import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/analytics_platform/predictive_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2E - Predictive Analytics"
DEFAULT_DOMAIN_NAME = "Predictive Analytics"
DEFAULT_OUTPUT_FORMAT = "parquet"

SUPPORTED_FILE_FORMATS = {"parquet", "csv", "json"}

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_WARNING = "WARNING"
STATUS_SKIPPED = "SKIPPED"


###############################################################################
# Runtime Data Classes
###############################################################################

@dataclass
class PredictiveAnalyticsRuntime:
    """
    Runtime context for one Predictive Analytics execution.
    """

    run_id: str
    project_root: Path
    config_path: Path
    start_time_utc: datetime
    config: Dict[str, Any]
    logger: logging.Logger
    layer_name: str
    domain_name: str
    audit_records: List[Dict[str, Any]] = field(default_factory=list)
    validation_records: List[Dict[str, Any]] = field(default_factory=list)
    dataset_records: List[Dict[str, Any]] = field(default_factory=list)
    rule_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildResult:
    """
    Standard build result returned by Predictive Analytics.
    """

    name: str
    status: str
    message: str
    row_count: int = 0
    column_count: int = 0


###############################################################################
# General Utilities
###############################################################################

def utc_now() -> datetime:
    """
    Return current timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


def generate_run_id() -> str:
    """
    Generate timestamp-based run ID.
    """

    return utc_now().strftime("%Y%m%d_%H%M%S")


def normalize_path(project_root: Path, raw_path: str | Path) -> Path:
    """
    Resolve configured path relative to project root.
    """

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_directory(path: Path) -> None:
    """
    Create directory if missing.
    """

    path.mkdir(parents=True, exist_ok=True)


def safe_string(value: Any) -> str:
    """
    Convert value to string safely.
    """

    if value is None:
        return ""

    return str(value)


###############################################################################
# Logging
###############################################################################

def configure_logging(
    project_root: Path,
    config: Dict[str, Any],
    run_id: str,
) -> logging.Logger:
    """
    Configure Predictive Analytics logging.
    """

    logging_config = config.get("logging", {})

    log_level_name = logging_config.get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)

    log_dir = normalize_path(
        project_root,
        logging_config.get("module_log_dir", "logs/modules"),
    )
    ensure_directory(log_dir)

    log_file_path = log_dir / logging_config.get(
        "log_file_name",
        "predictive_analytics.log",
    )

    logger = logging.getLogger("medfabric.analytics_platform.predictive_analytics")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [RUN_ID=%(run_id)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RunIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = run_id
            return True

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RunIdFilter())

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RunIdFilter())

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("MedFabric Predictive Analytics started")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Log file: %s", log_file_path)

    return logger


###############################################################################
# Configuration
###############################################################################

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required Predictive Analytics configuration sections.
    """

    errors: List[str] = []

    required_sections = [
        "predictive_analytics",
        "logging",
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

    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")

    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        errors.append("Configuration section 'paths' must be a dictionary.")
    else:
        for subsection in ["inputs", "outputs", "metadata_outputs", "audit_outputs"]:
            if subsection not in paths:
                errors.append(f"Missing required configuration section: paths.{subsection}")

    output_format = (
        config.get("predictive_analytics", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Predictive Analytics configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(
    config_path_raw: str = DEFAULT_CONFIG_PATH,
) -> PredictiveAnalyticsRuntime:
    """
    Initialize Predictive Analytics runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    predictive_config = config.get("predictive_analytics", {})

    layer_name = predictive_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = predictive_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = PredictiveAnalyticsRuntime(
        run_id=run_id,
        project_root=project_root,
        config_path=config_path,
        start_time_utc=utc_now(),
        config=config,
        logger=logger,
        layer_name=layer_name,
        domain_name=domain_name,
    )

    add_audit_record(
        runtime=runtime,
        step_name="initialize_runtime",
        status=STATUS_SUCCESS,
        message="Predictive Analytics runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: PredictiveAnalyticsRuntime,
    step_name: str,
    status: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Add audit record.
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "step_name": step_name,
            "status": status,
            "message": message,
            "row_count": row_count,
            "output_path": output_path,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_validation_record(
    runtime: PredictiveAnalyticsRuntime,
    dataset_name: str,
    rule_name: str,
    status: str,
    message: str,
    failed_count: Optional[int] = None,
) -> None:
    """
    Add validation record.
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "rule_name": rule_name,
            "status": status,
            "message": message,
            "failed_count": failed_count,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_dataset_record(
    runtime: PredictiveAnalyticsRuntime,
    dataset_name: str,
    dataset_type: str,
    status: str,
    path: Optional[str],
    row_count: int,
    column_count: int,
    message: str,
) -> None:
    """
    Add dataset inventory record.
    """

    runtime.dataset_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "status": status,
            "path": path,
            "row_count": row_count,
            "column_count": column_count,
            "message": message,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_rule_record(
    runtime: PredictiveAnalyticsRuntime,
    rule_group: str,
    rule_name: str,
    rule_type: str,
    description: str,
    source_dataset: str,
    rule_config: Dict[str, Any],
) -> None:
    """
    Add rule catalog record.
    """

    runtime.rule_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "rule_group": rule_group,
            "rule_name": rule_name,
            "rule_type": rule_type,
            "description": description,
            "source_dataset": source_dataset,
            "rule_config_json": safe_string(rule_config),
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


###############################################################################
# Dataset IO
###############################################################################

def get_output_format(runtime: PredictiveAnalyticsRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("predictive_analytics", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def read_dataset(path: Path, file_format: Optional[str]) -> pd.DataFrame:
    """
    Read dataset from disk.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    resolved_format = file_format or path.suffix.replace(".", "").lower()

    if resolved_format == "parquet":
        return pd.read_parquet(path)

    if resolved_format == "csv":
        return pd.read_csv(path)

    if resolved_format == "json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported input file format: {resolved_format}")


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Ensure output path has configured suffix.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


def write_dataset(dataframe: pd.DataFrame, path: Path, file_format: str) -> None:
    """
    Write dataframe to disk.
    """

    ensure_directory(path.parent)

    if file_format == "parquet":
        dataframe.to_parquet(path, index=False)
        return

    if file_format == "csv":
        dataframe.to_csv(path, index=False)
        return

    if file_format == "json":
        dataframe.to_json(path, orient="records", indent=2)
        return

    raise ValueError(f"Unsupported output file format: {file_format}")


def load_input_datasets(
    runtime: PredictiveAnalyticsRuntime,
) -> Dict[str, pd.DataFrame]:
    """
    Load configured scoring and modeling metadata inputs.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Predictive Analytics input datasets")

    for dataset_name, dataset_config in inputs_config.items():
        raw_path = dataset_config.get("path")
        file_format = dataset_config.get("format")
        required = bool(dataset_config.get("required", True))

        if not raw_path:
            message = f"No path configured for input dataset: {dataset_name}"
            if required:
                raise ValueError(message)
            logger.warning("Skipping optional dataset with no path: %s", dataset_name)
            continue

        dataset_path = normalize_path(runtime.project_root, raw_path)

        try:
            dataframe = read_dataset(dataset_path, file_format)
            datasets[dataset_name] = dataframe

            logger.info(
                "Loaded input dataset: %s | Rows: %s | Columns: %s | Path: %s",
                dataset_name,
                len(dataframe),
                len(dataframe.columns),
                dataset_path,
            )

            add_dataset_record(
                runtime=runtime,
                dataset_name=dataset_name,
                dataset_type="input",
                status=STATUS_SUCCESS,
                path=str(dataset_path),
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                message="Predictive Analytics input dataset loaded successfully.",
            )

        except Exception as exc:
            message = f"Failed to load input dataset '{dataset_name}': {exc}"

            if required:
                add_audit_record(
                    runtime=runtime,
                    step_name=f"load_input_dataset:{dataset_name}",
                    status=STATUS_FAILED,
                    message=message,
                    output_path=str(dataset_path),
                )
                raise

            logger.warning("Optional input skipped: %s | Reason: %s", dataset_name, exc)

            add_audit_record(
                runtime=runtime,
                step_name=f"load_input_dataset:{dataset_name}",
                status=STATUS_SKIPPED,
                message=message,
                output_path=str(dataset_path),
            )

    logger.info("COMPLETE: Load Predictive Analytics input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: PredictiveAnalyticsRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> bool:
    """
    Validate dataframe is not empty.
    """

    if dataframe.empty:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="not_empty",
            status=STATUS_FAILED,
            message="Dataset is empty.",
            failed_count=1,
        )
        return False

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="not_empty",
        status=STATUS_SUCCESS,
        message="Dataset is not empty.",
        failed_count=0,
    )
    return True


def validate_required_columns(
    runtime: PredictiveAnalyticsRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_columns: Iterable[str],
) -> bool:
    """
    Validate required columns exist.
    """

    required = list(required_columns)
    missing = [column for column in required if column not in dataframe.columns]

    if missing:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="required_columns",
            status=STATUS_FAILED,
            message=f"Missing required columns: {missing}",
            failed_count=len(missing),
        )
        return False

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="required_columns",
        status=STATUS_SUCCESS,
        message="All required columns are present.",
        failed_count=0,
    )
    return True


def validate_member_key_not_null(
    runtime: PredictiveAnalyticsRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    member_key: str,
) -> bool:
    """
    Validate member key is present and non-null.
    """

    if member_key not in dataframe.columns:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="member_key_exists",
            status=STATUS_FAILED,
            message=f"Missing member key column: {member_key}",
            failed_count=1,
        )
        return False

    null_count = int(dataframe[member_key].isna().sum())

    if null_count > 0:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="member_key_not_null",
            status=STATUS_FAILED,
            message=f"Member key contains nulls: {member_key}",
            failed_count=null_count,
        )
        return False

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="member_key_not_null",
        status=STATUS_SUCCESS,
        message="Member key is not null.",
        failed_count=0,
    )
    return True


def validate_inputs(
    runtime: PredictiveAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Validate loaded inputs and model score registry column contracts.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    for dataset_name, dataframe in datasets.items():
        validate_not_empty(runtime, dataframe, dataset_name)

        if member_key in dataframe.columns:
            validate_member_key_not_null(runtime, dataframe, dataset_name, member_key)

    for model_key, model_config in runtime.config.get("model_score_registry", {}).items():
        if not bool(model_config.get("enabled", True)):
            continue

        source_dataset = model_config.get("source_dataset")

        if source_dataset not in datasets:
            add_validation_record(
                runtime=runtime,
                dataset_name=source_dataset,
                rule_name="model_score_dataset_exists",
                status=STATUS_FAILED,
                message=f"Model scoring dataset not loaded for model: {model_key}",
                failed_count=1,
            )
            continue

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

        validate_required_columns(
            runtime=runtime,
            dataframe=score_df,
            dataset_name=source_dataset,
            required_columns=required_columns,
        )

    add_audit_record(
        runtime=runtime,
        step_name="validate_inputs",
        status=STATUS_SUCCESS,
        message="Predictive Analytics input validation completed.",
    )


###############################################################################
# Unified Prediction Registry
###############################################################################

def build_unified_prediction_registry(
    runtime: PredictiveAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build unified long-format prediction registry.

    Output grain:
        One row per member per model.
    """

    logger = runtime.logger
    config = runtime.config.get("unified_prediction_registry", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Unified prediction registry disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    registry_rows: List[pd.DataFrame] = []

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
            description=f"Standardize scoring output for {model_name}",
            source_dataset=source_dataset,
            rule_config=model_config,
        )

        if source_dataset not in datasets:
            raise ValueError(f"Scoring source dataset missing: {source_dataset}")

        source_df = datasets[source_dataset]

        required_columns = [
            member_key,
            score_column,
            prediction_column,
            tier_column,
            "modeling_layer_run_id",
            "scored_at_utc",
        ]

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=required_columns,
        )

        standardized = source_df[
            [
                member_key,
                score_column,
                prediction_column,
                tier_column,
                "modeling_layer_run_id",
                "scored_at_utc",
            ]
        ].copy()

        standardized = standardized.rename(
            columns={
                score_column: "prediction_score",
                prediction_column: "prediction_flag",
                tier_column: "prediction_tier",
            }
        )

        standardized["model_key"] = model_key
        standardized["model_name"] = model_name
        standardized["analytics_layer_run_id"] = runtime.run_id
        standardized["analytics_domain"] = runtime.domain_name
        standardized["analytics_asset_name"] = "unified_prediction_registry"
        standardized["built_at_utc"] = utc_now().isoformat()

        registry_rows.append(standardized)

        logger.info(
            "Standardized scoring output: %s | Rows: %s",
            model_key,
            len(standardized),
        )

    if registry_rows:
        output_df = pd.concat(registry_rows, ignore_index=True)
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
    )

    logger.info("COMPLETE: Build unified prediction registry | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Member Prediction Summary
###############################################################################

def build_member_prediction_summary(
    runtime: PredictiveAnalyticsRuntime,
    unified_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build member-level prediction summary.

    Output grain:
        One row per member.
    """

    logger = runtime.logger
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

    top_model = (
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

    output_df = output_df.merge(top_model, on=member_key, how="left")

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "member_prediction_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="member_prediction_summary",
        rule_name="member_level_prediction_summary",
        rule_type="aggregation",
        description="Aggregates model scores to one row per member.",
        source_dataset="unified_prediction_registry",
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
    )

    logger.info("COMPLETE: Build member prediction summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# High Priority Member Registry
###############################################################################

def build_high_priority_member_registry(
    runtime: PredictiveAnalyticsRuntime,
    member_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build high-priority member registry using configured priority rules.
    """

    logger = runtime.logger
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

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "high_priority_member_registry"
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
    )

    logger.info("COMPLETE: Build high priority member registry | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Model Risk Distribution
###############################################################################

def build_model_risk_distribution(
    runtime: PredictiveAnalyticsRuntime,
    unified_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build model-level risk distribution.
    """

    logger = runtime.logger
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

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "model_risk_distribution"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="model_risk_distribution",
        rule_name="model_tier_distribution",
        rule_type="aggregation",
        description="Aggregates prediction registry by model and tier.",
        source_dataset="unified_prediction_registry",
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
    )

    logger.info("COMPLETE: Build model risk distribution | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Prediction Model Summary
###############################################################################

def build_prediction_model_summary(
    runtime: PredictiveAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
    unified_registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build prediction model summary by combining modeling registry and scoring stats.
    """

    logger = runtime.logger
    config = runtime.config.get("prediction_model_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Prediction model summary disabled")
        return pd.DataFrame()

    model_registry_name = config.get("source_dataset", "modeling_model_registry")

    if model_registry_name in datasets:
        model_registry = datasets[model_registry_name].copy()
    else:
        model_registry = pd.DataFrame()

    if unified_registry.empty:
        score_stats = pd.DataFrame()
    else:
        score_stats = unified_registry.groupby(["model_key", "model_name"]).agg(
            scored_member_count=("member_id", "nunique"),
            average_prediction_score=("prediction_score", "mean"),
            max_prediction_score=("prediction_score", "max"),
            positive_prediction_count=("prediction_flag", "sum"),
        ).reset_index()

    if not model_registry.empty and "model_key" in model_registry.columns:
        output_df = model_registry.merge(score_stats, on="model_key", how="left")
    else:
        output_df = score_stats.copy()

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "prediction_model_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="prediction_model_summary",
        rule_name="model_registry_with_score_stats",
        rule_type="metadata_enrichment",
        description="Combines model registry metadata with scoring distribution.",
        source_dataset=model_registry_name,
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
    )

    logger.info("COMPLETE: Build prediction model summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: PredictiveAnalyticsRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: PredictiveAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build output column dictionary.
    """

    rows: List[Dict[str, Any]] = []

    for dataset_name, dataframe in output_assets.items():
        for column in dataframe.columns:
            rows.append(
                {
                    "run_id": runtime.run_id,
                    "layer_name": runtime.layer_name,
                    "domain_name": runtime.domain_name,
                    "dataset_name": dataset_name,
                    "column_name": column,
                    "data_type": str(dataframe[column].dtype),
                    "non_null_count": int(dataframe[column].notna().sum()),
                    "null_count": int(dataframe[column].isna().sum()),
                    "row_count": int(len(dataframe)),
                    "event_timestamp_utc": utc_now().isoformat(),
                }
            )

    return pd.DataFrame(rows)


def build_rule_catalog(runtime: PredictiveAnalyticsRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: PredictiveAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build execution summary.
    """

    end_time = utc_now()
    duration_seconds = (end_time - runtime.start_time_utc).total_seconds()

    failed_validation_count = sum(
        1 for record in runtime.validation_records
        if record.get("status") == STATUS_FAILED
    )

    summary = {
        "run_id": runtime.run_id,
        "layer_name": runtime.layer_name,
        "domain_name": runtime.domain_name,
        "config_path": str(runtime.config_path),
        "start_time_utc": runtime.start_time_utc.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "output_asset_count": len(output_assets),
        "audit_record_count": len(runtime.audit_records),
        "validation_record_count": len(runtime.validation_records),
        "failed_validation_count": failed_validation_count,
        "dataset_record_count": len(runtime.dataset_records),
        "rule_record_count": len(runtime.rule_records),
        "status": STATUS_SUCCESS if failed_validation_count == 0 else STATUS_WARNING,
    }

    return pd.DataFrame([summary])


###############################################################################
# Output Writing
###############################################################################

def get_output_path(
    runtime: PredictiveAnalyticsRuntime,
    output_group: str,
    output_name: str,
) -> Path:
    """
    Resolve configured output path.
    """

    output_config = runtime.config.get("paths", {}).get(output_group, {})
    output_entry = output_config.get(output_name)

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        raw_path = f"data/analytics_platform/{output_group}/{output_name}"

    return normalize_path(runtime.project_root, raw_path)


def write_predictive_outputs(
    runtime: PredictiveAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Predictive Analytics outputs.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Predictive Analytics output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Predictive Analytics output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: PredictiveAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "predictive_analytics_dataset_inventory": build_dataset_inventory(runtime),
        "predictive_analytics_column_dictionary": build_column_dictionary(runtime, output_assets),
        "predictive_analytics_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "predictive_analytics_audit_records": pd.DataFrame(runtime.audit_records),
        "predictive_analytics_validation_results": pd.DataFrame(runtime.validation_records),
        "predictive_analytics_execution_summary": build_execution_summary(runtime, output_assets),
    }

    for output_name, dataframe in metadata_assets.items():
        output_path = get_output_path(runtime, "metadata_outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)
        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote metadata output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

    for output_name, dataframe in audit_assets.items():
        output_path = get_output_path(runtime, "audit_outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)
        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote audit output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )


###############################################################################
# Main Orchestration
###############################################################################

def build_predictive_analytics_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> BuildResult:
    """
    Build complete Predictive Analytics layer.
    """

    runtime: Optional[PredictiveAnalyticsRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)
        logger.info("Architectural check: Predictive Analytics consumes scoring outputs only.")

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

        unified_prediction_registry = build_unified_prediction_registry(runtime, datasets)
        member_prediction_summary = build_member_prediction_summary(
            runtime,
            unified_prediction_registry,
        )
        high_priority_member_registry = build_high_priority_member_registry(
            runtime,
            member_prediction_summary,
        )
        model_risk_distribution = build_model_risk_distribution(
            runtime,
            unified_prediction_registry,
        )
        prediction_model_summary = build_prediction_model_summary(
            runtime,
            datasets,
            unified_prediction_registry,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "unified_prediction_registry": unified_prediction_registry,
            "member_prediction_summary": member_prediction_summary,
            "high_priority_member_registry": high_priority_member_registry,
            "model_risk_distribution": model_risk_distribution,
            "prediction_model_summary": prediction_model_summary,
        }

        write_predictive_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_predictive_analytics_layer",
            status=STATUS_SUCCESS,
            message="Predictive Analytics completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Predictive Analytics completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="predictive_analytics",
            status=STATUS_SUCCESS,
            message="Predictive Analytics completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Predictive Analytics failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_predictive_analytics_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="predictive_analytics",
            status=STATUS_FAILED,
            message=str(exc),
        )


###############################################################################
# CLI Entry Point
###############################################################################

def main() -> None:
    """
    Command-line entry point.

    Run:
        python -m src.analytics_platform.predictive_analytics.build_predictive_analytics_layer
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