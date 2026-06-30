###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/provider_analytics/build_provider_analytics_layer.py
#
# Layer:
#     Layer 2F - Provider Analytics
#
# Purpose:
#     Builds Provider Analytics outputs from configured provider-level datasets.
#
# Inputs:
#     config/analytics_platform/provider_analytics.yaml
#
# Outputs:
#     data/analytics_platform/provider_analytics/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.provider_analytics.build_provider_analytics_layer
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

DEFAULT_CONFIG_PATH = "config/analytics_platform/provider_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2F - Provider Analytics"
DEFAULT_DOMAIN_NAME = "Provider Analytics"
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
class ProviderAnalyticsRuntime:
    """
    Runtime context for one Provider Analytics execution.
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
    Standard build result returned by Provider Analytics.
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
    Generate timestamp-based run identifier.
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
    Convert value to safe string for metadata.
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
    Configure Provider Analytics logging.
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
        "provider_analytics.log",
    )

    logger = logging.getLogger("medfabric.analytics_platform.provider_analytics")
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
    logger.info("MedFabric Provider Analytics started")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Log file: %s", log_file_path)

    return logger


###############################################################################
# Configuration
###############################################################################

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
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
    Validate required Provider Analytics configuration sections.
    """

    errors: List[str] = []

    required_sections = [
        "provider_analytics",
        "logging",
        "paths",
        "join_keys",
        "provider_framework",
        "provider_performance_summary",
        "provider_network_summary",
        "provider_specialty_summary",
        "provider_cost_summary",
        "provider_utilization_summary",
        "high_performing_providers",
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
        config.get("provider_analytics", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Provider Analytics configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(
    config_path_raw: str = DEFAULT_CONFIG_PATH,
) -> ProviderAnalyticsRuntime:
    """
    Initialize Provider Analytics runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    provider_config = config.get("provider_analytics", {})

    layer_name = provider_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = provider_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = ProviderAnalyticsRuntime(
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
        message="Provider Analytics runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: ProviderAnalyticsRuntime,
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
    runtime: ProviderAnalyticsRuntime,
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
    runtime: ProviderAnalyticsRuntime,
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
    runtime: ProviderAnalyticsRuntime,
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

def get_output_format(runtime: ProviderAnalyticsRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("provider_analytics", {})
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
    runtime: ProviderAnalyticsRuntime,
) -> Dict[str, pd.DataFrame]:
    """
    Load configured Provider Analytics input datasets.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Provider Analytics input datasets")

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
                message="Provider Analytics input dataset loaded successfully.",
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

    logger.info("COMPLETE: Load Provider Analytics input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: ProviderAnalyticsRuntime,
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
    runtime: ProviderAnalyticsRuntime,
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


def validate_provider_key_not_null(
    runtime: ProviderAnalyticsRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    provider_key: str,
) -> bool:
    """
    Validate provider key is present and not null.
    """

    if provider_key not in dataframe.columns:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="provider_key_exists",
            status=STATUS_FAILED,
            message=f"Missing provider key column: {provider_key}",
            failed_count=1,
        )
        return False

    null_count = int(dataframe[provider_key].isna().sum())

    if null_count > 0:
        add_validation_record(
            runtime=runtime,
            dataset_name=dataset_name,
            rule_name="provider_key_not_null",
            status=STATUS_FAILED,
            message=f"Provider key contains nulls: {provider_key}",
            failed_count=null_count,
        )
        return False

    add_validation_record(
        runtime=runtime,
        dataset_name=dataset_name,
        rule_name="provider_key_not_null",
        status=STATUS_SUCCESS,
        message="Provider key is not null.",
        failed_count=0,
    )
    return True


def validate_inputs(
    runtime: ProviderAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Run basic validation for Provider Analytics input datasets.
    """

    provider_key = runtime.config.get("join_keys", {}).get("provider_key", "provider_id")

    for dataset_name, dataframe in datasets.items():
        validate_not_empty(runtime, dataframe, dataset_name)

        if provider_key in dataframe.columns:
            validate_provider_key_not_null(runtime, dataframe, dataset_name, provider_key)

    add_audit_record(
        runtime=runtime,
        step_name="validate_inputs",
        status=STATUS_SUCCESS,
        message="Provider Analytics input validation completed.",
    )


###############################################################################
# Calculation Helpers
###############################################################################

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two numeric series.
    """

    denominator_clean = denominator.replace({0: pd.NA})
    result = numerator / denominator_clean
    return result.fillna(0)


def calculate_group_metric(
    dataframe: pd.DataFrame,
    group_by: List[str],
    metric_name: str,
    metric_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate configured group metric.
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

    if calculation_type == "count_distinct":
        return (
            dataframe.groupby(group_by)[column]
            .nunique(dropna=True)
            .reset_index(name=metric_name)
        )

    if calculation_type == "sum":
        return (
            dataframe.groupby(group_by)[column]
            .sum()
            .reset_index(name=metric_name)
        )

    if calculation_type == "mean":
        return (
            dataframe.groupby(group_by)[column]
            .mean()
            .reset_index(name=metric_name)
        )

    if calculation_type == "sum_boolean":
        return (
            dataframe.groupby(group_by)[column]
            .apply(lambda series: series.fillna(False).astype(bool).sum())
            .reset_index(name=metric_name)
        )

    if calculation_type == "count_rows":
        return (
            dataframe.groupby(group_by)
            .size()
            .reset_index(name=metric_name)
        )

    raise ValueError(f"Unsupported calculation_type: {calculation_type}")


def build_group_summary(
    runtime: ProviderAnalyticsRuntime,
    dataframe: pd.DataFrame,
    config: Dict[str, Any],
    output_name: str,
) -> pd.DataFrame:
    """
    Build grouped provider summary from configuration.
    """

    source_dataset = config.get("source_dataset", "")
    group_by = config.get("group_by", [])
    metrics = config.get("metrics", {})

    if not group_by:
        raise ValueError(f"group_by is required for output: {output_name}")

    for group_column in group_by:
        if group_column not in dataframe.columns:
            raise ValueError(f"Missing group_by column for {output_name}: {group_column}")

    output_df = dataframe[group_by].drop_duplicates().copy()

    for metric_name, metric_config in metrics.items():
        add_rule_record(
            runtime=runtime,
            rule_group=output_name,
            rule_name=metric_name,
            rule_type=metric_config.get("calculation_type", ""),
            description=f"Provider grouped metric: {metric_name}",
            source_dataset=source_dataset,
            rule_config=metric_config,
        )

        metric_df = calculate_group_metric(
            dataframe=dataframe,
            group_by=group_by,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        output_df = output_df.merge(metric_df, on=group_by, how="left")

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = output_name
    output_df["built_at_utc"] = utc_now().isoformat()

    return output_df


###############################################################################
# Provider Performance Summary
###############################################################################

def build_provider_performance_summary(
    runtime: ProviderAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build provider-level performance summary.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_performance_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider performance summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance")
    provider_key = config.get("provider_key", "provider_id")

    if source_dataset not in datasets:
        raise ValueError(f"Provider performance source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]

    required_columns = config.get("required_columns", [])
    validate_required_columns(runtime, source_df, source_dataset, required_columns)

    output_columns = [column for column in config.get("output_columns", []) if column in source_df.columns]
    output_df = source_df[output_columns].drop_duplicates(subset=[provider_key]).copy()

    for metric_name, metric_config in config.get("calculated_metrics", {}).items():
        numerator = metric_config.get("numerator")
        denominator = metric_config.get("denominator")

        validate_required_columns(
            runtime=runtime,
            dataframe=output_df,
            dataset_name=source_dataset,
            required_columns=[numerator, denominator],
        )

        output_df[metric_name] = safe_divide(output_df[numerator], output_df[denominator])

        add_rule_record(
            runtime=runtime,
            rule_group="provider_performance_summary",
            rule_name=metric_name,
            rule_type="ratio",
            description=metric_config.get("description", ""),
            source_dataset=source_dataset,
            rule_config=metric_config,
        )

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "provider_performance_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_performance_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider performance summary built successfully.",
    )

    logger.info("COMPLETE: Build provider performance summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Network and Specialty Summaries
###############################################################################

def build_provider_network_summary(
    runtime: ProviderAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build provider network summary.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_network_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider network summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance")
    if source_dataset not in datasets:
        raise ValueError(f"Provider network source dataset missing: {source_dataset}")

    output_df = build_group_summary(
        runtime=runtime,
        dataframe=datasets[source_dataset],
        config=config,
        output_name="provider_network_summary",
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_network_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider network summary built successfully.",
    )

    logger.info("COMPLETE: Build provider network summary | Rows: %s", len(output_df))

    return output_df


def build_provider_specialty_summary(
    runtime: ProviderAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build provider specialty summary.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_specialty_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider specialty summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance")
    if source_dataset not in datasets:
        raise ValueError(f"Provider specialty source dataset missing: {source_dataset}")

    output_df = build_group_summary(
        runtime=runtime,
        dataframe=datasets[source_dataset],
        config=config,
        output_name="provider_specialty_summary",
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_specialty_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider specialty summary built successfully.",
    )

    logger.info("COMPLETE: Build provider specialty summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Cost and Utilization Summaries
###############################################################################

def build_provider_cost_summary(
    runtime: ProviderAnalyticsRuntime,
    provider_performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build provider cost summary.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_cost_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider cost summary disabled")
        return pd.DataFrame()

    required_columns = config.get("required_columns", [])
    validate_required_columns(
        runtime=runtime,
        dataframe=provider_performance_summary,
        dataset_name="provider_performance_summary",
        required_columns=required_columns,
    )

    output_df = provider_performance_summary[required_columns].copy()

    for metric_name, metric_config in config.get("cost_metrics", {}).items():
        numerator = metric_config.get("numerator")
        denominator = metric_config.get("denominator")

        output_df[metric_name] = safe_divide(output_df[numerator], output_df[denominator])

        add_rule_record(
            runtime=runtime,
            rule_group="provider_cost_summary",
            rule_name=metric_name,
            rule_type="ratio",
            description=f"Provider cost metric: {metric_name}",
            source_dataset="provider_performance_summary",
            rule_config=metric_config,
        )

    ranking = config.get("ranking", {})
    rank_column = ranking.get("rank_column", "total_paid_amount")
    rank_method = ranking.get("rank_method", "descending")
    output_rank_column = ranking.get("output_rank_column", "provider_cost_rank")

    ascending = rank_method != "descending"
    output_df[output_rank_column] = output_df[rank_column].rank(
        method="dense",
        ascending=ascending,
    ).astype(int)

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "provider_cost_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_cost_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider cost summary built successfully.",
    )

    logger.info("COMPLETE: Build provider cost summary | Rows: %s", len(output_df))

    return output_df


def build_provider_utilization_summary(
    runtime: ProviderAnalyticsRuntime,
    provider_performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build provider utilization summary.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_utilization_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider utilization summary disabled")
        return pd.DataFrame()

    required_columns = config.get("required_columns", [])
    validate_required_columns(
        runtime=runtime,
        dataframe=provider_performance_summary,
        dataset_name="provider_performance_summary",
        required_columns=required_columns,
    )

    output_df = provider_performance_summary[required_columns].copy()

    for metric_name, metric_config in config.get("utilization_metrics", {}).items():
        numerator = metric_config.get("numerator")
        denominator = metric_config.get("denominator")

        output_df[metric_name] = safe_divide(output_df[numerator], output_df[denominator])

        add_rule_record(
            runtime=runtime,
            rule_group="provider_utilization_summary",
            rule_name=metric_name,
            rule_type="ratio",
            description=f"Provider utilization metric: {metric_name}",
            source_dataset="provider_performance_summary",
            rule_config=metric_config,
        )

    ranking = config.get("ranking", {})
    rank_column = ranking.get("rank_column", "encounter_count")
    rank_method = ranking.get("rank_method", "descending")
    output_rank_column = ranking.get("output_rank_column", "provider_utilization_rank")

    ascending = rank_method != "descending"
    output_df[output_rank_column] = output_df[rank_column].rank(
        method="dense",
        ascending=ascending,
    ).astype(int)

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "provider_utilization_summary"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_utilization_summary",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider utilization summary built successfully.",
    )

    logger.info("COMPLETE: Build provider utilization summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# High Performing Providers
###############################################################################

def build_high_performing_providers(
    runtime: ProviderAnalyticsRuntime,
    provider_performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build high-performing provider registry using configured rules.
    """

    logger = runtime.logger
    config = runtime.config.get("high_performing_providers", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: High performing providers disabled")
        return pd.DataFrame()

    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build high performing providers")

    for rule_key, rule_config in config.get("performance_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            continue

        label = rule_config.get("label", rule_key)
        priority_rank = rule_config.get("priority_rank")
        description = rule_config.get("description", "")

        mask = pd.Series(True, index=provider_performance_summary.index)

        if "min_encounter_count" in rule_config:
            mask = mask & (
                provider_performance_summary["encounter_count"]
                >= rule_config.get("min_encounter_count")
            )

        if "max_paid_per_encounter" in rule_config:
            if "paid_per_encounter" not in provider_performance_summary.columns:
                raise ValueError("paid_per_encounter required for high performing provider rule.")
            mask = mask & (
                provider_performance_summary["paid_per_encounter"]
                <= rule_config.get("max_paid_per_encounter")
            )

        if "attribution_eligible_required" in rule_config:
            required_value = bool(rule_config.get("attribution_eligible_required"))
            mask = mask & (
                provider_performance_summary["attribution_eligible"].fillna(False).astype(bool)
                == required_value
            )

        selected = provider_performance_summary.loc[mask].copy()
        selected["performance_rule_key"] = rule_key
        selected["performance_label"] = label
        selected["performance_priority_rank"] = priority_rank
        selected["performance_description"] = description

        add_rule_record(
            runtime=runtime,
            rule_group="high_performing_providers",
            rule_name=rule_key,
            rule_type="provider_selection",
            description=description,
            source_dataset="provider_performance_summary",
            rule_config=rule_config,
        )

        output_frames.append(selected)

        logger.info(
            "Applied high-performing provider rule: %s | Providers: %s",
            rule_key,
            len(selected),
        )

    if output_frames:
        output_df = pd.concat(output_frames, ignore_index=True)
    else:
        output_df = pd.DataFrame()

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "high_performing_providers"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="high_performing_providers",
        dataset_type="provider_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="High performing providers built successfully.",
    )

    logger.info("COMPLETE: Build high performing providers | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: ProviderAnalyticsRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: ProviderAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build column dictionary.
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


def build_rule_catalog(runtime: ProviderAnalyticsRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: ProviderAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one-row execution summary.
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
    runtime: ProviderAnalyticsRuntime,
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


def write_provider_outputs(
    runtime: ProviderAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Provider Analytics outputs.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Provider Analytics output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Provider Analytics output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: ProviderAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "provider_analytics_dataset_inventory": build_dataset_inventory(runtime),
        "provider_analytics_column_dictionary": build_column_dictionary(runtime, output_assets),
        "provider_analytics_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "provider_analytics_audit_records": pd.DataFrame(runtime.audit_records),
        "provider_analytics_validation_results": pd.DataFrame(runtime.validation_records),
        "provider_analytics_execution_summary": build_execution_summary(runtime, output_assets),
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

def build_provider_analytics_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> BuildResult:
    """
    Build complete Provider Analytics layer.
    """

    runtime: Optional[ProviderAnalyticsRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

        provider_performance_summary = build_provider_performance_summary(runtime, datasets)
        provider_network_summary = build_provider_network_summary(runtime, datasets)
        provider_specialty_summary = build_provider_specialty_summary(runtime, datasets)
        provider_cost_summary = build_provider_cost_summary(runtime, provider_performance_summary)
        provider_utilization_summary = build_provider_utilization_summary(
            runtime,
            provider_performance_summary,
        )
        high_performing_providers = build_high_performing_providers(
            runtime,
            provider_performance_summary,
        )

        output_assets: Dict[str, pd.DataFrame] = {
            "provider_performance_summary": provider_performance_summary,
            "provider_network_summary": provider_network_summary,
            "provider_specialty_summary": provider_specialty_summary,
            "provider_cost_summary": provider_cost_summary,
            "provider_utilization_summary": provider_utilization_summary,
            "high_performing_providers": high_performing_providers,
        }

        write_provider_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_provider_analytics_layer",
            status=STATUS_SUCCESS,
            message="Provider Analytics completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Provider Analytics completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="provider_analytics",
            status=STATUS_SUCCESS,
            message="Provider Analytics completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Provider Analytics failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_provider_analytics_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="provider_analytics",
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
        python -m src.analytics_platform.provider_analytics.build_provider_analytics_layer
    """

    config_path = os.environ.get(
        "MEDFABRIC_PROVIDER_ANALYTICS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_provider_analytics_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Provider Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()