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

DEFAULT_CONFIG_PATH = "config/analytics_platform/value_based_care.yaml"

DEFAULT_LAYER_NAME = "Layer 2H - Value-Based Care"
DEFAULT_DOMAIN_NAME = "Value-Based Care"
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
class ValueBasedCareRuntime:
    """
    Runtime context for one Value-Based Care execution.
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
    Standard build result returned by Value-Based Care.
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
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


def generate_run_id() -> str:
    """
    Generate timestamp-based run identifier.
    """

    return utc_now().strftime("%Y%m%d_%H%M%S")


def normalize_path(project_root: Path, raw_path: str | Path) -> Path:
    """
    Resolve relative path against project root.
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
    Safely convert value to string.
    """

    if value is None:
        return ""

    return str(value)


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide numeric values.
    """

    if denominator is None or denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


###############################################################################
# Logging
###############################################################################

def configure_logging(
    project_root: Path,
    config: Dict[str, Any],
    run_id: str,
) -> logging.Logger:
    """
    Configure Value-Based Care logging.
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
        "value_based_care.log",
    )

    logger = logging.getLogger("medfabric.analytics_platform.value_based_care")
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
    logger.info("MedFabric Value-Based Care started")
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
    Validate required Value-Based Care configuration sections.
    """

    errors: List[str] = []

    required_sections = [
        "value_based_care",
        "logging",
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
        config.get("value_based_care", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Value-Based Care configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(
    config_path_raw: str = DEFAULT_CONFIG_PATH,
) -> ValueBasedCareRuntime:
    """
    Initialize Value-Based Care runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    layer_config = config.get("value_based_care", {})

    layer_name = layer_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = layer_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = ValueBasedCareRuntime(
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
        message="Value-Based Care runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: ValueBasedCareRuntime,
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
    runtime: ValueBasedCareRuntime,
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
    runtime: ValueBasedCareRuntime,
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
    runtime: ValueBasedCareRuntime,
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

def get_output_format(runtime: ValueBasedCareRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("value_based_care", {})
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
    runtime: ValueBasedCareRuntime,
) -> Dict[str, pd.DataFrame]:
    """
    Load configured Value-Based Care input datasets.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Value-Based Care input datasets")

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
                message="Value-Based Care input dataset loaded successfully.",
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

    logger.info("COMPLETE: Load Value-Based Care input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: ValueBasedCareRuntime,
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
    runtime: ValueBasedCareRuntime,
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


def validate_inputs(
    runtime: ValueBasedCareRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Validate loaded inputs.
    """

    for dataset_name, dataframe in datasets.items():
        validate_not_empty(runtime, dataframe, dataset_name)

    add_audit_record(
        runtime=runtime,
        step_name="validate_inputs",
        status=STATUS_SUCCESS,
        message="Value-Based Care input validation completed.",
    )


###############################################################################
# Rule Helpers
###############################################################################

def apply_operator(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Apply configured comparison operator.
    """

    if operator == "equals":
        return series == value

    if operator == "not_equals":
        return series != value

    if operator == "greater_than":
        return series > value

    if operator == "greater_than_or_equal":
        return series >= value

    if operator == "less_than":
        return series < value

    if operator == "less_than_or_equal":
        return series <= value

    if operator == "in":
        return series.isin(value)

    if operator == "not_in":
        return ~series.isin(value)

    if operator == "is_null":
        return series.isna()

    if operator == "is_not_null":
        return series.notna()

    raise ValueError(f"Unsupported operator: {operator}")


###############################################################################
# Value-Based Contract Summary
###############################################################################

def build_value_based_contract_summary(
    runtime: ValueBasedCareRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build value-based contract summary.

    Output grain:
        One row per synthetic contract.
    """

    logger = runtime.logger
    config = runtime.config.get("value_based_contract_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Value-based contract summary disabled")
        return pd.DataFrame()

    source_datasets = config.get("source_datasets", {})
    pmpm_dataset_name = source_datasets.get("pmpm_dataset", "pmpm_summary")
    cost_dataset_name = source_datasets.get("cost_dataset", "cost_summary")
    provider_dataset_name = source_datasets.get("provider_dataset", "provider_performance_summary")

    if pmpm_dataset_name not in datasets:
        raise ValueError(f"Missing PMPM dataset: {pmpm_dataset_name}")

    if cost_dataset_name not in datasets:
        raise ValueError(f"Missing cost dataset: {cost_dataset_name}")

    if provider_dataset_name not in datasets:
        raise ValueError(f"Missing provider dataset: {provider_dataset_name}")

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

    validate_required_columns(
        runtime=runtime,
        dataframe=pmpm_df,
        dataset_name=pmpm_dataset_name,
        required_columns=required_pmpm_columns,
    )

    pmpm_row = pmpm_df.iloc[0]

    member_month_count = float(pmpm_row[financial_columns.get("member_month_count")])
    total_paid_amount = float(pmpm_row[financial_columns.get("total_paid_amount")])
    total_allowed_amount = float(pmpm_row[financial_columns.get("total_allowed_amount")])
    pmpm_paid_amount = float(pmpm_row[financial_columns.get("pmpm_paid_amount")])
    pmpm_allowed_amount = float(pmpm_row[financial_columns.get("pmpm_allowed_amount")])

    benchmark_pmpm_paid_amount = float(benchmark.get("benchmark_pmpm_paid_amount", 0.0))
    benchmark_pmpm_allowed_amount = float(benchmark.get("benchmark_pmpm_allowed_amount", 0.0))
    minimum_savings_rate = float(benchmark.get("minimum_savings_rate", 0.0))
    provider_share_rate = float(benchmark.get("provider_share_rate", 0.0))

    benchmark_total_paid_amount = benchmark_pmpm_paid_amount * member_month_count
    gross_savings_amount = benchmark_total_paid_amount - total_paid_amount
    gross_savings_rate = safe_divide(gross_savings_amount, benchmark_total_paid_amount)

    minimum_savings_met = gross_savings_rate >= minimum_savings_rate
    shared_savings_amount = gross_savings_amount * provider_share_rate if minimum_savings_met else 0.0

    output_df = pd.DataFrame(
        [
            {
                "value_based_contract_name": config.get("contract_name"),
                "contract_type": config.get("contract_type"),
                "payment_model": config.get("payment_model"),
                "measurement_period": config.get("measurement_period"),
                "provider_count": int(provider_df["provider_id"].nunique()) if "provider_id" in provider_df.columns else len(provider_df),
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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "value_based_contract_summary",
                "built_at_utc": utc_now().isoformat(),
            }
        ]
    )

    add_rule_record(
        runtime=runtime,
        rule_group="value_based_contract_summary",
        rule_name="synthetic_shared_savings_contract",
        rule_type="contract_financial_summary",
        description="Builds synthetic value-based contract summary using PMPM benchmark comparison.",
        source_dataset=pmpm_dataset_name,
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
    )

    logger.info("COMPLETE: Build value-based contract summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Provider Incentive Summary
###############################################################################

def build_provider_incentive_summary(
    runtime: ValueBasedCareRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build provider incentive summary.

    Output grain:
        One row per provider per incentive rule earned.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_incentive_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider incentive summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "provider_performance_summary")
    provider_key = config.get("provider_key", "provider_id")

    if source_dataset not in datasets:
        raise ValueError(f"Provider incentive source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    output_frames: List[pd.DataFrame] = []

    validate_required_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=[provider_key],
    )

    logger.info("START: Build provider incentive summary")

    for rule_key, rule_config in config.get("incentive_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")
        incentive_amount = float(rule_config.get("incentive_amount", 0.0))

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[provider_key, rule_column],
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        incentive_df = source_df.loc[mask].copy()
        incentive_df["incentive_rule_key"] = rule_key
        incentive_df["incentive_name"] = rule_config.get("incentive_name", rule_key)
        incentive_df["incentive_amount"] = incentive_amount
        incentive_df["incentive_description"] = rule_config.get("description", "")
        incentive_df["analytics_layer_run_id"] = runtime.run_id
        incentive_df["analytics_domain"] = runtime.domain_name
        incentive_df["analytics_asset_name"] = "provider_incentive_summary"
        incentive_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="provider_incentive_summary",
            rule_name=rule_key,
            rule_type="provider_incentive_assignment",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
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
        output_df = pd.DataFrame()

    add_dataset_record(
        runtime=runtime,
        dataset_name="provider_incentive_summary",
        dataset_type="value_based_care_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Provider incentive summary built successfully.",
    )

    logger.info("COMPLETE: Build provider incentive summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Shared Savings Summary
###############################################################################

def build_shared_savings_summary(
    runtime: ValueBasedCareRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build shared savings summary.
    """

    logger = runtime.logger
    config = runtime.config.get("shared_savings_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Shared savings summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "pmpm_summary")

    if source_dataset not in datasets:
        raise ValueError(f"Shared savings source dataset missing: {source_dataset}")

    pmpm_df = datasets[source_dataset]
    validate_required_columns(
        runtime=runtime,
        dataframe=pmpm_df,
        dataset_name=source_dataset,
        required_columns=[
            "member_month_count",
            "total_paid_amount",
            "pmpm_paid_amount",
        ],
    )

    row = pmpm_df.iloc[0]
    benchmark = config.get("benchmark", {})

    benchmark_pmpm_paid_amount = float(benchmark.get("benchmark_pmpm_paid_amount", 0.0))
    minimum_savings_rate = float(benchmark.get("minimum_savings_rate", 0.0))
    provider_share_rate = float(benchmark.get("provider_share_rate", 0.0))

    member_month_count = float(row["member_month_count"])
    actual_total_paid_amount = float(row["total_paid_amount"])
    actual_pmpm_paid_amount = float(row["pmpm_paid_amount"])

    benchmark_total_paid_amount = benchmark_pmpm_paid_amount * member_month_count
    gross_savings_amount = benchmark_total_paid_amount - actual_total_paid_amount
    gross_savings_rate = safe_divide(gross_savings_amount, benchmark_total_paid_amount)
    minimum_savings_met = gross_savings_rate >= minimum_savings_rate
    provider_shared_savings_amount = (
        gross_savings_amount * provider_share_rate if minimum_savings_met else 0.0
    )
    payer_retained_savings_amount = (
        gross_savings_amount - provider_shared_savings_amount if minimum_savings_met else 0.0
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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "shared_savings_summary",
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
    )

    logger.info("COMPLETE: Build shared savings summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Risk Adjustment Summary
###############################################################################

def build_risk_adjustment_summary(
    runtime: ValueBasedCareRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build risk adjustment summary from member prediction summary.
    """

    logger = runtime.logger
    config = runtime.config.get("risk_adjustment_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Risk adjustment summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "member_prediction_summary")

    if source_dataset not in datasets:
        raise ValueError(f"Risk adjustment source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    risk_columns = config.get("risk_columns", {})
    score_column = risk_columns.get("max_prediction_score", "max_prediction_score")

    validate_required_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=[score_column],
    )

    output_rows: List[Dict[str, Any]] = []

    logger.info("START: Build risk adjustment summary")

    for band_key, band_config in config.get("risk_bands", {}).items():
        min_score = float(band_config.get("min_score", 0.0))
        max_score = float(band_config.get("max_score", 1.0))
        risk_weight = float(band_config.get("risk_weight", 1.0))
        label = band_config.get("label", band_key)

        mask = (source_df[score_column] >= min_score) & (source_df[score_column] <= max_score)
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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "risk_adjustment_summary",
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
    )

    logger.info("COMPLETE: Build risk adjustment summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Bundle Opportunity Summary
###############################################################################

def build_bundle_opportunity_summary(
    runtime: ValueBasedCareRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build bundle opportunity summary from cost summary.
    """

    logger = runtime.logger
    config = runtime.config.get("bundle_opportunity_summary", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Bundle opportunity summary disabled")
        return pd.DataFrame()

    source_dataset = config.get("source_dataset", "cost_summary")

    if source_dataset not in datasets:
        raise ValueError(f"Bundle opportunity source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    group_by = config.get("group_by", [])

    validate_required_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset,
        required_columns=group_by,
    )

    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build bundle opportunity summary")

    for rule_key, rule_config in config.get("opportunity_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=group_by + [rule_column],
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        opportunity_df = source_df.loc[mask].copy()
        opportunity_df["opportunity_rule_key"] = rule_key
        opportunity_df["opportunity_name"] = rule_config.get("opportunity_name", rule_key)
        opportunity_df["opportunity_priority_rank"] = rule_config.get("opportunity_priority_rank")
        opportunity_df["opportunity_description"] = rule_config.get("description", "")
        opportunity_df["analytics_layer_run_id"] = runtime.run_id
        opportunity_df["analytics_domain"] = runtime.domain_name
        opportunity_df["analytics_asset_name"] = "bundle_opportunity_summary"
        opportunity_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="bundle_opportunity_summary",
            rule_name=rule_key,
            rule_type="bundle_opportunity_selection",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
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
    )

    logger.info("COMPLETE: Build bundle opportunity summary | Rows: %s", len(output_df))

    return output_df


###############################################################################
# VBC Executive Scorecard
###############################################################################

def build_vbc_executive_scorecard(
    runtime: ValueBasedCareRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build executive scorecard from Value-Based Care output assets.
    """

    logger = runtime.logger
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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "vbc_executive_scorecard",
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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "vbc_executive_scorecard",
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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "vbc_executive_scorecard",
                "built_at_utc": utc_now().isoformat(),
            }
        )

    output_df = pd.DataFrame(rows)

    add_rule_record(
        runtime=runtime,
        rule_group="vbc_executive_scorecard",
        rule_name="executive_scorecard_assembly",
        rule_type="scorecard_summary",
        description="Builds executive scorecard from Value-Based Care output assets.",
        source_dataset="value_based_care_outputs",
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
    )

    logger.info("COMPLETE: Build VBC executive scorecard | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: ValueBasedCareRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: ValueBasedCareRuntime,
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


def build_rule_catalog(runtime: ValueBasedCareRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: ValueBasedCareRuntime,
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
    runtime: ValueBasedCareRuntime,
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


def write_value_based_care_outputs(
    runtime: ValueBasedCareRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Value-Based Care outputs.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Value-Based Care output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Value-Based Care output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: ValueBasedCareRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "value_based_care_dataset_inventory": build_dataset_inventory(runtime),
        "value_based_care_column_dictionary": build_column_dictionary(runtime, output_assets),
        "value_based_care_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "value_based_care_audit_records": pd.DataFrame(runtime.audit_records),
        "value_based_care_validation_results": pd.DataFrame(runtime.validation_records),
        "value_based_care_execution_summary": build_execution_summary(runtime, output_assets),
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

def build_value_based_care_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> BuildResult:
    """
    Build complete Value-Based Care layer.
    """

    runtime: Optional[ValueBasedCareRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

        value_based_contract_summary = build_value_based_contract_summary(runtime, datasets)
        provider_incentive_summary = build_provider_incentive_summary(runtime, datasets)
        shared_savings_summary = build_shared_savings_summary(runtime, datasets)
        risk_adjustment_summary = build_risk_adjustment_summary(runtime, datasets)
        bundle_opportunity_summary = build_bundle_opportunity_summary(runtime, datasets)

        output_assets: Dict[str, pd.DataFrame] = {
            "value_based_contract_summary": value_based_contract_summary,
            "provider_incentive_summary": provider_incentive_summary,
            "shared_savings_summary": shared_savings_summary,
            "risk_adjustment_summary": risk_adjustment_summary,
            "bundle_opportunity_summary": bundle_opportunity_summary,
        }

        vbc_executive_scorecard = build_vbc_executive_scorecard(runtime, output_assets)
        output_assets["vbc_executive_scorecard"] = vbc_executive_scorecard

        write_value_based_care_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_value_based_care_layer",
            status=STATUS_SUCCESS,
            message="Value-Based Care completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Value-Based Care completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="value_based_care",
            status=STATUS_SUCCESS,
            message="Value-Based Care completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Value-Based Care failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_value_based_care_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="value_based_care",
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
        python -m src.analytics_platform.value_based_care.build_value_based_care_layer
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