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

DEFAULT_CONFIG_PATH = "config/analytics_platform/population_health.yaml"

DEFAULT_LAYER_NAME = "Layer 2A - Population Health Analytics"

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
class PopulationHealthRuntime:
    """
    Runtime context for one Population Health execution.
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
    Standard build result returned by the orchestrator.
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
    Return current UTC timestamp.
    """

    return datetime.now(timezone.utc)


def generate_run_id() -> str:
    """
    Generate timestamp-based run identifier.
    """

    return utc_now().strftime("%Y%m%d_%H%M%S")


def normalize_path(project_root: Path, raw_path: str | Path) -> Path:
    """
    Resolve configured paths relative to project root.
    """

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_directory(path: Path) -> None:
    """
    Create a directory if missing.
    """

    path.mkdir(parents=True, exist_ok=True)


def safe_string(value: Any) -> str:
    """
    Convert a value to string safely for metadata output.
    """

    if value is None:
        return ""

    return str(value)


###############################################################################
# Logging
###############################################################################

def configure_logging(project_root: Path, config: Dict[str, Any], run_id: str) -> logging.Logger:
    """
    Configure module logging.
    """

    logging_config = config.get("logging", {})

    log_level_name = logging_config.get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)

    log_dir_raw = logging_config.get("module_log_dir", "logs/modules")
    log_dir = normalize_path(project_root, log_dir_raw)
    ensure_directory(log_dir)

    log_file_name = logging_config.get("log_file_name", "population_health.log")
    log_file_path = log_dir / log_file_name

    logger = logging.getLogger("medfabric.analytics_platform.population_health")
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
    logger.info("MedFabric Population Health Analytics started")
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
    Validate required Population Health configuration sections.
    """

    required_sections = [
        "population_health",
        "paths",
        "join_keys",
        "population_cohorts",
        "risk_stratification",
        "member_segmentation",
        "provider_attribution_analytics",
    ]

    errors: List[str] = []

    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")

    paths = config.get("paths", {})

    if not isinstance(paths, dict):
        errors.append("Configuration section 'paths' must be a dictionary.")
    else:
        if "inputs" not in paths:
            errors.append("Missing required configuration section: paths.inputs")
        if "outputs" not in paths:
            errors.append("Missing required configuration section: paths.outputs")

    output_format = (
        config.get("population_health", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Population Health configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> PopulationHealthRuntime:
    """
    Initialize runtime context.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    population_config = config.get("population_health", {})

    layer_name = population_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = population_config.get("domain_name", "Population Health")

    logger = configure_logging(project_root, config, run_id)

    runtime = PopulationHealthRuntime(
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
        message="Runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: PopulationHealthRuntime,
    step_name: str,
    status: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Add an audit record.
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
    runtime: PopulationHealthRuntime,
    dataset_name: str,
    rule_name: str,
    status: str,
    message: str,
    failed_count: Optional[int] = None,
) -> None:
    """
    Add a validation record.
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
    runtime: PopulationHealthRuntime,
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
    runtime: PopulationHealthRuntime,
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

def get_output_format(runtime: PopulationHealthRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("population_health", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def read_dataset(path: Path, file_format: Optional[str]) -> pd.DataFrame:
    """
    Read a configured dataset.
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
    Ensure output path uses configured file suffix.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


def write_dataset(dataframe: pd.DataFrame, path: Path, file_format: str) -> None:
    """
    Write dataframe using configured format.
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


def load_input_datasets(runtime: PopulationHealthRuntime) -> Dict[str, pd.DataFrame]:
    """
    Load configured Population Health input datasets.
    """

    logger = runtime.logger
    input_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Population Health input datasets")

    for dataset_name, dataset_config in input_config.items():
        raw_path = dataset_config.get("path")
        file_format = dataset_config.get("format")
        required = bool(dataset_config.get("required", True))

        if not raw_path:
            message = f"No path configured for input dataset: {dataset_name}"

            if required:
                raise ValueError(message)

            logger.warning("Skipping optional dataset with no configured path: %s", dataset_name)
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
                message="Input dataset loaded successfully.",
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

    logger.info("COMPLETE: Load input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: PopulationHealthRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> bool:
    """
    Validate dataset is not empty.
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
    runtime: PopulationHealthRuntime,
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
    runtime: PopulationHealthRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    member_key: str,
) -> bool:
    """
    Validate member key is present and not null.
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


def validate_inputs(runtime: PopulationHealthRuntime, datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Run basic validation against loaded inputs.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    for dataset_name, dataframe in datasets.items():
        validate_not_empty(runtime, dataframe, dataset_name)

        if member_key in dataframe.columns:
            validate_member_key_not_null(runtime, dataframe, dataset_name, member_key)

    add_audit_record(
        runtime=runtime,
        step_name="validate_inputs",
        status=STATUS_SUCCESS,
        message="Input validation completed.",
    )


###############################################################################
# Business Rule Helpers
###############################################################################

def apply_operator(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Apply a configured comparison operator.
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


def build_member_universe(
    runtime: PopulationHealthRuntime,
    datasets: Dict[str, pd.DataFrame],
    preferred_dataset_name: str,
) -> pd.DataFrame:
    """
    Build the base member universe.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    if preferred_dataset_name in datasets:
        source_df = datasets[preferred_dataset_name]
    elif "demographic_features" in datasets:
        source_df = datasets["demographic_features"]
    else:
        raise ValueError("Unable to build member universe. No member-level dataset found.")

    if member_key not in source_df.columns:
        raise ValueError(
            f"Member universe source dataset does not contain member key: {member_key}"
        )

    universe = source_df[[member_key]].drop_duplicates().copy()

    universe["analytics_layer_run_id"] = runtime.run_id
    universe["analytics_domain"] = runtime.domain_name
    universe["built_at_utc"] = utc_now().isoformat()

    return universe


###############################################################################
# Population Cohorts
###############################################################################

def build_population_cohorts(
    runtime: PopulationHealthRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build configured population cohorts.
    """

    logger = runtime.logger
    config = runtime.config.get("population_cohorts", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Population cohorts disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    source_dataset = config.get("source_dataset", "member_360_semantic_view")

    member_universe = build_member_universe(runtime, datasets, source_dataset)
    cohort_rows: List[pd.DataFrame] = []

    logger.info("START: Build population cohorts")

    for cohort_name, cohort_config in config.get("cohorts", {}).items():
        if not bool(cohort_config.get("enabled", True)):
            continue

        rule_type = cohort_config.get("rule_type")
        description = cohort_config.get("description", "")
        source_dataset_name = cohort_config.get("source_dataset", source_dataset)

        add_rule_record(
            runtime=runtime,
            rule_group="population_cohorts",
            rule_name=cohort_name,
            rule_type=rule_type,
            description=description,
            source_dataset=source_dataset_name,
            rule_config=cohort_config,
        )

        if rule_type == "all_members":
            cohort_df = member_universe[[member_key]].copy()

        elif rule_type == "numeric_threshold":
            if source_dataset_name not in datasets:
                logger.warning(
                    "Skipping cohort %s because source dataset is missing: %s",
                    cohort_name,
                    source_dataset_name,
                )
                continue

            source_df = datasets[source_dataset_name]
            column = cohort_config.get("column")
            operator = cohort_config.get("operator")
            threshold = cohort_config.get("threshold")

            required_columns = [member_key, column]
            if not validate_required_columns(
                runtime=runtime,
                dataframe=source_df,
                dataset_name=source_dataset_name,
                required_columns=required_columns,
            ):
                continue

            mask = apply_operator(source_df[column], operator, threshold)
            cohort_df = source_df.loc[mask, [member_key]].drop_duplicates().copy()

        else:
            logger.warning("Unsupported cohort rule_type: %s", rule_type)
            continue

        cohort_df["cohort_name"] = cohort_name
        cohort_df["cohort_description"] = description
        cohort_df["analytics_layer_run_id"] = runtime.run_id
        cohort_df["analytics_domain"] = runtime.domain_name
        cohort_df["analytics_asset_name"] = "population_cohorts"
        cohort_df["built_at_utc"] = utc_now().isoformat()

        cohort_rows.append(cohort_df)

        logger.info(
            "Built cohort: %s | Members: %s",
            cohort_name,
            len(cohort_df),
        )

    if cohort_rows:
        output_df = pd.concat(cohort_rows, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "cohort_name",
                "cohort_description",
                "analytics_layer_run_id",
                "analytics_domain",
                "analytics_asset_name",
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
    )

    logger.info("COMPLETE: Build population cohorts | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Risk Stratification
###############################################################################

def build_risk_stratification(
    runtime: PopulationHealthRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build member risk stratification output.
    """

    logger = runtime.logger
    config = runtime.config.get("risk_stratification", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Risk stratification disabled")
        return pd.DataFrame()

    source_dataset_name = config.get("source_dataset", "risk_features")
    member_key = config.get(
        "member_key",
        runtime.config.get("join_keys", {}).get("member_key", "member_id"),
    )
    risk_score_column = config.get("risk_score_column", "composite_risk_score")

    if source_dataset_name not in datasets:
        raise ValueError(f"Risk stratification source dataset missing: {source_dataset_name}")

    source_df = datasets[source_dataset_name]

    validate_required_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=[member_key, risk_score_column],
    )

    output_df = source_df[[member_key, risk_score_column]].drop_duplicates(
        subset=[member_key]
    ).copy()

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
            description=f"Risk tier {label}",
            source_dataset=source_dataset_name,
            rule_config=tier_config,
        )

        mask = (
            (output_df[risk_score_column] >= min_value)
            & (output_df[risk_score_column] <= max_value)
        )

        output_df.loc[mask, "risk_tier"] = label
        output_df.loc[mask, "risk_priority_rank"] = priority_rank

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "risk_stratification"
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
    )

    logger.info("COMPLETE: Build risk stratification | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Member Segmentation
###############################################################################

def prepare_segmentation_base(
    runtime: PopulationHealthRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a joined member-level dataframe for segmentation.
    """

    config = runtime.config.get("member_segmentation", {})
    member_key = config.get(
        "member_key",
        runtime.config.get("join_keys", {}).get("member_key", "member_id"),
    )
    base_dataset_name = config.get("base_dataset", "demographic_features")

    if base_dataset_name not in datasets:
        raise ValueError(f"Segmentation base dataset missing: {base_dataset_name}")

    base_df = datasets[base_dataset_name]

    if member_key not in base_df.columns:
        raise ValueError(f"Segmentation base dataset missing member key: {member_key}")

    joined_df = base_df[[member_key]].drop_duplicates().copy()

    for dataset_name in config.get("required_datasets", []):
        if dataset_name not in datasets:
            raise ValueError(f"Required segmentation dataset missing: {dataset_name}")

        source_df = datasets[dataset_name]

        if member_key not in source_df.columns:
            raise ValueError(
                f"Required segmentation dataset {dataset_name} missing member key: {member_key}"
            )

        source_deduped = source_df.drop_duplicates(subset=[member_key]).copy()

        non_key_columns = [
            column for column in source_deduped.columns
            if column != member_key
        ]

        rename_map = {
            column: f"{dataset_name}__{column}"
            for column in non_key_columns
            if column in joined_df.columns
        }

        source_deduped = source_deduped.rename(columns=rename_map)

        joined_df = joined_df.merge(source_deduped, on=member_key, how="left")

    return joined_df


def find_condition_column(dataframe: pd.DataFrame, dataset_name: str, column: str) -> str:
    """
    Find condition column after segmentation joins.
    """

    if column in dataframe.columns:
        return column

    prefixed_column = f"{dataset_name}__{column}"

    if prefixed_column in dataframe.columns:
        return prefixed_column

    raise ValueError(
        f"Segmentation condition column not found. Dataset: {dataset_name}, Column: {column}"
    )


def build_member_segmentation(
    runtime: PopulationHealthRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build member segmentation output.
    """

    logger = runtime.logger
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

    for segment_name, segment_config in sorted(
        config.get("segments", {}).items(),
        key=lambda item: item[1].get("priority_rank", 999),
    ):
        description = segment_config.get("description", "")
        priority_rank = segment_config.get("priority_rank")

        add_rule_record(
            runtime=runtime,
            rule_group="member_segmentation",
            rule_name=segment_name,
            rule_type="multi_condition",
            description=description,
            source_dataset=config.get("base_dataset", "demographic_features"),
            rule_config=segment_config,
        )

        segment_mask = pd.Series(True, index=segmentation_df.index)

        for condition in segment_config.get("conditions", []):
            dataset_name = condition.get("dataset")
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            actual_column = find_condition_column(segmentation_df, dataset_name, column)

            condition_mask = apply_operator(segmentation_df[actual_column], operator, value)
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

    output_columns = [
        member_key,
        "member_segment",
        "segment_priority_rank",
    ]

    output_df = segmentation_df[output_columns].copy()
    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "member_segmentation"
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
    Calculate provider attribution metric.
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

    if calculation_type == "count_distinct":
        if not column:
            raise ValueError(f"Metric {metric_name} requires column.")
        metric_df = (
            dataframe.groupby(provider_key)[column]
            .nunique(dropna=True)
            .reset_index(name=metric_name)
        )
        return metric_df

    if calculation_type == "count_rows":
        metric_df = (
            dataframe.groupby(provider_key)
            .size()
            .reset_index(name=metric_name)
        )
        return metric_df

    if calculation_type == "sum":
        if not column:
            raise ValueError(f"Metric {metric_name} requires column.")
        metric_df = (
            dataframe.groupby(provider_key)[column]
            .sum()
            .reset_index(name=metric_name)
        )
        return metric_df

    if calculation_type == "mean":
        if not column:
            raise ValueError(f"Metric {metric_name} requires column.")
        metric_df = (
            dataframe.groupby(provider_key)[column]
            .mean()
            .reset_index(name=metric_name)
        )
        return metric_df

    raise ValueError(f"Unsupported provider metric calculation_type: {calculation_type}")


def build_provider_attribution_analytics(
    runtime: PopulationHealthRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build provider attribution analytics.
    """

    logger = runtime.logger
    config = runtime.config.get("provider_attribution_analytics", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Provider attribution analytics disabled")
        return pd.DataFrame()

    source_dataset_name = config.get("source_dataset", "provider_attribution_features")
    member_key = config.get("member_key", "member_id")
    provider_key = config.get("provider_key", "provider_id")

    if source_dataset_name not in datasets:
        raise ValueError(
            f"Provider attribution source dataset missing: {source_dataset_name}"
        )

    source_df = datasets[source_dataset_name]

    validate_required_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=[member_key, provider_key],
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
            rule_config=metric_config,
        )

        metric_df = calculate_provider_metric(
            dataframe=source_df,
            provider_key=provider_key,
            metric_name=metric_name,
            metric_config=metric_config,
        )

        analytics_df = analytics_df.merge(metric_df, on=provider_key, how="left")

    analytics_df["analytics_layer_run_id"] = runtime.run_id
    analytics_df["analytics_domain"] = runtime.domain_name
    analytics_df["analytics_asset_name"] = "provider_attribution_analytics"
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
    )

    logger.info(
        "COMPLETE: Build provider attribution analytics | Rows: %s",
        len(analytics_df),
    )

    return analytics_df


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(
    runtime: PopulationHealthRuntime,
) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: PopulationHealthRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build column dictionary for Population Health outputs.
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


def build_rule_catalog(runtime: PopulationHealthRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: PopulationHealthRuntime,
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

def get_output_path(runtime: PopulationHealthRuntime, output_group: str, output_name: str) -> Path:
    """
    Resolve output path from configuration.
    """

    output_config = runtime.config.get("paths", {}).get(output_group, {})
    output_entry = output_config.get(output_name)

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        default_path = f"data/analytics_platform/{output_group}/{output_name}"
        raw_path = default_path

    return normalize_path(runtime.project_root, raw_path)


def write_population_health_outputs(
    runtime: PopulationHealthRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Population Health analytics outputs.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Population Health output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Population Health output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: PopulationHealthRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "population_health_dataset_inventory": build_dataset_inventory(runtime),
        "population_health_column_dictionary": build_column_dictionary(runtime, output_assets),
        "population_health_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "population_health_audit_records": pd.DataFrame(runtime.audit_records),
        "population_health_validation_results": pd.DataFrame(runtime.validation_records),
        "population_health_execution_summary": build_execution_summary(runtime, output_assets),
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

def build_population_health_layer(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build complete Population Health Analytics layer.
    """

    runtime: Optional[PopulationHealthRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

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

        write_population_health_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_population_health_layer",
            status=STATUS_SUCCESS,
            message="Population Health Analytics completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Population Health Analytics completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="population_health",
            status=STATUS_SUCCESS,
            message="Population Health Analytics completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Population Health Analytics failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_population_health_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="population_health",
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
        python -m src.analytics_platform.population_health.build_population_health_layer
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