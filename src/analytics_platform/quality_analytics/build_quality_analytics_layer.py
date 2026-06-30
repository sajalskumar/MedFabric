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
#     The current Feature Store contains member-level aggregate signals, not
#     detailed HEDIS/CMS event-level evidence. Therefore, this implementation
#     builds signal-based quality analytics that can later be upgraded when
#     detailed clinical, claims, pharmacy, and quality-event data are available.
#
# Inputs:
#     Feature Store:
#         data/feature_store/claims_features.parquet
#         data/feature_store/pharmacy_features.parquet
#         data/feature_store/laboratory_features.parquet
#         data/feature_store/risk_features.parquet
#
#     Population Health:
#         data/analytics_platform/population_health/member_segmentation.parquet
#         data/analytics_platform/population_health/risk_stratification.parquet
#
#     Clinical Analytics:
#         data/analytics_platform/clinical_analytics/condition_registry.parquet
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

DEFAULT_CONFIG_PATH = "config/analytics_platform/quality_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2C - Quality Analytics"

DEFAULT_DOMAIN_NAME = "Quality Analytics"

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
class QualityAnalyticsRuntime:
    """
    Runtime context for one Quality Analytics execution.

    This object holds run metadata, configuration, logger, audit records,
    validation records, dataset inventory records, and rule catalog records.
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
    Standard build result returned by the Quality Analytics builder.
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

    Args:
        project_root:
            Current repository root.

        raw_path:
            Configured path. Can be relative or absolute.

    Returns:
        Resolved absolute path.
    """

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_directory(path: Path) -> None:
    """
    Create directory if it does not already exist.
    """

    path.mkdir(parents=True, exist_ok=True)


def safe_string(value: Any) -> str:
    """
    Convert value to safe string for metadata output.
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
    Configure Quality Analytics logging.
    """

    logging_config = config.get("logging", {})

    log_level_name = logging_config.get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)

    log_dir_raw = logging_config.get("module_log_dir", "logs/modules")
    log_dir = normalize_path(project_root, log_dir_raw)
    ensure_directory(log_dir)

    log_file_name = logging_config.get("log_file_name", "quality_analytics.log")
    log_file_path = log_dir / log_file_name

    logger = logging.getLogger("medfabric.analytics_platform.quality_analytics")
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
    logger.info("MedFabric Quality Analytics started")
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
    Validate required Quality Analytics configuration sections.
    """

    errors: List[str] = []

    required_sections = [
        "quality_analytics",
        "logging",
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
        if "metadata_outputs" not in paths:
            errors.append("Missing required configuration section: paths.metadata_outputs")
        if "audit_outputs" not in paths:
            errors.append("Missing required configuration section: paths.audit_outputs")

    output_format = (
        config.get("quality_analytics", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Quality Analytics configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> QualityAnalyticsRuntime:
    """
    Initialize Quality Analytics runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    quality_config = config.get("quality_analytics", {})

    layer_name = quality_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = quality_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = QualityAnalyticsRuntime(
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
        message="Quality Analytics runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, and Metadata Records
###############################################################################

def add_audit_record(
    runtime: QualityAnalyticsRuntime,
    step_name: str,
    status: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Add execution audit record.
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
    runtime: QualityAnalyticsRuntime,
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
    runtime: QualityAnalyticsRuntime,
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
    runtime: QualityAnalyticsRuntime,
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

def get_output_format(runtime: QualityAnalyticsRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("quality_analytics", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def read_dataset(path: Path, file_format: Optional[str]) -> pd.DataFrame:
    """
    Read configured dataset from disk.
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
    Write dataframe to disk using configured output format.
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


def load_input_datasets(runtime: QualityAnalyticsRuntime) -> Dict[str, pd.DataFrame]:
    """
    Load configured Quality Analytics input datasets.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Quality Analytics input datasets")

    for dataset_name, dataset_config in inputs_config.items():
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

    logger.info("COMPLETE: Load Quality Analytics input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: QualityAnalyticsRuntime,
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
    runtime: QualityAnalyticsRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    required_columns: Iterable[str],
) -> bool:
    """
    Validate required columns exist in a dataframe.
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
    runtime: QualityAnalyticsRuntime,
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


def validate_inputs(
    runtime: QualityAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Run basic validation against loaded input datasets.
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
        message="Quality Analytics input validation completed.",
    )


###############################################################################
# Business Rule Helpers
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


def build_base_member_universe(
    runtime: QualityAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build base member universe for Quality Analytics.
    """

    framework_config = runtime.config.get("quality_framework", {})
    base_dataset_name = framework_config.get("base_member_dataset", "risk_features")
    member_key = framework_config.get("member_key", "member_id")

    if base_dataset_name not in datasets:
        raise ValueError(f"Base member dataset missing: {base_dataset_name}")

    base_df = datasets[base_dataset_name]

    if member_key not in base_df.columns:
        raise ValueError(
            f"Base member dataset '{base_dataset_name}' missing member key: {member_key}"
        )

    universe = base_df[[member_key]].drop_duplicates().copy()

    for enrichment_dataset_name in framework_config.get("enrichment_datasets", []):
        if enrichment_dataset_name not in datasets:
            continue

        enrichment_df = datasets[enrichment_dataset_name]

        if member_key not in enrichment_df.columns:
            continue

        enrichment_deduped = enrichment_df.drop_duplicates(subset=[member_key]).copy()
        universe = universe.merge(enrichment_deduped, on=member_key, how="left")

    return universe


###############################################################################
# Quality Measures
###############################################################################

def build_quality_measures(
    runtime: QualityAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build configured signal-based quality measures.

    Output grain:
        One row per configured quality measure.
    """

    logger = runtime.logger
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
            source_dataset=source_dataset_name,
            rule_config=measure_config,
        )

        if source_dataset_name not in datasets:
            raise ValueError(f"Quality measure source dataset missing: {source_dataset_name}")

        source_df = datasets[source_dataset_name]

        required_columns = [member_key, numerator_rule.get("column")]
        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=required_columns,
        )

        denominator_count = int(source_df[member_key].nunique(dropna=True))

        numerator_column = numerator_rule.get("column")
        operator = numerator_rule.get("operator")
        value = numerator_rule.get("value")

        numerator_mask = apply_operator(source_df[numerator_column], operator, value)
        numerator_count = int(source_df.loc[numerator_mask, member_key].nunique(dropna=True))

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
                "analytics_layer_run_id": runtime.run_id,
                "analytics_domain": runtime.domain_name,
                "analytics_asset_name": "quality_measures",
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
    )

    logger.info("COMPLETE: Build quality measures | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Care Gaps
###############################################################################

def build_care_gaps(
    runtime: QualityAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build configured care gaps.

    Output grain:
        One row per member per care gap.
    """

    logger = runtime.logger
    config = runtime.config.get("care_gaps", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Care gaps disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    gap_rows: List[pd.DataFrame] = []

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
            source_dataset=source_dataset_name,
            rule_config=gap_config,
        )

        if source_dataset_name not in datasets:
            raise ValueError(f"Care gap source dataset missing: {source_dataset_name}")

        source_df = datasets[source_dataset_name]

        signal_column = gap_rule.get("column")
        operator = gap_rule.get("operator")
        value = gap_rule.get("value")

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=[member_key, signal_column],
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched = source_df.loc[mask, [member_key, signal_column]].copy()
        matched = matched.rename(columns={signal_column: "care_gap_evidence_value"})

        matched["care_gap_key"] = gap_key
        matched["care_gap_name"] = care_gap_name
        matched["care_gap_category"] = care_gap_category
        matched["care_gap_severity"] = severity
        matched["source_dataset"] = source_dataset_name
        matched["care_gap_description"] = description
        matched["analytics_layer_run_id"] = runtime.run_id
        matched["analytics_domain"] = runtime.domain_name
        matched["analytics_asset_name"] = "care_gaps"
        matched["built_at_utc"] = utc_now().isoformat()

        gap_rows.append(matched)

        logger.info(
            "Built care gap: %s | Members: %s",
            care_gap_name,
            len(matched),
        )

    if gap_rows:
        output_df = pd.concat(gap_rows, ignore_index=True)
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
    )

    logger.info("COMPLETE: Build care gaps | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Preventive Care
###############################################################################

def build_preventive_care(
    runtime: QualityAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build configured preventive care signal outputs.

    Output grain:
        One row per member per preventive care signal.
    """

    logger = runtime.logger
    config = runtime.config.get("preventive_care", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Preventive care disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    signal_rows: List[pd.DataFrame] = []

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
            source_dataset=source_dataset_name,
            rule_config=signal_config,
        )

        if source_dataset_name not in datasets:
            raise ValueError(
                f"Preventive care source dataset missing: {source_dataset_name}"
            )

        source_df = datasets[source_dataset_name]

        signal_column = signal_rule.get("column")
        operator = signal_rule.get("operator")
        value = signal_rule.get("value")

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=[member_key, signal_column],
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched = source_df.loc[mask, [member_key, signal_column]].copy()
        matched = matched.rename(columns={signal_column: "preventive_evidence_value"})

        matched["preventive_care_key"] = signal_key
        matched["preventive_care_name"] = preventive_care_name
        matched["preventive_care_category"] = preventive_care_category
        matched["source_dataset"] = source_dataset_name
        matched["preventive_care_description"] = description
        matched["analytics_layer_run_id"] = runtime.run_id
        matched["analytics_domain"] = runtime.domain_name
        matched["analytics_asset_name"] = "preventive_care"
        matched["built_at_utc"] = utc_now().isoformat()

        signal_rows.append(matched)

        logger.info(
            "Built preventive care signal: %s | Members: %s",
            preventive_care_name,
            len(matched),
        )

    if signal_rows:
        output_df = pd.concat(signal_rows, ignore_index=True)
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
    )

    logger.info("COMPLETE: Build preventive care | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Medication Adherence
###############################################################################

def build_medication_adherence(
    runtime: QualityAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build configured medication adherence signal outputs.

    Output grain:
        One row per member per adherence signal.
    """

    logger = runtime.logger
    config = runtime.config.get("medication_adherence", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Medication adherence disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    adherence_rows: List[pd.DataFrame] = []

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
            source_dataset=source_dataset_name,
            rule_config=signal_config,
        )

        if source_dataset_name not in datasets:
            raise ValueError(
                f"Medication adherence source dataset missing: {source_dataset_name}"
            )

        source_df = datasets[source_dataset_name]

        signal_column = signal_rule.get("column")
        operator = signal_rule.get("operator")
        value = signal_rule.get("value")

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=[member_key, signal_column, evidence_column],
        )

        mask = apply_operator(source_df[signal_column], operator, value)

        matched = source_df.loc[mask, [member_key, evidence_column]].copy()
        matched = matched.rename(columns={evidence_column: "adherence_evidence_value"})

        matched["adherence_key"] = signal_key
        matched["adherence_name"] = adherence_name
        matched["adherence_category"] = adherence_category
        matched["source_dataset"] = source_dataset_name
        matched["adherence_description"] = description
        matched["analytics_layer_run_id"] = runtime.run_id
        matched["analytics_domain"] = runtime.domain_name
        matched["analytics_asset_name"] = "medication_adherence"
        matched["built_at_utc"] = utc_now().isoformat()

        adherence_rows.append(matched)

        logger.info(
            "Built medication adherence signal: %s | Members: %s",
            adherence_name,
            len(matched),
        )

    if adherence_rows:
        output_df = pd.concat(adherence_rows, ignore_index=True)
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
    )

    logger.info("COMPLETE: Build medication adherence | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Summary Outputs
###############################################################################

def build_measure_summary(
    runtime: QualityAnalyticsRuntime,
    quality_measures: pd.DataFrame,
    summary_config: Dict[str, Any],
    output_name: str,
) -> pd.DataFrame:
    """
    Build HEDIS or CMS placeholder summary from configured measure names.
    """

    if quality_measures.empty:
        return pd.DataFrame()

    included_measure_keys = summary_config.get("measures_included", [])
    summary_method = summary_config.get("summary_method", "signal_based_placeholder")
    note = summary_config.get("note", "")

    summary_df = quality_measures[
        quality_measures["measure_key"].isin(included_measure_keys)
    ].copy()

    if summary_df.empty:
        summary_df = quality_measures.copy()

    summary_df["summary_name"] = output_name
    summary_df["summary_method"] = summary_method
    summary_df["summary_note"] = note
    summary_df["analytics_asset_name"] = output_name
    summary_df["analytics_layer_run_id"] = runtime.run_id
    summary_df["analytics_domain"] = runtime.domain_name
    summary_df["built_at_utc"] = utc_now().isoformat()

    return summary_df


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: QualityAnalyticsRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: QualityAnalyticsRuntime,
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


def build_rule_catalog(runtime: QualityAnalyticsRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: QualityAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one-row execution summary.
    """

    end_time = utc_now()
    duration_seconds = (end_time - runtime.start_time_utc).total_seconds()

    failed_validation_count = sum(
        1
        for record in runtime.validation_records
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
    runtime: QualityAnalyticsRuntime,
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


def write_quality_outputs(
    runtime: QualityAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Quality Analytics output datasets.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Quality Analytics output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Quality Analytics output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: QualityAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "quality_analytics_dataset_inventory": build_dataset_inventory(runtime),
        "quality_analytics_column_dictionary": build_column_dictionary(runtime, output_assets),
        "quality_analytics_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "quality_analytics_audit_records": pd.DataFrame(runtime.audit_records),
        "quality_analytics_validation_results": pd.DataFrame(runtime.validation_records),
        "quality_analytics_execution_summary": build_execution_summary(runtime, output_assets),
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

def build_quality_analytics_layer(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build complete Quality Analytics layer.
    """

    runtime: Optional[QualityAnalyticsRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

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

        write_quality_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_quality_analytics_layer",
            status=STATUS_SUCCESS,
            message="Quality Analytics completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Quality Analytics completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="quality_analytics",
            status=STATUS_SUCCESS,
            message="Quality Analytics completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Quality Analytics failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_quality_analytics_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="quality_analytics",
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
        python -m src.analytics_platform.quality_analytics.build_quality_analytics_layer
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