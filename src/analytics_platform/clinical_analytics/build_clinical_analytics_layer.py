###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/clinical_analytics/build_clinical_analytics_layer.py
#
# Layer:
#     Layer 2B - Clinical Analytics
#
# Purpose:
#     Builds Clinical Analytics outputs using the configured Clinical Analytics
#     YAML contract:
#
#         config/analytics_platform/clinical_analytics.yaml
#
# Business Context:
#     Clinical Analytics identifies clinically relevant member populations using
#     available enterprise data assets.
#
#     The current MedFabric Feature Store provides aggregate member-level
#     clinical signals, not diagnosis-line detail. Therefore, this builder
#     creates signal-based clinical registries. These registries are intentionally
#     designed so they can later be upgraded to ICD-based registries when
#     diagnosis-level evidence is exposed from Silver, Gold, or a future clinical
#     feature group.
#
# Outputs:
#     data/analytics_platform/clinical_analytics/
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.clinical_analytics.build_clinical_analytics_layer
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

DEFAULT_CONFIG_PATH = "config/analytics_platform/clinical_analytics.yaml"

DEFAULT_LAYER_NAME = "Layer 2B - Clinical Analytics"

DEFAULT_DOMAIN_NAME = "Clinical Analytics"

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
class ClinicalAnalyticsRuntime:
    """
    Runtime context for one Clinical Analytics execution.
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
    Standard build result returned by the Clinical Analytics builder.
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
    Resolve configured path relative to project root.
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
    Configure Clinical Analytics logging.
    """

    logging_config = config.get("logging", {})

    log_level_name = logging_config.get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)

    log_dir_raw = logging_config.get("module_log_dir", "logs/modules")
    log_dir = normalize_path(project_root, log_dir_raw)
    ensure_directory(log_dir)

    log_file_name = logging_config.get("log_file_name", "clinical_analytics.log")
    log_file_path = log_dir / log_file_name

    logger = logging.getLogger("medfabric.analytics_platform.clinical_analytics")
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
    logger.info("MedFabric Clinical Analytics started")
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
    Validate required Clinical Analytics configuration sections.
    """

    errors: List[str] = []

    required_sections = [
        "clinical_analytics",
        "logging",
        "paths",
        "join_keys",
        "registry_framework",
        "condition_registry",
        "disease_registries",
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
        config.get("clinical_analytics", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Clinical Analytics configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> ClinicalAnalyticsRuntime:
    """
    Initialize Clinical Analytics runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    clinical_config = config.get("clinical_analytics", {})

    layer_name = clinical_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = clinical_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = ClinicalAnalyticsRuntime(
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
        message="Clinical Analytics runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: ClinicalAnalyticsRuntime,
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
    runtime: ClinicalAnalyticsRuntime,
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
    runtime: ClinicalAnalyticsRuntime,
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
    runtime: ClinicalAnalyticsRuntime,
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

def get_output_format(runtime: ClinicalAnalyticsRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("clinical_analytics", {})
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


def load_input_datasets(runtime: ClinicalAnalyticsRuntime) -> Dict[str, pd.DataFrame]:
    """
    Load configured Clinical Analytics input datasets.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Clinical Analytics input datasets")

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

    logger.info("COMPLETE: Load Clinical Analytics input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: ClinicalAnalyticsRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> bool:
    """
    Validate that dataset is not empty.
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
    runtime: ClinicalAnalyticsRuntime,
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
    runtime: ClinicalAnalyticsRuntime,
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
    runtime: ClinicalAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Run input validation.
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
        message="Clinical Analytics input validation completed.",
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
    runtime: ClinicalAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build base member universe for Clinical Analytics.
    """

    framework_config = runtime.config.get("registry_framework", {})
    base_dataset_name = framework_config.get("base_member_dataset", "demographic_features")
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
# Condition Registry
###############################################################################

def build_condition_registry(
    runtime: ClinicalAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build signal-based condition registry.
    """

    logger = runtime.logger
    config = runtime.config.get("condition_registry", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Condition registry disabled")
        return pd.DataFrame()

    member_key = config.get("member_key", "member_id")
    condition_rows: List[pd.DataFrame] = []

    logger.info("START: Build signal-based condition registry")

    for signal_name, signal_config in config.get("condition_signals", {}).items():
        condition_name = signal_config.get("condition_name", signal_name)
        condition_category = signal_config.get("condition_category", "")
        source_dataset_name = signal_config.get("source_dataset")
        signal_column = signal_config.get("signal_column")
        operator = signal_config.get("operator")
        value = signal_config.get("value")
        evidence_column = signal_config.get("evidence_column", signal_column)

        add_rule_record(
            runtime=runtime,
            rule_group="condition_registry",
            rule_name=signal_name,
            rule_type="signal_based_condition",
            description=f"Signal-based condition rule for {condition_name}",
            source_dataset=source_dataset_name,
            rule_config=signal_config,
        )

        if source_dataset_name not in datasets:
            logger.warning(
                "Skipping condition signal %s because source dataset is missing: %s",
                signal_name,
                source_dataset_name,
            )
            continue

        source_df = datasets[source_dataset_name]

        required_columns = [member_key, signal_column]
        if evidence_column:
            required_columns.append(evidence_column)

        if not validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset_name,
            required_columns=required_columns,
        ):
            continue

        mask = apply_operator(source_df[signal_column], operator, value)
        matched = source_df.loc[mask, [member_key, evidence_column]].copy()

        matched = matched.rename(columns={evidence_column: "condition_evidence_value"})
        matched["condition_name"] = condition_name
        matched["condition_category"] = condition_category
        matched["condition_source"] = source_dataset_name
        matched["condition_rule"] = signal_name
        matched["condition_evidence_count"] = 1
        matched["analytics_layer_run_id"] = runtime.run_id
        matched["analytics_domain"] = runtime.domain_name
        matched["analytics_asset_name"] = "condition_registry"
        matched["built_at_utc"] = utc_now().isoformat()

        condition_rows.append(matched)

        logger.info(
            "Built condition signal: %s | Members: %s",
            signal_name,
            len(matched),
        )

    if condition_rows:
        output_df = pd.concat(condition_rows, ignore_index=True)
    else:
        output_df = pd.DataFrame(
            columns=[
                member_key,
                "condition_evidence_value",
                "condition_name",
                "condition_category",
                "condition_source",
                "condition_rule",
                "condition_evidence_count",
                "analytics_layer_run_id",
                "analytics_domain",
                "analytics_asset_name",
                "built_at_utc",
            ]
        )

    add_dataset_record(
        runtime=runtime,
        dataset_name="condition_registry",
        dataset_type="clinical_analytics_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(output_df),
        column_count=len(output_df.columns),
        message="Signal-based condition registry built successfully.",
    )

    logger.info("COMPLETE: Build condition registry | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Disease Registries
###############################################################################

def build_single_disease_registry(
    runtime: ClinicalAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
    registry_key: str,
    registry_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build one signal-based disease registry.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    source_dataset_name = registry_config.get("source_dataset")
    signal_column = registry_config.get("signal_column")
    operator = registry_config.get("operator")
    value = registry_config.get("value")
    evidence_column = registry_config.get("evidence_column", signal_column)

    registry_name = registry_config.get("registry_name", registry_key)
    condition_name = registry_config.get("condition_name", registry_name)
    condition_category = registry_config.get("condition_category", "")
    registry_method = registry_config.get("registry_method", "signal_based_placeholder")
    note = registry_config.get("note", "")

    add_rule_record(
        runtime=runtime,
        rule_group="disease_registries",
        rule_name=registry_key,
        rule_type=registry_method,
        description=note,
        source_dataset=source_dataset_name,
        rule_config=registry_config,
    )

    if source_dataset_name not in datasets:
        raise ValueError(
            f"Disease registry source dataset missing for {registry_key}: "
            f"{source_dataset_name}"
        )

    source_df = datasets[source_dataset_name]

    required_columns = [member_key, signal_column]
    if evidence_column:
        required_columns.append(evidence_column)

    validate_required_columns(
        runtime=runtime,
        dataframe=source_df,
        dataset_name=source_dataset_name,
        required_columns=required_columns,
    )

    mask = apply_operator(source_df[signal_column], operator, value)

    registry_df = source_df.loc[mask, [member_key, evidence_column]].copy()
    registry_df = registry_df.rename(columns={evidence_column: "registry_evidence_value"})

    registry_df["registry_key"] = registry_key
    registry_df["registry_name"] = registry_name
    registry_df["condition_name"] = condition_name
    registry_df["condition_category"] = condition_category
    registry_df["registry_method"] = registry_method
    registry_df["source_dataset"] = source_dataset_name
    registry_df["signal_column"] = signal_column
    registry_df["registry_note"] = note
    registry_df["analytics_layer_run_id"] = runtime.run_id
    registry_df["analytics_domain"] = runtime.domain_name
    registry_df["analytics_asset_name"] = registry_config.get("output_name", registry_key)
    registry_df["built_at_utc"] = utc_now().isoformat()

    return registry_df


def build_disease_registries(
    runtime: ClinicalAnalyticsRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Build all configured disease registries.
    """

    logger = runtime.logger
    registries_config = runtime.config.get("disease_registries", {})

    registry_outputs: Dict[str, pd.DataFrame] = {}

    logger.info("START: Build signal-based disease registries")

    for registry_key, registry_config in registries_config.items():
        if not bool(registry_config.get("enabled", True)):
            logger.info("SKIP registry disabled: %s", registry_key)
            continue

        output_name = registry_config.get("output_name", f"{registry_key}_registry")

        registry_df = build_single_disease_registry(
            runtime=runtime,
            datasets=datasets,
            registry_key=registry_key,
            registry_config=registry_config,
        )

        registry_outputs[output_name] = registry_df

        add_dataset_record(
            runtime=runtime,
            dataset_name=output_name,
            dataset_type="clinical_analytics_output",
            status=STATUS_SUCCESS,
            path=None,
            row_count=len(registry_df),
            column_count=len(registry_df.columns),
            message=f"Signal-based disease registry built successfully: {output_name}",
        )

        logger.info(
            "Built disease registry: %s | Rows: %s",
            output_name,
            len(registry_df),
        )

    logger.info("COMPLETE: Build disease registries | Count: %s", len(registry_outputs))

    return registry_outputs


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: ClinicalAnalyticsRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: ClinicalAnalyticsRuntime,
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


def build_rule_catalog(runtime: ClinicalAnalyticsRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: ClinicalAnalyticsRuntime,
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
    runtime: ClinicalAnalyticsRuntime,
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


def write_clinical_outputs(
    runtime: ClinicalAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Clinical Analytics output datasets.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Clinical Analytics output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Clinical Analytics output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: ClinicalAnalyticsRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "clinical_analytics_dataset_inventory": build_dataset_inventory(runtime),
        "clinical_analytics_column_dictionary": build_column_dictionary(runtime, output_assets),
        "clinical_analytics_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "clinical_analytics_audit_records": pd.DataFrame(runtime.audit_records),
        "clinical_analytics_validation_results": pd.DataFrame(runtime.validation_records),
        "clinical_analytics_execution_summary": build_execution_summary(runtime, output_assets),
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

def build_clinical_analytics_layer(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build complete Clinical Analytics layer.
    """

    runtime: Optional[ClinicalAnalyticsRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

        condition_registry = build_condition_registry(runtime, datasets)
        disease_registry_outputs = build_disease_registries(runtime, datasets)

        output_assets: Dict[str, pd.DataFrame] = {
            "condition_registry": condition_registry,
        }
        output_assets.update(disease_registry_outputs)

        write_clinical_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_clinical_analytics_layer",
            status=STATUS_SUCCESS,
            message="Clinical Analytics completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Clinical Analytics completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="clinical_analytics",
            status=STATUS_SUCCESS,
            message="Clinical Analytics completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Clinical Analytics failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_clinical_analytics_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="clinical_analytics",
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
        python -m src.analytics_platform.clinical_analytics.build_clinical_analytics_layer
    """

    config_path = os.environ.get(
        "MEDFABRIC_CLINICAL_ANALYTICS_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_clinical_analytics_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Clinical Analytics failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()