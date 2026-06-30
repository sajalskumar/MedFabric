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

DEFAULT_CONFIG_PATH = "config/analytics_platform/care_management.yaml"

DEFAULT_LAYER_NAME = "Layer 2G - Care Management"
DEFAULT_DOMAIN_NAME = "Care Management"
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
class CareManagementRuntime:
    """
    Runtime context for one Care Management execution.
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
    Standard build result returned by Care Management.
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
    Create directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)


def safe_string(value: Any) -> str:
    """
    Convert value to safe string.
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
    Configure Care Management logging.
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
        "care_management.log",
    )

    logger = logging.getLogger("medfabric.analytics_platform.care_management")
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
    logger.info("MedFabric Care Management started")
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
    Validate required Care Management configuration sections.
    """

    errors: List[str] = []

    required_sections = [
        "care_management",
        "logging",
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
        config.get("care_management", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Care Management configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(
    config_path_raw: str = DEFAULT_CONFIG_PATH,
) -> CareManagementRuntime:
    """
    Initialize Care Management runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    layer_config = config.get("care_management", {})

    layer_name = layer_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = layer_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = CareManagementRuntime(
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
        message="Care Management runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: CareManagementRuntime,
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
    runtime: CareManagementRuntime,
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
    runtime: CareManagementRuntime,
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
    runtime: CareManagementRuntime,
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

def get_output_format(runtime: CareManagementRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("care_management", {})
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
    runtime: CareManagementRuntime,
) -> Dict[str, pd.DataFrame]:
    """
    Load configured Care Management input datasets.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})

    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Care Management input datasets")

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
                message="Care Management input dataset loaded successfully.",
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

    logger.info("COMPLETE: Load Care Management input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: CareManagementRuntime,
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
    runtime: CareManagementRuntime,
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
    runtime: CareManagementRuntime,
    dataframe: pd.DataFrame,
    dataset_name: str,
    member_key: str,
) -> bool:
    """
    Validate member key exists and is not null.
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
    runtime: CareManagementRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Validate loaded inputs.
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
        message="Care Management input validation completed.",
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


def deduplicate_member_dataset(
    dataframe: pd.DataFrame,
    member_key: str,
    sort_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Deduplicate a member-level dataset.
    """

    if member_key not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()

    if sort_columns:
        existing_sort_columns = [column for column in sort_columns if column in result.columns]
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
    Merge member-level enrichment datasets onto base dataframe.
    """

    output_df = base_df.copy()

    for dataset_name in enrichment_dataset_names:
        if dataset_name not in datasets:
            continue

        enrichment_df = datasets[dataset_name]

        if member_key not in enrichment_df.columns:
            continue

        enrichment_deduped = deduplicate_member_dataset(
            enrichment_df,
            member_key,
            sort_columns=[
                "program_priority_rank",
                "priority_rank",
                "segment_priority_rank",
                "risk_priority_rank",
            ],
        )

        columns_to_keep = [column for column in enrichment_deduped.columns if column != member_key]

        rename_map = {}
        for column in columns_to_keep:
            if column in output_df.columns:
                rename_map[column] = f"{dataset_name}__{column}"

        enrichment_deduped = enrichment_deduped.rename(columns=rename_map)

        output_df = output_df.merge(enrichment_deduped, on=member_key, how="left")

    return output_df


###############################################################################
# Care Programs
###############################################################################

def build_care_programs(
    runtime: CareManagementRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build configured care program assignments.

    Output grain:
        One row per member per care program.
    """

    logger = runtime.logger
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
            raise ValueError(f"Care program source dataset missing: {source_dataset}")

        source_df = datasets[source_dataset]

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[member_key, rule_column],
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        program_df = source_df.loc[mask].copy()

        program_df = deduplicate_member_dataset(
            program_df,
            member_key,
            sort_columns=["priority_rank", "segment_priority_rank", "risk_priority_rank"],
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
        program_df["analytics_layer_run_id"] = runtime.run_id
        program_df["analytics_domain"] = runtime.domain_name
        program_df["analytics_asset_name"] = "care_programs"
        program_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="care_programs",
            rule_name=rule_key,
            rule_type="care_program_assignment",
            description=description,
            source_dataset=source_dataset,
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
    )

    logger.info("COMPLETE: Build care programs | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Case Management Worklist
###############################################################################

def select_available_columns(dataframe: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Return columns that exist in dataframe.
    """

    return [column for column in columns if column in dataframe.columns]


def build_case_management_worklist(
    runtime: CareManagementRuntime,
    care_programs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build case management worklist.

    Output grain:
        One row per member.
    """

    logger = runtime.logger
    config = runtime.config.get("case_management_worklist", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Case management worklist disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    if care_programs.empty:
        return pd.DataFrame()

    priority_columns = select_available_columns(
        care_programs,
        config.get("priority_columns", []),
    )

    sort_columns = priority_columns if priority_columns else ["program_priority_rank"]

    output_df = (
        care_programs.sort_values(sort_columns, ascending=True)
        .drop_duplicates(subset=[member_key])
        .copy()
    )

    output_columns = select_available_columns(
        output_df,
        config.get("output_columns", []),
    )

    if output_columns:
        output_df = output_df[output_columns].copy()

    output_df["case_status"] = "Open"
    output_df["case_priority_rank"] = range(1, len(output_df) + 1)
    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "case_management_worklist"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="case_management_worklist",
        rule_name="member_case_prioritization",
        rule_type="dedupe_and_rank",
        description="Creates one prioritized care management row per member.",
        source_dataset="care_programs",
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
    )

    logger.info("COMPLETE: Build case management worklist | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Transitions of Care
###############################################################################

def build_transitions_of_care(
    runtime: CareManagementRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build transitions of care candidate registry.
    """

    logger = runtime.logger
    config = runtime.config.get("transitions_of_care", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Transitions of care disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    source_dataset = config.get("source_dataset", "high_priority_member_registry")

    if source_dataset not in datasets:
        raise ValueError(f"Transitions of care source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build transitions of care")

    for rule_key, rule_config in config.get("transition_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[member_key, rule_column],
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        transition_df = source_df.loc[mask].copy()
        transition_df = deduplicate_member_dataset(
            transition_df,
            member_key,
            sort_columns=["priority_rank"],
        )

        transition_df["transition_key"] = rule_key
        transition_df["transition_name"] = rule_config.get("transition_name", rule_key)
        transition_df["transition_category"] = rule_config.get("transition_category", "")
        transition_df["transition_priority_rank"] = rule_config.get("transition_priority_rank")
        transition_df["transition_description"] = rule_config.get("description", "")
        transition_df["analytics_layer_run_id"] = runtime.run_id
        transition_df["analytics_domain"] = runtime.domain_name
        transition_df["analytics_asset_name"] = "transitions_of_care"
        transition_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="transitions_of_care",
            rule_name=rule_key,
            rule_type="transition_candidate_selection",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
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
    )

    logger.info("COMPLETE: Build transitions of care | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Disease Management
###############################################################################

def build_disease_management(
    runtime: CareManagementRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build disease management program candidates.
    """

    logger = runtime.logger
    config = runtime.config.get("disease_management", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Disease management disabled")
        return pd.DataFrame()

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    source_dataset = config.get("source_dataset", "condition_registry")

    if source_dataset not in datasets:
        raise ValueError(f"Disease management source dataset missing: {source_dataset}")

    source_df = datasets[source_dataset]
    output_frames: List[pd.DataFrame] = []

    logger.info("START: Build disease management")

    for rule_key, rule_config in config.get("disease_rules", {}).items():
        if not bool(rule_config.get("enabled", True)):
            continue

        rule_column = rule_config.get("rule_column")
        operator = rule_config.get("operator")
        value = rule_config.get("value")

        validate_required_columns(
            runtime=runtime,
            dataframe=source_df,
            dataset_name=source_dataset,
            required_columns=[member_key, rule_column],
        )

        mask = apply_operator(source_df[rule_column], operator, value)

        disease_df = source_df.loc[mask].copy()
        disease_df = deduplicate_member_dataset(disease_df, member_key)

        disease_df["disease_rule_key"] = rule_key
        disease_df["disease_program_name"] = rule_config.get("disease_program_name", rule_key)
        disease_df["disease_category"] = rule_config.get("disease_category", "")
        disease_df["disease_priority_rank"] = rule_config.get("disease_priority_rank")
        disease_df["disease_program_description"] = rule_config.get("description", "")
        disease_df["analytics_layer_run_id"] = runtime.run_id
        disease_df["analytics_domain"] = runtime.domain_name
        disease_df["analytics_asset_name"] = "disease_management"
        disease_df["built_at_utc"] = utc_now().isoformat()

        add_rule_record(
            runtime=runtime,
            rule_group="disease_management",
            rule_name=rule_key,
            rule_type="disease_program_assignment",
            description=rule_config.get("description", ""),
            source_dataset=source_dataset,
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
    )

    logger.info("COMPLETE: Build disease management | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Outreach Tracking
###############################################################################

def build_outreach_tracking(
    runtime: CareManagementRuntime,
    case_management_worklist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build outreach tracking placeholder records.
    """

    logger = runtime.logger
    config = runtime.config.get("outreach_tracking", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Outreach tracking disabled")
        return pd.DataFrame()

    if case_management_worklist.empty:
        return pd.DataFrame()

    output_df = case_management_worklist.copy()

    output_df["outreach_status"] = config.get("default_outreach_status", "Pending")
    output_df["outreach_channel"] = config.get("default_outreach_channel", "Care Manager Review")
    output_df["owner_role"] = config.get("default_owner_role", "Care Management Team")
    output_df["outreach_attempt_count"] = 0
    output_df["last_outreach_date"] = None
    output_df["next_outreach_action"] = "Initial Review"
    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "outreach_tracking"
    output_df["built_at_utc"] = utc_now().isoformat()

    add_rule_record(
        runtime=runtime,
        rule_group="outreach_tracking",
        rule_name="default_outreach_tracking",
        rule_type="status_initialization",
        description="Initializes outreach tracking fields for case management worklist.",
        source_dataset="case_management_worklist",
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
    )

    logger.info("COMPLETE: Build outreach tracking | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Program Effectiveness
###############################################################################

def calculate_group_metric(
    dataframe: pd.DataFrame,
    group_by: List[str],
    metric_name: str,
    metric_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate configured grouped metric.
    """

    calculation_type = metric_config.get("calculation_type")
    column = metric_config.get("column")

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

    if calculation_type == "count_rows":
        return (
            dataframe.groupby(group_by)
            .size()
            .reset_index(name=metric_name)
        )

    raise ValueError(f"Unsupported calculation_type: {calculation_type}")


def build_program_effectiveness(
    runtime: CareManagementRuntime,
    care_programs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build program effectiveness summary.

    Current version summarizes assigned members by care program.
    Future versions can add actual outcome tracking once intervention and
    follow-up event data exists.
    """

    logger = runtime.logger
    config = runtime.config.get("program_effectiveness", {})

    if not bool(config.get("enabled", True)):
        logger.info("SKIP: Program effectiveness disabled")
        return pd.DataFrame()

    if care_programs.empty:
        return pd.DataFrame()

    group_by = config.get("group_by", [])
    metrics = config.get("metrics", {})

    validate_required_columns(
        runtime=runtime,
        dataframe=care_programs,
        dataset_name="care_programs",
        required_columns=group_by,
    )

    output_df = care_programs[group_by].drop_duplicates().copy()

    for metric_name, metric_config in metrics.items():
        validate_required_columns(
            runtime=runtime,
            dataframe=care_programs,
            dataset_name="care_programs",
            required_columns=[metric_config.get("column")],
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
            rule_config=metric_config,
        )

    output_df["analytics_layer_run_id"] = runtime.run_id
    output_df["analytics_domain"] = runtime.domain_name
    output_df["analytics_asset_name"] = "program_effectiveness"
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
    )

    logger.info("COMPLETE: Build program effectiveness | Rows: %s", len(output_df))

    return output_df


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: CareManagementRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: CareManagementRuntime,
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


def build_rule_catalog(runtime: CareManagementRuntime) -> pd.DataFrame:
    """
    Build rule catalog.
    """

    return pd.DataFrame(runtime.rule_records)


def build_execution_summary(
    runtime: CareManagementRuntime,
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
    runtime: CareManagementRuntime,
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


def write_care_management_outputs(
    runtime: CareManagementRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Care Management outputs.
    """

    output_format = get_output_format(runtime)

    for output_name, dataframe in output_assets.items():
        output_path = get_output_path(runtime, "outputs", output_name)
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote Care Management output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_output:{output_name}",
            status=STATUS_SUCCESS,
            message="Care Management output written successfully.",
            row_count=len(dataframe),
            output_path=str(output_path),
        )


def write_metadata_and_audit_outputs(
    runtime: CareManagementRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Write metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "care_management_dataset_inventory": build_dataset_inventory(runtime),
        "care_management_column_dictionary": build_column_dictionary(runtime, output_assets),
        "care_management_rule_catalog": build_rule_catalog(runtime),
    }

    audit_assets = {
        "care_management_audit_records": pd.DataFrame(runtime.audit_records),
        "care_management_validation_results": pd.DataFrame(runtime.validation_records),
        "care_management_execution_summary": build_execution_summary(runtime, output_assets),
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

def build_care_management_layer(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> BuildResult:
    """
    Build complete Care Management layer.
    """

    runtime: Optional[CareManagementRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

        care_programs = build_care_programs(runtime, datasets)
        case_management_worklist = build_case_management_worklist(runtime, care_programs)
        transitions_of_care = build_transitions_of_care(runtime, datasets)
        disease_management = build_disease_management(runtime, datasets)
        outreach_tracking = build_outreach_tracking(runtime, case_management_worklist)
        program_effectiveness = build_program_effectiveness(runtime, care_programs)

        output_assets: Dict[str, pd.DataFrame] = {
            "care_programs": care_programs,
            "case_management_worklist": case_management_worklist,
            "transitions_of_care": transitions_of_care,
            "disease_management": disease_management,
            "outreach_tracking": outreach_tracking,
            "program_effectiveness": program_effectiveness,
        }

        write_care_management_outputs(runtime, output_assets)

        add_audit_record(
            runtime=runtime,
            step_name="build_care_management_layer",
            status=STATUS_SUCCESS,
            message="Care Management completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, output_assets)

        logger.info("=" * 80)
        logger.info("MedFabric Care Management completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="care_management",
            status=STATUS_SUCCESS,
            message="Care Management completed successfully.",
            row_count=sum(len(df) for df in output_assets.values()),
            column_count=sum(len(df.columns) for df in output_assets.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Care Management failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_care_management_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="care_management",
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
        python -m src.analytics_platform.care_management.build_care_management_layer
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