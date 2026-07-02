###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/build_modeling_layer.py
#
# Layer:
#     Layer 2D - Model Training & Scoring
#
# Purpose:
#     Builds the MedFabric Modeling layer.
#
#     This module trains and scores configured predictive models using only
#     Feature Store datasets. It produces model artifacts, scoring datasets,
#     model metrics, feature importance outputs, metadata, validation results,
#     audit records, and execution summaries.
#
# Architectural Rule:
#     Modeling is independent of the Analytics Platform.
#
#     This file must NOT read from:
#
#         data/analytics_platform/
#
#     This file must NOT import from:
#
#         src.analytics_platform
#
#     Predictive Analytics will later consume outputs from:
#
#         data/scoring/
#         models/
#         data/metadata/
#
# Important Modeling Standard:
#     Columns used to generate synthetic target labels must NOT be used as model
#     training features. Keeping the target source column in the feature matrix
#     creates target leakage and causes unrealistic model metrics such as:
#
#         accuracy = 1.0
#         precision = 1.0
#         recall = 1.0
#         roc_auc = 1.0
#
#     This file explicitly excludes target source columns from model features.
#
# Inputs:
#     config/modeling/modeling.yaml
#     data/feature_store/*.parquet
#
# Outputs:
#     data/modeling/
#     data/scoring/
#     data/metadata/
#     data/audit/
#     models/
#
# Run:
#     python -m src.modeling.build_modeling_layer
#
###############################################################################

from __future__ import annotations

import logging
import os
import pickle
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/modeling/modeling.yaml"

DEFAULT_LAYER_NAME = "Layer 2D - Model Training & Scoring"

DEFAULT_DOMAIN_NAME = "Modeling"

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
class ModelingRuntime:
    """
    Runtime context for one Modeling layer execution.

    Purpose
    -------
    Holds run-level metadata, loaded configuration, logger, and in-memory
    records used to build audit, validation, dataset, model, and feature
    metadata outputs.

    Notes
    -----
    This runtime is intentionally local to the Modeling layer because this file
    is independent of the Analytics Platform package.
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
    model_records: List[Dict[str, Any]] = field(default_factory=list)
    feature_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildResult:
    """
    Standard result returned by the Modeling layer.
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
    Create directory when it does not already exist.
    """

    path.mkdir(parents=True, exist_ok=True)


def safe_string(value: Any) -> str:
    """
    Convert value to string safely for metadata outputs.
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
    Configure Modeling layer logging.

    Parameters
    ----------
    project_root:
        Repository root.

    config:
        Loaded Modeling YAML configuration.

    run_id:
        Current Modeling layer run identifier.

    Returns
    -------
    logging.Logger
        Configured Modeling logger.

    Raises
    ------
    OSError
        Raised when the configured log directory or log file cannot be created.
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
        "modeling_layer.log",
    )

    logger = logging.getLogger("medfabric.modeling")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [RUN_ID=%(run_id)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RunIdFilter(logging.Filter):
        """
        Inject run_id into every log record.
        """

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
    logger.info("MedFabric Modeling Layer started")
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

    Parameters
    ----------
    config_path:
        Path to YAML configuration file.

    Returns
    -------
    dict
        Loaded YAML configuration.

    Raises
    ------
    FileNotFoundError
        Raised when the configuration file does not exist.

    ValueError
        Raised when the configuration is empty or is not a YAML mapping.
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
    Validate required Modeling configuration sections.

    Parameters
    ----------
    config:
        Loaded Modeling YAML configuration.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when required configuration sections are missing or invalid.
    """

    errors: List[str] = []

    required_sections = [
        "modeling",
        "logging",
        "paths",
        "join_keys",
        "feature_matrix",
        "modeling_defaults",
        "models",
        "risk_tiers",
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
        for section in [
            "inputs",
            "outputs",
            "model_outputs",
            "metadata_outputs",
            "audit_outputs",
        ]:
            if section not in paths:
                errors.append(f"Missing required configuration section: paths.{section}")

    output_format = config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if errors:
        raise ValueError(
            "Modeling configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> ModelingRuntime:
    """
    Initialize Modeling runtime.

    Parameters
    ----------
    config_path_raw:
        Relative or absolute path to Modeling YAML configuration.

    Returns
    -------
    ModelingRuntime
        Initialized runtime context.

    Raises
    ------
    Exception
        Propagates configuration, path, and logging setup failures.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    modeling_config = config.get("modeling", {})
    layer_name = modeling_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = modeling_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = ModelingRuntime(
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
        message="Modeling runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit, Validation, Metadata Records
###############################################################################

def add_audit_record(
    runtime: ModelingRuntime,
    step_name: str,
    status: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Add audit record to runtime.

    Parameters
    ----------
    runtime:
        Modeling runtime.

    step_name:
        Logical processing step.

    status:
        Step status.

    message:
        Human-readable audit message.

    row_count:
        Optional row count.

    output_path:
        Optional output path.

    Returns
    -------
    None
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
    runtime: ModelingRuntime,
    dataset_name: str,
    rule_name: str,
    status: str,
    message: str,
    failed_count: Optional[int] = None,
) -> None:
    """
    Add validation record to runtime.
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
    runtime: ModelingRuntime,
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


def add_model_record(
    runtime: ModelingRuntime,
    model_key: str,
    model_name: str,
    status: str,
    model_path: str,
    scoring_path: str,
    row_count: int,
    metric_summary: Dict[str, Any],
) -> None:
    """
    Add model registry record.
    """

    runtime.model_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "domain_name": runtime.domain_name,
            "model_key": model_key,
            "model_name": model_name,
            "status": status,
            "model_path": model_path,
            "scoring_path": scoring_path,
            "scored_row_count": row_count,
            "metric_summary_json": safe_string(metric_summary),
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_feature_records(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
    member_key: str,
) -> None:
    """
    Add feature registry records from modeling feature matrix.
    """

    for column in feature_matrix.columns:
        if column == member_key:
            continue

        runtime.feature_records.append(
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "feature_name": column,
                "data_type": str(feature_matrix[column].dtype),
                "non_null_count": int(feature_matrix[column].notna().sum()),
                "null_count": int(feature_matrix[column].isna().sum()),
                "event_timestamp_utc": utc_now().isoformat(),
            }
        )


###############################################################################
# Dataset IO
###############################################################################

def get_output_format(runtime: ModelingRuntime) -> str:
    """
    Return configured Modeling output format.
    """

    return runtime.config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
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


def load_input_datasets(runtime: ModelingRuntime) -> Dict[str, pd.DataFrame]:
    """
    Load configured Feature Store inputs only.

    Notes
    -----
    This function must only load Feature Store inputs. Modeling must remain
    independent from Analytics Platform outputs.
    """

    logger = runtime.logger
    inputs_config = runtime.config.get("paths", {}).get("inputs", {})
    datasets: Dict[str, pd.DataFrame] = {}

    logger.info("START: Load Modeling Feature Store input datasets")

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
                dataset_type="feature_store_input",
                status=STATUS_SUCCESS,
                path=str(dataset_path),
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                message="Feature Store input dataset loaded successfully.",
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

    logger.info("COMPLETE: Load Modeling input datasets | Count: %s", len(datasets))

    return datasets


###############################################################################
# Validation Helpers
###############################################################################

def validate_not_empty(
    runtime: ModelingRuntime,
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
    runtime: ModelingRuntime,
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
    runtime: ModelingRuntime,
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
    runtime: ModelingRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> None:
    """
    Validate loaded input datasets.
    """

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    for dataset_name, dataframe in datasets.items():
        validate_not_empty(runtime, dataframe, dataset_name)

        if member_key in dataframe.columns:
            validate_member_key_not_null(
                runtime=runtime,
                dataframe=dataframe,
                dataset_name=dataset_name,
                member_key=member_key,
            )

    add_audit_record(
        runtime=runtime,
        step_name="validate_inputs",
        status=STATUS_SUCCESS,
        message="Modeling input validation completed.",
    )


###############################################################################
# Business Rule Helpers
###############################################################################

def apply_operator(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Apply configured comparison operator to a pandas Series.
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


def clean_duplicate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate column names created by joins.
    """

    return dataframe.loc[:, ~dataframe.columns.duplicated()].copy()


###############################################################################
# Feature Matrix
###############################################################################

def prepare_dataset_for_join(
    dataframe: pd.DataFrame,
    dataset_name: str,
    member_key: str,
    exclude_columns: List[str],
    existing_columns: List[str],
) -> pd.DataFrame:
    """
    Prepare one dataset for member-level feature matrix joining.

    Purpose
    -------
    Deduplicate the dataset to member grain, drop configured excluded columns,
    and prefix colliding non-key columns with dataset name.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    dataset_name:
        Logical dataset name from configuration.

    member_key:
        Member identifier column.

    exclude_columns:
        Columns excluded from the joined modeling matrix.

    existing_columns:
        Columns already present in the modeling matrix.

    Returns
    -------
    pandas.DataFrame
        Prepared dataframe ready to join.

    Raises
    ------
    ValueError
        Raised when the dataset does not contain the member key.
    """

    if member_key not in dataframe.columns:
        raise ValueError(f"Dataset {dataset_name} missing member key: {member_key}")

    prepared = dataframe.drop_duplicates(subset=[member_key]).copy()

    drop_columns = [column for column in exclude_columns if column in prepared.columns]

    if drop_columns:
        prepared = prepared.drop(columns=drop_columns)

    rename_map: Dict[str, str] = {}

    for column in prepared.columns:
        if column == member_key:
            continue

        if column in existing_columns:
            rename_map[column] = f"{dataset_name}__{column}"

    if rename_map:
        prepared = prepared.rename(columns=rename_map)

    return prepared


def build_feature_matrix(
    runtime: ModelingRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build unified modeling feature matrix from Feature Store datasets.

    Notes
    -----
    This function does not remove target leakage columns because target rules
    are model-specific. Leakage columns are excluded later when final training
    feature columns are selected.
    """

    logger = runtime.logger
    config = runtime.config.get("feature_matrix", {})

    if not bool(config.get("enabled", True)):
        raise ValueError("feature_matrix.enabled must be true for Modeling layer.")

    member_key = config.get("member_key", "member_id")
    base_dataset_name = config.get("base_dataset", "risk_features")
    join_datasets = config.get("join_datasets", [])
    exclude_columns = config.get("exclude_columns", [])

    if base_dataset_name not in datasets:
        raise ValueError(f"Feature matrix base dataset missing: {base_dataset_name}")

    base_df = prepare_dataset_for_join(
        dataframe=datasets[base_dataset_name],
        dataset_name=base_dataset_name,
        member_key=member_key,
        exclude_columns=exclude_columns,
        existing_columns=[],
    )

    matrix = base_df.copy()

    logger.info(
        "START: Build modeling feature matrix | Base: %s | Rows: %s | Columns: %s",
        base_dataset_name,
        len(matrix),
        len(matrix.columns),
    )

    for dataset_name in join_datasets:
        if dataset_name not in datasets:
            logger.warning("Skipping missing optional feature dataset: %s", dataset_name)
            continue

        join_df = prepare_dataset_for_join(
            dataframe=datasets[dataset_name],
            dataset_name=dataset_name,
            member_key=member_key,
            exclude_columns=exclude_columns,
            existing_columns=list(matrix.columns),
        )

        matrix = matrix.merge(join_df, on=member_key, how="left")
        matrix = clean_duplicate_columns(matrix)

        logger.info(
            "Joined feature dataset: %s | Rows: %s | Columns: %s",
            dataset_name,
            len(matrix),
            len(matrix.columns),
        )

    matrix["modeling_layer_run_id"] = runtime.run_id
    matrix["modeling_layer_built_at_utc"] = utc_now().isoformat()

    add_feature_records(runtime, matrix, member_key)

    add_dataset_record(
        runtime=runtime,
        dataset_name="modeling_feature_matrix",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(matrix),
        column_count=len(matrix.columns),
        message="Unified modeling feature matrix built successfully.",
    )

    logger.info(
        "COMPLETE: Build modeling feature matrix | Rows: %s | Columns: %s",
        len(matrix),
        len(matrix.columns),
    )

    return matrix


###############################################################################
# Target Generation and Leakage Control
###############################################################################

def generate_target_for_model(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
    model_key: str,
    model_config: Dict[str, Any],
) -> pd.Series:
    """
    Generate binary target variable using YAML rule.

    Parameters
    ----------
    runtime:
        Modeling runtime.

    feature_matrix:
        Unified modeling matrix.

    model_key:
        Model key from configuration.

    model_config:
        Model configuration.

    Returns
    -------
    pandas.Series
        Binary target label.

    Raises
    ------
    ValueError
        Raised when target source column is missing.

    Notes
    -----
    The target rule is defined in config/modeling/modeling.yaml.
    """

    target_rule = model_config.get("target_rule", {})
    source_column = target_rule.get("source_column")
    operator = target_rule.get("operator")
    value = target_rule.get("value")

    if source_column not in feature_matrix.columns:
        source_dataset = target_rule.get("source_dataset")
        prefixed_column = f"{source_dataset}__{source_column}"

        if prefixed_column in feature_matrix.columns:
            source_column = prefixed_column
        else:
            raise ValueError(
                f"Target source column missing for model {model_key}: {source_column}"
            )

    mask = apply_operator(feature_matrix[source_column], operator, value)

    return mask.astype(int)


def get_target_source_columns(models_config: Dict[str, Any]) -> List[str]:
    """
    Collect target source columns used to generate labels.

    Purpose
    -------
    Identify columns that directly define target labels so they can be excluded
    from model training features.

    Parameters
    ----------
    models_config:
        Configured models from modeling.yaml.

    Returns
    -------
    list[str]
        Unique target source columns and possible dataset-prefixed variants.

    Notes
    -----
    This is the core target-leakage prevention logic.
    """

    source_columns: List[str] = []

    for model_config in models_config.values():
        if not bool(model_config.get("enabled", True)):
            continue

        target_rule = model_config.get("target_rule", {})
        source_column = target_rule.get("source_column")
        source_dataset = target_rule.get("source_dataset")

        if source_column:
            source_columns.append(source_column)

        if source_dataset and source_column:
            source_columns.append(f"{source_dataset}__{source_column}")

    return sorted(set(source_columns))


def build_target_summary(
    runtime: ModelingRuntime,
    target_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build target distribution summary for all enabled models.
    """

    rows: List[Dict[str, Any]] = []

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    target_columns = [column for column in target_frame.columns if column != member_key]

    for column in target_columns:
        value_counts = target_frame[column].value_counts(dropna=False).to_dict()
        total_count = int(len(target_frame))
        positive_count = int(value_counts.get(1, 0))
        negative_count = int(value_counts.get(0, 0))
        positive_rate = positive_count / total_count if total_count else 0.0

        rows.append(
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "target_column": column,
                "total_count": total_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": positive_rate,
                "event_timestamp_utc": utc_now().isoformat(),
            }
        )

        add_validation_record(
            runtime=runtime,
            dataset_name="target_summary",
            rule_name=f"target_distribution:{column}",
            status=(
                STATUS_SUCCESS
                if positive_count > 0 and negative_count > 0
                else STATUS_WARNING
            ),
            message=(
                f"Target distribution for {column}: "
                f"positive={positive_count}, negative={negative_count}, "
                f"positive_rate={positive_rate:.6f}"
            ),
            failed_count=0 if positive_count > 0 and negative_count > 0 else 1,
        )

    return pd.DataFrame(rows)


###############################################################################
# Preprocessing and Model Training
###############################################################################

def get_feature_columns(
    dataframe: pd.DataFrame,
    member_key: str,
    target_columns: List[str],
    leakage_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Select eligible feature columns.

    Parameters
    ----------
    dataframe:
        Modeling dataframe containing features and targets.

    member_key:
        Member identifier column.

    target_columns:
        Generated target columns.

    leakage_columns:
        Columns used to create target labels and therefore excluded from
        training features.

    Returns
    -------
    list[str]
        Eligible model feature columns.

    Notes
    -----
    Excluding leakage columns prevents models from directly learning the label
    generation rule.
    """

    excluded = set(target_columns)
    excluded.add(member_key)
    excluded.add("modeling_layer_run_id")
    excluded.add("modeling_layer_built_at_utc")

    for column in leakage_columns or []:
        excluded.add(column)

    feature_columns = [
        column for column in dataframe.columns
        if column not in excluded
    ]

    return feature_columns


def build_preprocessor(
    X: pd.DataFrame,
    defaults: Dict[str, Any],
) -> ColumnTransformer:
    """
    Build sklearn preprocessing transformer.
    """

    preprocessing_config = defaults.get("preprocessing", {})

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]

    numeric_steps: List[Tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config.get(
                    "numeric_imputation_strategy",
                    "median",
                )
            ),
        )
    ]

    if bool(preprocessing_config.get("scale_numeric_features", False)):
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: List[Tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config.get(
                    "categorical_imputation_strategy",
                    "most_frequent",
                )
            ),
        )
    ]

    if bool(preprocessing_config.get("one_hot_encode_categorical_features", True)):
        categorical_steps.append(
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        )

    transformers: List[Tuple[str, Pipeline, List[str]]] = []

    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))

    if categorical_features:
        transformers.append(
            ("categorical", Pipeline(categorical_steps), categorical_features)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_classifier(defaults: Dict[str, Any]) -> RandomForestClassifier:
    """
    Build configured classifier.

    Current supported algorithm:
        random_forest_classifier
    """

    algorithm = defaults.get("algorithm", {})
    algorithm_name = algorithm.get("name", "random_forest_classifier")

    if algorithm_name != "random_forest_classifier":
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    return RandomForestClassifier(
        n_estimators=int(algorithm.get("n_estimators", 100)),
        max_depth=algorithm.get("max_depth"),
        min_samples_split=int(algorithm.get("min_samples_split", 2)),
        min_samples_leaf=int(algorithm.get("min_samples_leaf", 1)),
        random_state=int(defaults.get("random_state", 42)),
        n_jobs=-1,
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    """

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def assign_risk_tiers(
    scores: pd.Series,
    risk_tiers: Dict[str, Any],
) -> pd.Series:
    """
    Assign configured risk tier labels.
    """

    output = pd.Series("Unassigned", index=scores.index)

    for _, tier_config in risk_tiers.items():
        min_value = tier_config.get("min_value")
        max_value = tier_config.get("max_value")
        label = tier_config.get("label")

        mask = (scores >= min_value) & (scores <= max_value)
        output.loc[mask] = label

    return output


def extract_feature_importance(
    pipeline: Pipeline,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Extract feature importance from trained model pipeline.
    """

    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature_name", "importance"])

    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
        feature_names = [str(name) for name in transformed_feature_names]
    except Exception:
        feature_names = feature_columns

    importances = model.feature_importances_
    length = min(len(feature_names), len(importances))

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names[:length],
            "importance": importances[:length],
        }
    ).sort_values("importance", ascending=False)

    return importance_df


###############################################################################
# Model Output Path Helpers
###############################################################################

def get_model_output_config(
    runtime: ModelingRuntime,
    model_key: str,
) -> Dict[str, Any]:
    """
    Return configured output paths for one model.
    """

    return runtime.config.get("paths", {}).get("model_outputs", {}).get(model_key, {})


def get_output_path(
    runtime: ModelingRuntime,
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
        raw_path = f"data/{output_group}/{output_name}"

    return normalize_path(runtime.project_root, raw_path)


def save_pickle_object(obj: Any, path: Path) -> None:
    """
    Save Python object as pickle.
    """

    ensure_directory(path.parent)

    with path.open("wb") as file:
        pickle.dump(obj, file)


###############################################################################
# Model Training and Scoring
###############################################################################

def train_and_score_model(
    runtime: ModelingRuntime,
    modeling_frame: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    model_key: str,
    model_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train and score one configured model.
    """

    logger = runtime.logger
    defaults = runtime.config.get("modeling_defaults", {})
    risk_tiers = runtime.config.get("risk_tiers", {})

    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    X = modeling_frame[feature_columns].copy()
    y = modeling_frame[target_column].astype(int)

    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"Target for model {model_key} has fewer than two classes. "
            f"Target column: {target_column}"
        )

    test_size = float(defaults.get("test_size", 0.20))
    random_state = int(defaults.get("random_state", 42))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train, defaults)
    classifier = build_classifier(defaults)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )

    logger.info("START: Train model | Model: %s | Rows: %s", model_key, len(X_train))

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    metrics = calculate_metrics(y_test, y_pred, y_score)

    logger.info("COMPLETE: Train model | Model: %s | Metrics: %s", model_key, metrics)

    full_scores = pipeline.predict_proba(X)[:, 1]
    full_predictions = pipeline.predict(X)

    score_column = model_config.get("score_column", f"{model_key}_score")
    prediction_column = model_config.get(
        "prediction_column",
        f"{model_key}_prediction",
    )
    risk_tier_column = model_config.get(
        "risk_tier_column",
        f"{model_key}_risk_tier",
    )

    scoring_df = pd.DataFrame(
        {
            member_key: modeling_frame[member_key],
            score_column: full_scores,
            prediction_column: full_predictions,
        }
    )

    scoring_df[risk_tier_column] = assign_risk_tiers(
        pd.Series(full_scores),
        risk_tiers,
    )

    scoring_df["model_key"] = model_key
    scoring_df["model_name"] = model_config.get("model_name", model_key)
    scoring_df["modeling_layer_run_id"] = runtime.run_id
    scoring_df["scored_at_utc"] = utc_now().isoformat()

    metrics_df = pd.DataFrame(
        [
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "model_key": model_key,
                "model_name": model_config.get("model_name", model_key),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "event_timestamp_utc": utc_now().isoformat(),
            }
            for metric_name, metric_value in metrics.items()
        ]
    )

    feature_importance_df = extract_feature_importance(pipeline, feature_columns)
    feature_importance_df["run_id"] = runtime.run_id
    feature_importance_df["model_key"] = model_key
    feature_importance_df["model_name"] = model_config.get("model_name", model_key)
    feature_importance_df["event_timestamp_utc"] = utc_now().isoformat()

    output_config = get_model_output_config(runtime, model_key)

    model_path = normalize_path(runtime.project_root, output_config.get("model_path"))

    metrics_path = output_path_with_format(
        normalize_path(runtime.project_root, output_config.get("metrics_path")),
        "parquet",
    )

    feature_importance_path = output_path_with_format(
        normalize_path(
            runtime.project_root,
            output_config.get("feature_importance_path"),
        ),
        "parquet",
    )

    scoring_path = output_path_with_format(
        normalize_path(runtime.project_root, output_config.get("scoring_path")),
        get_output_format(runtime),
    )

    save_pickle_object(pipeline, model_path)
    write_dataset(metrics_df, metrics_path, "parquet")
    write_dataset(feature_importance_df, feature_importance_path, "parquet")
    write_dataset(scoring_df, scoring_path, get_output_format(runtime))

    add_model_record(
        runtime=runtime,
        model_key=model_key,
        model_name=model_config.get("model_name", model_key),
        status=STATUS_SUCCESS,
        model_path=str(model_path),
        scoring_path=str(scoring_path),
        row_count=len(scoring_df),
        metric_summary=metrics,
    )

    add_dataset_record(
        runtime=runtime,
        dataset_name=f"{model_key}_scores",
        dataset_type="scoring_output",
        status=STATUS_SUCCESS,
        path=str(scoring_path),
        row_count=len(scoring_df),
        column_count=len(scoring_df.columns),
        message=f"Scoring output created successfully for model {model_key}.",
    )

    add_audit_record(
        runtime=runtime,
        step_name=f"train_and_score_model:{model_key}",
        status=STATUS_SUCCESS,
        message=f"Model trained and scored successfully: {model_key}",
        row_count=len(scoring_df),
        output_path=str(scoring_path),
    )

    return scoring_df, metrics_df, feature_importance_df


def train_and_score_enabled_models(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Train and score all enabled models.

    Notes
    -----
    This function now excludes target source columns from feature_columns.
    That prevents target leakage from synthetic label generation rules.
    """

    logger = runtime.logger
    models_config = runtime.config.get("models", {})
    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")

    modeling_frame = feature_matrix.copy()
    target_columns: List[str] = []

    logger.info("START: Generate model targets")

    for model_key, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            continue

        target_column = model_config.get("target_column")

        modeling_frame[target_column] = generate_target_for_model(
            runtime=runtime,
            feature_matrix=modeling_frame,
            model_key=model_key,
            model_config=model_config,
        )

        target_columns.append(target_column)

    target_summary = build_target_summary(
        runtime,
        modeling_frame[[member_key] + target_columns],
    )

    target_summary_path = output_path_with_format(
        get_output_path(runtime, "outputs", "modeling_target_summary"),
        get_output_format(runtime),
    )

    write_dataset(target_summary, target_summary_path, get_output_format(runtime))

    logger.info("COMPLETE: Generate model targets | Targets: %s", len(target_columns))

    leakage_columns = get_target_source_columns(models_config)

    logger.info(
        "Target leakage columns excluded from training features: %s",
        leakage_columns,
    )

    add_validation_record(
        runtime=runtime,
        dataset_name="modeling_feature_matrix",
        rule_name="target_leakage_columns_excluded",
        status=STATUS_SUCCESS,
        message=(
            "Target source columns excluded from model training features: "
            f"{leakage_columns}"
        ),
        failed_count=0,
    )

    feature_columns = get_feature_columns(
        dataframe=modeling_frame,
        member_key=member_key,
        target_columns=target_columns,
        leakage_columns=leakage_columns,
    )

    logger.info(
        "START: Train and score enabled models | Feature Columns: %s",
        len(feature_columns),
    )

    scoring_outputs: Dict[str, pd.DataFrame] = {}

    for model_key, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            logger.info("SKIP model disabled: %s", model_key)
            continue

        target_column = model_config.get("target_column")

        scoring_df, _, _ = train_and_score_model(
            runtime=runtime,
            modeling_frame=modeling_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key=model_key,
            model_config=model_config,
        )

        scoring_outputs[model_key] = scoring_df

    logger.info("COMPLETE: Train and score enabled models | Count: %s", len(scoring_outputs))

    return scoring_outputs


###############################################################################
# Metadata Outputs
###############################################################################

def build_dataset_inventory(runtime: ModelingRuntime) -> pd.DataFrame:
    """
    Build dataset inventory.
    """

    return pd.DataFrame(runtime.dataset_records)


def build_column_dictionary(
    runtime: ModelingRuntime,
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


def build_model_registry(runtime: ModelingRuntime) -> pd.DataFrame:
    """
    Build model registry.
    """

    return pd.DataFrame(runtime.model_records)


def build_feature_registry(runtime: ModelingRuntime) -> pd.DataFrame:
    """
    Build feature registry.
    """

    return pd.DataFrame(runtime.feature_records)


def build_execution_summary(
    runtime: ModelingRuntime,
    scoring_outputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build Modeling execution summary.
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
        "model_count": len(scoring_outputs),
        "scored_dataset_count": len(scoring_outputs),
        "audit_record_count": len(runtime.audit_records),
        "validation_record_count": len(runtime.validation_records),
        "failed_validation_count": failed_validation_count,
        "dataset_record_count": len(runtime.dataset_records),
        "model_record_count": len(runtime.model_records),
        "feature_record_count": len(runtime.feature_records),
        "status": STATUS_SUCCESS if failed_validation_count == 0 else STATUS_WARNING,
    }

    return pd.DataFrame([summary])


###############################################################################
# Output Writing
###############################################################################

def write_core_modeling_outputs(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
) -> None:
    """
    Write core Modeling outputs.
    """

    output_format = get_output_format(runtime)

    feature_matrix_path = output_path_with_format(
        get_output_path(runtime, "outputs", "modeling_feature_matrix"),
        output_format,
    )

    write_dataset(feature_matrix, feature_matrix_path, output_format)

    add_audit_record(
        runtime=runtime,
        step_name="write_output:modeling_feature_matrix",
        status=STATUS_SUCCESS,
        message="Modeling feature matrix written successfully.",
        row_count=len(feature_matrix),
        output_path=str(feature_matrix_path),
    )


def write_metadata_and_audit_outputs(
    runtime: ModelingRuntime,
    scoring_outputs: Dict[str, pd.DataFrame],
) -> None:
    """
    Write Modeling metadata and audit outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "modeling_dataset_inventory": build_dataset_inventory(runtime),
        "modeling_column_dictionary": build_column_dictionary(runtime, scoring_outputs),
        "modeling_model_registry": build_model_registry(runtime),
        "modeling_feature_registry": build_feature_registry(runtime),
    }

    audit_assets = {
        "modeling_audit_records": pd.DataFrame(runtime.audit_records),
        "modeling_validation_results": pd.DataFrame(runtime.validation_records),
        "modeling_execution_summary": build_execution_summary(runtime, scoring_outputs),
    }

    for output_name, dataframe in metadata_assets.items():
        output_path = output_path_with_format(
            get_output_path(runtime, "metadata_outputs", output_name),
            output_format,
        )

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote metadata output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

    for output_name, dataframe in audit_assets.items():
        output_path = output_path_with_format(
            get_output_path(runtime, "audit_outputs", output_name),
            output_format,
        )

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

def build_modeling_layer(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build complete Layer 2D Model Training & Scoring layer.
    """

    runtime: Optional[ModelingRuntime] = None

    try:
        runtime = initialize_runtime(config_path)
        logger = runtime.logger

        logger.info("Configuration path: %s", runtime.config_path)
        logger.info("Architectural check: Modeling consumes Feature Store only.")

        datasets = load_input_datasets(runtime)
        validate_inputs(runtime, datasets)

        feature_matrix = build_feature_matrix(runtime, datasets)
        write_core_modeling_outputs(runtime, feature_matrix)

        scoring_outputs = train_and_score_enabled_models(runtime, feature_matrix)

        add_audit_record(
            runtime=runtime,
            step_name="build_modeling_layer",
            status=STATUS_SUCCESS,
            message="Modeling Layer completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, scoring_outputs)

        logger.info("=" * 80)
        logger.info("MedFabric Modeling Layer completed successfully")
        logger.info("Models trained and scored: %s", len(scoring_outputs))
        logger.info("=" * 80)

        return BuildResult(
            name="modeling",
            status=STATUS_SUCCESS,
            message="Modeling Layer completed successfully.",
            row_count=sum(len(df) for df in scoring_outputs.values()),
            column_count=sum(len(df.columns) for df in scoring_outputs.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Modeling Layer failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_modeling_layer",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_metadata_and_audit_outputs(runtime, {})
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="modeling",
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
        python -m src.modeling.build_modeling_layer
    """

    config_path = os.environ.get(
        "MEDFABRIC_MODELING_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_modeling_layer(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Modeling Layer failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()