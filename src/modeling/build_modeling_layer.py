###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/build_modeling_layer.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Lightweight orchestrator for the MedFabric Modeling layer.
#
#     This builder coordinates the Modeling Framework components:
#
#       - Feature Store input loading
#       - Modeling feature matrix construction
#       - Target Builder Framework
#       - Multi-algorithm training framework
#       - Champion model selection
#       - Population scoring
#       - Feature importance extraction
#       - Model registry creation
#       - Metadata and audit output writing
#
# Architectural Rules:
#     - Modeling consumes Feature Store outputs only.
#     - Modeling must not read from data/analytics_platform/.
#     - Modeling must not import from src.analytics_platform.
#     - Modeling must use the global MedFabric PipelineContext run_id.
#     - The builder orchestrates only; business logic belongs in framework
#       modules under src/modeling/.
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
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.common.pipeline_context import create_pipeline_context
from src.modeling.evaluation.feature_importance import (
    build_feature_importance_output,
)
from src.modeling.registry.model_registry import (
    build_model_registry_dataframe,
    build_model_registry_record,
)
from src.modeling.scoring.scorer import score_population
from src.modeling.targets.leakage_detection import get_target_output_column
from src.modeling.targets.target_builder import build_targets_for_enabled_models
from src.modeling.training.trainer import train_model_candidates


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/modeling/modeling.yaml"

DEFAULT_LAYER_NAME = "Layer 2D - Enterprise Modeling Framework"

DEFAULT_DOMAIN_NAME = "Modeling"

DEFAULT_OUTPUT_FORMAT = "parquet"

SUPPORTED_OUTPUT_FORMATS = {"parquet", "csv", "json"}

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_WARNING = "WARNING"
STATUS_SKIPPED = "SKIPPED"


###############################################################################
# Runtime Objects
###############################################################################

@dataclass
class ModelingRuntime:
    """
    Runtime context for one Modeling layer execution.

    Important
    ---------
    run_id must come from the global MedFabric PipelineContext. Modeling should
    not generate an isolated local run_id because its outputs need to align with
    the enterprise pipeline audit trail.
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
    model_registry_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildResult:
    """
    Standard result returned by the Modeling builder.
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
    Create directory when missing.
    """

    path.mkdir(parents=True, exist_ok=True)


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Ensure an output path has the configured file suffix.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


###############################################################################
# Logging
###############################################################################

def configure_logging(
    project_root: Path,
    config: Dict[str, Any],
    run_id: str,
) -> logging.Logger:
    """
    Configure Modeling Framework logging.

    Notes
    -----
    This logger writes to the Modeling module log while using the global
    PipelineContext run_id.
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
        Inject global run_id into every Modeling log record.
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
    logger.info("MedFabric Modeling Framework started")
    logger.info("=" * 80)
    logger.info("Global Run ID: %s", run_id)
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

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required Modeling configuration sections.
    """

    required_sections = [
        "modeling",
        "logging",
        "paths",
        "join_keys",
        "feature_matrix",
        "modeling_defaults",
        "training",
        "models",
        "risk_tiers",
        "validation",
        "metadata",
        "audit",
    ]

    missing = [section for section in required_sections if section not in config]

    if missing:
        raise ValueError(f"Missing required Modeling config sections: {missing}")

    output_format = config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
    )

    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output_format '{output_format}'. "
            f"Supported: {sorted(SUPPORTED_OUTPUT_FORMATS)}"
        )

    paths = config.get("paths", {})

    required_path_sections = [
        "inputs",
        "outputs",
        "model_outputs",
        "metadata_outputs",
        "audit_outputs",
    ]

    missing_path_sections = [
        section for section in required_path_sections
        if section not in paths
    ]

    if missing_path_sections:
        raise ValueError(
            f"Missing required paths sections: {missing_path_sections}"
        )


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> ModelingRuntime:
    """
    Initialize Modeling runtime using global PipelineContext.

    Important
    ---------
    Modeling must not generate its own local run_id. The run_id must come from
    PipelineContext so all MedFabric layers share the same audit and metadata
    lineage when executed as part of the enterprise pipeline.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)

    pipeline_context = create_pipeline_context(
        pipeline_name="MedFabric Modeling Framework",
    )

    run_id = pipeline_context.run_id

    config = load_yaml_config(config_path)
    validate_config(config)

    modeling_config = config.get("modeling", {})

    layer_name = modeling_config.get("layer_name", DEFAULT_LAYER_NAME)
    domain_name = modeling_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(
        project_root=project_root,
        config=config,
        run_id=run_id,
    )

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

    logger.info("Global pipeline run ID resolved from PipelineContext: %s", run_id)

    add_audit_record(
        runtime=runtime,
        step_name="initialize_runtime",
        status=STATUS_SUCCESS,
        message="Modeling runtime initialized successfully using PipelineContext.",
    )

    return runtime


###############################################################################
# Audit / Validation / Metadata Records
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
    Add a Modeling audit record.
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
    failed_count: int = 0,
) -> None:
    """
    Add a Modeling validation record.
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
    Add a Modeling dataset inventory record.
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


###############################################################################
# IO Helpers
###############################################################################

def get_output_format(runtime: ModelingRuntime) -> str:
    """
    Return configured Modeling output format.
    """

    return runtime.config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
    )


def read_dataset(path: Path, file_format: str) -> pd.DataFrame:
    """
    Read dataset from disk.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    if file_format == "parquet":
        return pd.read_parquet(path)

    if file_format == "csv":
        return pd.read_csv(path)

    if file_format == "json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported input file format: {file_format}")


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


def save_pickle_object(obj: Any, path: Path) -> None:
    """
    Save Python object as pickle.
    """

    ensure_directory(path.parent)

    with path.open("wb") as file:
        pickle.dump(obj, file)


def get_output_path(
    runtime: ModelingRuntime,
    output_group: str,
    output_name: str,
) -> Path:
    """
    Resolve a named output path from modeling.yaml.
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


def get_model_output_config(
    runtime: ModelingRuntime,
    model_key: str,
) -> Dict[str, Any]:
    """
    Return configured artifact/scoring output paths for a model.
    """

    output_config = (
        runtime.config
        .get("paths", {})
        .get("model_outputs", {})
        .get(model_key, {})
    )

    if not output_config:
        raise ValueError(f"Missing model output config for model: {model_key}")

    required_keys = [
        "model_path",
        "metrics_path",
        "feature_importance_path",
        "scoring_path",
    ]

    missing = [key for key in required_keys if key not in output_config]

    if missing:
        raise ValueError(
            f"Model output config for {model_key} missing keys: {missing}"
        )

    return output_config


###############################################################################
# Input Loading
###############################################################################

def load_input_datasets(runtime: ModelingRuntime) -> Dict[str, pd.DataFrame]:
    """
    Load configured Feature Store inputs.

    Modeling must consume Feature Store only. Optional inputs are skipped if
    missing. Required inputs fail fast.
    """

    inputs_config = runtime.config.get("paths", {}).get("inputs", {})
    datasets: Dict[str, pd.DataFrame] = {}

    runtime.logger.info("START: Load Modeling Feature Store inputs")

    for dataset_name, dataset_config in inputs_config.items():
        raw_path = dataset_config.get("path")
        file_format = dataset_config.get("format", "parquet")
        required = bool(dataset_config.get("required", True))

        if not raw_path:
            message = f"No path configured for dataset: {dataset_name}"

            if required:
                raise ValueError(message)

            add_audit_record(
                runtime=runtime,
                step_name=f"load_input:{dataset_name}",
                status=STATUS_SKIPPED,
                message=message,
            )
            continue

        dataset_path = normalize_path(runtime.project_root, raw_path)

        try:
            dataframe = read_dataset(dataset_path, file_format)
            datasets[dataset_name] = dataframe

            add_dataset_record(
                runtime=runtime,
                dataset_name=dataset_name,
                dataset_type="feature_store_input",
                status=STATUS_SUCCESS,
                path=str(dataset_path),
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                message="Feature Store input loaded successfully.",
            )

            runtime.logger.info(
                "Loaded %s | Rows: %s | Columns: %s | Path: %s",
                dataset_name,
                len(dataframe),
                len(dataframe.columns),
                dataset_path,
            )

        except Exception as exc:
            if required:
                add_audit_record(
                    runtime=runtime,
                    step_name=f"load_input:{dataset_name}",
                    status=STATUS_FAILED,
                    message=str(exc),
                    output_path=str(dataset_path),
                )
                raise

            add_audit_record(
                runtime=runtime,
                step_name=f"load_input:{dataset_name}",
                status=STATUS_SKIPPED,
                message=str(exc),
                output_path=str(dataset_path),
            )

            runtime.logger.warning(
                "Skipped optional dataset: %s | Reason: %s",
                dataset_name,
                exc,
            )

    runtime.logger.info("COMPLETE: Load inputs | Count: %s", len(datasets))

    return datasets


###############################################################################
# Feature Matrix
###############################################################################

def clean_duplicate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate column names after joins.
    """

    return dataframe.loc[:, ~dataframe.columns.duplicated()].copy()


def prepare_dataset_for_join(
    dataframe: pd.DataFrame,
    dataset_name: str,
    member_key: str,
    exclude_columns: List[str],
    existing_columns: List[str],
) -> pd.DataFrame:
    """
    Prepare one Feature Store dataset for member-level joining.

    This function:
      - validates member key
      - deduplicates to member grain
      - drops configured excluded columns
      - prefixes colliding columns with dataset name
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
    Build unified modeling feature matrix from configured Feature Store datasets.
    """

    config = runtime.config.get("feature_matrix", {})

    if not bool(config.get("enabled", True)):
        raise ValueError("feature_matrix.enabled must be true.")

    member_key = config.get("member_key", "member_id")
    base_dataset_name = config.get("base_dataset", "risk_features")
    join_datasets = config.get("join_datasets", [])
    exclude_columns = config.get("exclude_columns", [])

    if base_dataset_name not in datasets:
        raise ValueError(f"Feature matrix base dataset missing: {base_dataset_name}")

    matrix = prepare_dataset_for_join(
        dataframe=datasets[base_dataset_name],
        dataset_name=base_dataset_name,
        member_key=member_key,
        exclude_columns=exclude_columns,
        existing_columns=[],
    )

    runtime.logger.info(
        "START: Build feature matrix | Base: %s | Rows: %s | Columns: %s",
        base_dataset_name,
        len(matrix),
        len(matrix.columns),
    )

    for dataset_name in join_datasets:
        if dataset_name not in datasets:
            runtime.logger.warning("Skipping missing feature dataset: %s", dataset_name)
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

        runtime.logger.info(
            "Joined %s | Rows: %s | Columns: %s",
            dataset_name,
            len(matrix),
            len(matrix.columns),
        )

    matrix["modeling_layer_run_id"] = runtime.run_id
    matrix["modeling_layer_built_at_utc"] = utc_now().isoformat()

    add_dataset_record(
        runtime=runtime,
        dataset_name="modeling_feature_matrix",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(matrix),
        column_count=len(matrix.columns),
        message="Modeling feature matrix built successfully.",
    )

    runtime.logger.info(
        "COMPLETE: Build feature matrix | Rows: %s | Columns: %s",
        len(matrix),
        len(matrix.columns),
    )

    return matrix


def get_feature_columns(
    dataframe: pd.DataFrame,
    member_key: str,
    target_columns: List[str],
    leakage_columns: List[str],
) -> List[str]:
    """
    Select safe training feature columns.

    Excludes:
      - member key
      - generated target columns
      - target leakage source columns
      - Modeling runtime metadata columns
    """

    excluded = set(target_columns)
    excluded.update(leakage_columns)
    excluded.add(member_key)
    excluded.add("modeling_layer_run_id")
    excluded.add("modeling_layer_built_at_utc")

    return [column for column in dataframe.columns if column not in excluded]


###############################################################################
# Modeling Execution
###############################################################################

def get_algorithms_config(runtime: ModelingRuntime) -> Dict[str, Any]:
    """
    Return algorithm configuration from YAML.

    Algorithms must be YAML-controlled. This function intentionally reads
    config/modeling/modeling.yaml rather than hardcoding algorithm behavior.
    """

    training_config = runtime.config.get("training", {})
    algorithms_config = training_config.get("algorithms")

    if not isinstance(algorithms_config, dict):
        raise ValueError("training.algorithms must be configured in modeling.yaml")

    enabled_algorithms = [
        key for key, value in algorithms_config.items()
        if bool(value.get("enabled", True))
    ]

    if not enabled_algorithms:
        raise ValueError("No enabled algorithms found in training.algorithms.")

    runtime.logger.info("Enabled algorithms: %s", enabled_algorithms)

    return algorithms_config


def run_models(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Execute Target, Training, Scoring, Feature Importance, and Registry workflows.
    """

    models_config = runtime.config.get("models", {})
    modeling_defaults = runtime.config.get("modeling_defaults", {})
    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    risk_tiers_config = runtime.config.get("risk_tiers", {})
    output_format = get_output_format(runtime)

    runtime.logger.info("START: Target Framework")

    target_result = build_targets_for_enabled_models(
        feature_matrix=feature_matrix,
        models_config=models_config,
        run_id=runtime.run_id,
        layer_name=runtime.layer_name,
        domain_name=runtime.domain_name,
    )

    modeling_frame = target_result.modeling_frame
    target_columns = target_result.target_columns
    leakage_columns = target_result.leakage_columns

    target_summary_path = output_path_with_format(
        get_output_path(runtime, "outputs", "modeling_target_summary"),
        output_format,
    )

    write_dataset(target_result.target_summary, target_summary_path, output_format)

    add_dataset_record(
        runtime=runtime,
        dataset_name="modeling_target_summary",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=str(target_summary_path),
        row_count=len(target_result.target_summary),
        column_count=len(target_result.target_summary.columns),
        message="Target summary written successfully.",
    )

    add_validation_record(
        runtime=runtime,
        dataset_name="modeling_feature_matrix",
        rule_name="target_leakage_columns_excluded",
        status=STATUS_SUCCESS,
        message=f"Leakage columns excluded from training: {leakage_columns}",
    )

    feature_columns = get_feature_columns(
        dataframe=modeling_frame,
        member_key=member_key,
        target_columns=target_columns,
        leakage_columns=leakage_columns,
    )

    if not feature_columns:
        raise ValueError("No eligible training feature columns found.")

    runtime.logger.info("Training feature count: %s", len(feature_columns))

    scoring_outputs: Dict[str, pd.DataFrame] = {}
    algorithms_config = get_algorithms_config(runtime)

    for model_key, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            runtime.logger.info("SKIP disabled model: %s", model_key)
            continue

        model_name = model_config.get("model_name", model_key)
        model_type = model_config.get("model_type", "classification")
        target_column = get_target_output_column(model_config)

        runtime.logger.info("=" * 80)
        runtime.logger.info("START MODEL: %s", model_key)
        runtime.logger.info("=" * 80)

        training_result = train_model_candidates(
            dataframe=modeling_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            model_key=model_key,
            model_name=model_name,
            modeling_defaults=modeling_defaults,
            algorithms_config=algorithms_config,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
        )

        runtime.logger.info(
            "Champion selected | Model: %s | Algorithm: %s | Metrics: %s",
            model_key,
            training_result.champion_algorithm_key,
            training_result.champion_metrics,
        )

        scoring_result = score_population(
            dataframe=modeling_frame,
            feature_columns=feature_columns,
            member_key=member_key,
            model_key=model_key,
            model_name=model_name,
            pipeline=training_result.champion_pipeline,
            model_config=model_config,
            risk_tiers_config=risk_tiers_config,
            run_id=runtime.run_id,
        )

        feature_importance_df = build_feature_importance_output(
            pipeline=training_result.champion_pipeline,
            feature_columns=feature_columns,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
        )

        output_config = get_model_output_config(runtime, model_key)

        model_path = normalize_path(
            runtime.project_root,
            output_config.get("model_path"),
        )

        metrics_path = output_path_with_format(
            normalize_path(
                runtime.project_root,
                output_config.get("metrics_path"),
            ),
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
            normalize_path(
                runtime.project_root,
                output_config.get("scoring_path"),
            ),
            output_format,
        )

        save_pickle_object(training_result.champion_pipeline, model_path)
        write_dataset(training_result.metrics_dataframe, metrics_path, "parquet")
        write_dataset(feature_importance_df, feature_importance_path, "parquet")
        write_dataset(scoring_result.scoring_dataframe, scoring_path, output_format)

        scoring_outputs[model_key] = scoring_result.scoring_dataframe

        registry_record = build_model_registry_record(
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            model_type=model_type,
            target_column=target_column,
            champion_algorithm_key=training_result.champion_algorithm_key,
            champion_algorithm_name=training_result.champion_algorithm_name,
            selection_metric=modeling_defaults.get("selection_metric", "roc_auc"),
            champion_metrics=training_result.champion_metrics,
            model_path=str(model_path),
            scoring_path=str(scoring_path),
            metrics_path=str(metrics_path),
            feature_importance_path=str(feature_importance_path),
            scored_row_count=scoring_result.row_count,
        )

        runtime.model_registry_records.append(registry_record)

        add_dataset_record(
            runtime=runtime,
            dataset_name=f"{model_key}_scores",
            dataset_type="scoring_output",
            status=STATUS_SUCCESS,
            path=str(scoring_path),
            row_count=scoring_result.row_count,
            column_count=len(scoring_result.scoring_dataframe.columns),
            message=f"Scoring output written successfully for {model_key}.",
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"model_complete:{model_key}",
            status=STATUS_SUCCESS,
            message=(
                f"Model completed. Champion="
                f"{training_result.champion_algorithm_key}"
            ),
            row_count=scoring_result.row_count,
            output_path=str(scoring_path),
        )

        runtime.logger.info("COMPLETE MODEL: %s", model_key)

    return scoring_outputs


###############################################################################
# Core Output Writing
###############################################################################

def write_core_modeling_outputs(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
) -> None:
    """
    Write core Modeling output datasets.
    """

    output_format = get_output_format(runtime)

    feature_matrix_path = output_path_with_format(
        get_output_path(runtime, "outputs", "modeling_feature_matrix"),
        output_format,
    )

    write_dataset(feature_matrix, feature_matrix_path, output_format)

    add_audit_record(
        runtime=runtime,
        step_name="write_modeling_feature_matrix",
        status=STATUS_SUCCESS,
        message="Modeling feature matrix written successfully.",
        row_count=len(feature_matrix),
        output_path=str(feature_matrix_path),
    )


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
        1 for record in runtime.validation_records
        if record.get("status") == STATUS_FAILED
    )

    return pd.DataFrame(
        [
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "config_path": str(runtime.config_path),
                "start_time_utc": runtime.start_time_utc.isoformat(),
                "end_time_utc": end_time.isoformat(),
                "duration_seconds": duration_seconds,
                "model_count": len(scoring_outputs),
                "scored_dataset_count": len(scoring_outputs),
                "dataset_record_count": len(runtime.dataset_records),
                "model_registry_record_count": len(runtime.model_registry_records),
                "audit_record_count": len(runtime.audit_records),
                "validation_record_count": len(runtime.validation_records),
                "failed_validation_count": failed_validation_count,
                "status": (
                    STATUS_SUCCESS
                    if failed_validation_count == 0
                    else STATUS_WARNING
                ),
            }
        ]
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
        "modeling_dataset_inventory": pd.DataFrame(runtime.dataset_records),
        "modeling_model_registry": build_model_registry_dataframe(
            runtime.model_registry_records
        ),
    }

    audit_assets = {
        "modeling_audit_records": pd.DataFrame(runtime.audit_records),
        "modeling_validation_results": pd.DataFrame(runtime.validation_records),
        "modeling_execution_summary": build_execution_summary(
            runtime,
            scoring_outputs,
        ),
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
    Build complete Modeling Framework.

    Returns
    -------
    BuildResult
        Standard result object used by standalone execution and pipeline
        orchestration.
    """

    runtime: Optional[ModelingRuntime] = None

    try:
        runtime = initialize_runtime(config_path)

        runtime.logger.info("Configuration path: %s", runtime.config_path)
        runtime.logger.info("Architectural check: Modeling consumes Feature Store only.")

        datasets = load_input_datasets(runtime)

        feature_matrix = build_feature_matrix(runtime, datasets)
        write_core_modeling_outputs(runtime, feature_matrix)

        scoring_outputs = run_models(runtime, feature_matrix)

        add_audit_record(
            runtime=runtime,
            step_name="build_modeling_layer",
            status=STATUS_SUCCESS,
            message="Modeling Framework completed successfully.",
        )

        write_metadata_and_audit_outputs(runtime, scoring_outputs)

        runtime.logger.info("=" * 80)
        runtime.logger.info("MedFabric Modeling Framework completed successfully")
        runtime.logger.info("Models trained and scored: %s", len(scoring_outputs))
        runtime.logger.info("=" * 80)

        return BuildResult(
            name="modeling",
            status=STATUS_SUCCESS,
            message="Modeling Framework completed successfully.",
            row_count=sum(len(df) for df in scoring_outputs.values()),
            column_count=sum(len(df.columns) for df in scoring_outputs.values()),
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("Modeling Framework failed")
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
                runtime.logger.error(
                    "Failed to write metadata/audit outputs: %s",
                    audit_exc,
                )

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

    print(f"Modeling Framework failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()