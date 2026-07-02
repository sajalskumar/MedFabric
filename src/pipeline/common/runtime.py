###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/pipeline/common/runtime.py
#
# Layer:
#     Enterprise Pipeline
#
# Purpose:
#     Provides shared runtime objects and helper functions for the MedFabric
#     master pipeline orchestrator.
#
# Business Context:
#     The Pipeline layer is the top-level enterprise orchestration layer for
#     MedFabric. It coordinates execution of the full platform from synthetic
#     data generation through Layer 3 Insights.
#
#     The Pipeline layer does not perform business transformations. Instead, it
#     manages:
#
#         - configuration
#         - run identity
#         - logging
#         - layer execution status
#         - audit records
#         - validation records
#         - metadata records
#         - execution summaries
#
# Architectural Rule:
#     This module contains Pipeline runtime infrastructure only.
#
#     It does NOT contain:
#         - data generation logic
#         - medallion-layer transformation logic
#         - feature engineering logic
#         - modeling logic
#         - analytics logic
#         - reporting logic
#
# Inputs:
#     config/pipeline/medfabric_platform.yaml
#
# Outputs:
#     Runtime objects used by:
#         src/pipeline/build_medfabric_platform.py
#
# Run:
#     This file is imported by the Pipeline orchestrator.
#
###############################################################################

from __future__ import annotations

import getpass
import logging
import platform
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


###############################################################################
# Status Constants
###############################################################################

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_WARNING = "WARNING"
STATUS_SKIPPED = "SKIPPED"
STATUS_RUNNING = "RUNNING"


###############################################################################
# Default Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/pipeline/medfabric_platform.yaml"
DEFAULT_PIPELINE_NAME = "MedFabric Enterprise Pipeline"
DEFAULT_LAYER_NAME = "Master Platform Orchestrator"
DEFAULT_OUTPUT_FORMAT = "parquet"


###############################################################################
# Runtime Data Classes
###############################################################################

@dataclass
class PipelineRuntime:
    """
    Purpose
    -------
    Runtime context for one MedFabric master pipeline execution.

    Parameters
    ----------
    run_id:
        Unique identifier for the current pipeline run.

    project_root:
        Root directory of the MedFabric project.

    config_path:
        Resolved path to the master pipeline configuration file.

    config:
        Loaded master pipeline configuration.

    start_time_utc:
        Pipeline start timestamp in UTC.

    pipeline_name:
        Human-readable pipeline name.

    layer_name:
        Human-readable orchestration layer name.

    logger:
        Logger used by the master pipeline.

    user:
        Operating-system user executing the pipeline.

    hostname:
        Hostname of the machine executing the pipeline.

    python_version:
        Python version used for execution.

    platform_name:
        Operating-system/platform name.

    audit_records:
        In-memory audit records collected during execution.

    validation_records:
        In-memory validation records collected during execution.

    dataset_records:
        In-memory dataset inventory records collected during execution.

    rule_records:
        In-memory rule catalog records collected during execution.

    layer_results:
        In-memory layer execution result records.

    Returns
    -------
    PipelineRuntime
        Runtime context used by the master pipeline orchestrator.

    Notes
    -----
    The runtime object is passed through Pipeline common modules to avoid
    repeatedly loading configuration, creating loggers, or recreating metadata
    containers.
    """

    run_id: str
    project_root: Path
    config_path: Path
    config: Dict[str, Any]
    start_time_utc: datetime
    pipeline_name: str
    layer_name: str
    logger: logging.Logger
    user: str
    hostname: str
    python_version: str
    platform_name: str
    audit_records: List[Dict[str, Any]] = field(default_factory=list)
    validation_records: List[Dict[str, Any]] = field(default_factory=list)
    dataset_records: List[Dict[str, Any]] = field(default_factory=list)
    rule_records: List[Dict[str, Any]] = field(default_factory=list)
    layer_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineBuildResult:
    """
    Purpose
    -------
    Standard build result returned by the MedFabric master pipeline.

    Parameters
    ----------
    name:
        Name of the pipeline or layer.

    status:
        Execution status. Expected values are SUCCESS, FAILED, WARNING, or
        SKIPPED.

    message:
        Human-readable execution message.

    row_count:
        Optional total row count produced by the pipeline or layer.

    column_count:
        Optional total column count produced by the pipeline or layer.

    Returns
    -------
    PipelineBuildResult
        Standardized build result.

    Notes
    -----
    This result allows the master pipeline to treat layer-level results
    consistently even when underlying modules return different result classes.
    """

    name: str
    status: str
    message: str
    row_count: int = 0
    column_count: int = 0


###############################################################################
# Time and Run ID Helpers
###############################################################################

def utc_now() -> datetime:
    """
    Purpose
    -------
    Return the current timezone-aware UTC timestamp.

    Parameters
    ----------
    None

    Returns
    -------
    datetime
        Current UTC timestamp.

    Raises
    ------
    None

    Notes
    -----
    All MedFabric pipeline audit and metadata timestamps should use UTC.
    """

    return datetime.now(timezone.utc)


def generate_run_id(timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Purpose
    -------
    Generate a timestamp-based pipeline run identifier.

    Parameters
    ----------
    timestamp_format:
        Timestamp format used to create the run identifier.

    Returns
    -------
    str
        Pipeline run identifier.

    Raises
    ------
    None

    Notes
    -----
    The run ID is used across audit, validation, metadata, and execution history
    outputs.
    """

    return utc_now().strftime(timestamp_format)


###############################################################################
# Path Helpers
###############################################################################

def get_project_root() -> Path:
    """
    Purpose
    -------
    Return the current MedFabric project root.

    Parameters
    ----------
    None

    Returns
    -------
    pathlib.Path
        Project root path.

    Raises
    ------
    None

    Notes
    -----
    The pipeline should be run from the MedFabric project root using:

        python -m src.pipeline.build_medfabric_platform
    """

    return Path.cwd()


def normalize_path(project_root: Path, raw_path: str | Path) -> Path:
    """
    Purpose
    -------
    Resolve a raw path relative to the project root.

    Parameters
    ----------
    project_root:
        MedFabric project root.

    raw_path:
        Raw configured path.

    Returns
    -------
    pathlib.Path
        Resolved path.

    Raises
    ------
    ValueError
        Raised when raw_path is missing.

    Notes
    -----
    Absolute paths are preserved. Relative paths are resolved against the
    project root.
    """

    if raw_path is None or str(raw_path).strip() == "":
        raise ValueError("Path value is required.")

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return project_root / path


def normalize_config_path(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """
    Purpose
    -------
    Normalize the master pipeline configuration path.

    Parameters
    ----------
    config_path:
        Pipeline configuration path.

    Returns
    -------
    pathlib.Path
        Resolved configuration path.

    Raises
    ------
    ValueError
        Raised when config_path is missing.

    Notes
    -----
    Unlike PipelineContext-based layers, this master pipeline runtime loads the
    YAML file directly from the project root. Therefore the default includes the
    full path:

        config/pipeline/medfabric_platform.yaml
    """

    project_root = get_project_root()
    return normalize_path(project_root, config_path)


###############################################################################
# Configuration Helpers
###############################################################################

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Purpose
    -------
    Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Resolved YAML configuration path.

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

    Notes
    -----
    The master pipeline loads configuration directly to keep orchestration
    independent of any lower-layer configuration manager behavior.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Pipeline configuration file is empty: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Pipeline configuration must be a YAML mapping: {config_path}"
        )

    return config


def get_nested_config_value(
    config: Dict[str, Any],
    section: str,
    key: str,
    default_value: Any,
) -> Any:
    """
    Purpose
    -------
    Safely read a value from a top-level configuration section.

    Parameters
    ----------
    config:
        Loaded pipeline configuration.

    section:
        Top-level YAML section name.

    key:
        Key inside the section.

    default_value:
        Value returned when the key is not configured.

    Returns
    -------
    Any
        Configured value or default value.

    Raises
    ------
    None

    Notes
    -----
    This helper avoids repeated defensive dictionary handling throughout the
    Pipeline common modules.
    """

    section_config = config.get(section, {})

    if not isinstance(section_config, dict):
        return default_value

    return section_config.get(key, default_value)


###############################################################################
# Logging Helpers
###############################################################################

def ensure_directory(path: Path) -> None:
    """
    Purpose
    -------
    Ensure a directory exists.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    None

    Raises
    ------
    OSError
        Raised by pathlib when the directory cannot be created.

    Notes
    -----
    Directory creation uses parents=True so nested output paths are supported.
    """

    path.mkdir(parents=True, exist_ok=True)


def configure_pipeline_logging(
    project_root: Path,
    config: Dict[str, Any],
    run_id: str,
) -> logging.Logger:
    """
    Purpose
    -------
    Configure logging for the MedFabric master pipeline.

    Parameters
    ----------
    project_root:
        MedFabric project root.

    config:
        Loaded pipeline configuration.

    run_id:
        Current pipeline run identifier.

    Returns
    -------
    logging.Logger
        Configured pipeline logger.

    Raises
    ------
    None

    Notes
    -----
    Logging configuration is controlled by the logging section in
    config/pipeline/medfabric_platform.yaml.
    """

    logging_config = config.get("logging", {})

    log_level_name = str(logging_config.get("level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_directory = normalize_path(
        project_root,
        logging_config.get("log_directory", "logs/pipeline"),
    )

    ensure_directory(log_directory)

    log_file_path = log_directory / logging_config.get(
        "log_file_name",
        "medfabric_pipeline.log",
    )

    logger = logging.getLogger("medfabric.pipeline")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [RUN_ID=%(run_id)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RunIdFilter(logging.Filter):
        """
        Purpose
        -------
        Inject the pipeline run ID into every log record.
        """

        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = run_id
            return True

    if bool(logging_config.get("console_logging", True)):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(RunIdFilter())
        logger.addHandler(stream_handler)

    if bool(logging_config.get("file_logging", True)):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RunIdFilter())
        logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("MedFabric Enterprise Pipeline logging initialized")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Log file: %s", log_file_path)

    return logger


###############################################################################
# Environment Helpers
###############################################################################

def get_runtime_user() -> str:
    """
    Purpose
    -------
    Return the operating-system user executing the pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Current user name.

    Raises
    ------
    None
    """

    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def get_hostname() -> str:
    """
    Purpose
    -------
    Return the hostname of the machine executing the pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Hostname.

    Raises
    ------
    None
    """

    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def get_python_version() -> str:
    """
    Purpose
    -------
    Return the Python version used for execution.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Python version string.

    Raises
    ------
    None
    """

    return sys.version.replace("\n", " ")


def get_platform_name() -> str:
    """
    Purpose
    -------
    Return the operating-system/platform name.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Platform description.

    Raises
    ------
    None
    """

    try:
        return platform.platform()
    except Exception:
        return "unknown"


###############################################################################
# Runtime Factory
###############################################################################

def initialize_pipeline_runtime(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> PipelineRuntime:
    """
    Purpose
    -------
    Initialize the MedFabric master pipeline runtime.

    Parameters
    ----------
    config_path:
        Path to config/pipeline/medfabric_platform.yaml.

    Returns
    -------
    PipelineRuntime
        Initialized runtime for the master pipeline.

    Raises
    ------
    FileNotFoundError
        Raised when the pipeline configuration file does not exist.

    ValueError
        Raised when configuration is invalid.

    Notes
    -----
    This function is the main entry point used by
    src.pipeline.build_medfabric_platform to prepare runtime state before
    executing platform layers.
    """

    project_root = get_project_root()
    resolved_config_path = normalize_config_path(config_path)
    config = load_yaml_config(resolved_config_path)

    timestamp_format = get_nested_config_value(
        config=config,
        section="runtime",
        key="timestamp_format",
        default_value="%Y%m%d_%H%M%S",
    )

    run_id = generate_run_id(timestamp_format=timestamp_format)

    pipeline_name = get_nested_config_value(
        config=config,
        section="project",
        key="pipeline_name",
        default_value=DEFAULT_PIPELINE_NAME,
    )

    layer_name = get_nested_config_value(
        config=config,
        section="project",
        key="layer_name",
        default_value=DEFAULT_LAYER_NAME,
    )

    logger = configure_pipeline_logging(
        project_root=project_root,
        config=config,
        run_id=run_id,
    )

    runtime = PipelineRuntime(
        run_id=run_id,
        project_root=project_root,
        config_path=resolved_config_path,
        config=config,
        start_time_utc=utc_now(),
        pipeline_name=pipeline_name,
        layer_name=layer_name,
        logger=logger,
        user=get_runtime_user(),
        hostname=get_hostname(),
        python_version=get_python_version(),
        platform_name=get_platform_name(),
    )

    logger.info("Pipeline runtime initialized successfully.")
    logger.info("Project root: %s", project_root)
    logger.info("Configuration file: %s", resolved_config_path)
    logger.info("Pipeline name: %s", pipeline_name)
    logger.info("Layer name: %s", layer_name)

    return runtime


###############################################################################
# Result Normalization
###############################################################################

def normalize_layer_result(
    layer_name: str,
    raw_result: Any,
) -> PipelineBuildResult:
    """
    Purpose
    -------
    Normalize any layer result object into PipelineBuildResult.

    Parameters
    ----------
    layer_name:
        Name of the layer that produced the result.

    raw_result:
        Result object returned by the layer builder.

    Returns
    -------
    PipelineBuildResult
        Normalized result object.

    Raises
    ------
    None

    Notes
    -----
    Some MedFabric layers return their own BuildResult objects. This helper lets
    the master pipeline normalize them without requiring every layer to import
    Pipeline-specific classes.
    """

    if raw_result is None:
        return PipelineBuildResult(
            name=layer_name,
            status=STATUS_SUCCESS,
            message=f"{layer_name} completed successfully.",
        )

    return PipelineBuildResult(
        name=getattr(raw_result, "name", layer_name),
        status=getattr(raw_result, "status", STATUS_SUCCESS),
        message=getattr(
            raw_result,
            "message",
            f"{layer_name} completed successfully.",
        ),
        row_count=int(getattr(raw_result, "row_count", 0) or 0),
        column_count=int(getattr(raw_result, "column_count", 0) or 0),
    )


###############################################################################
# End of File
###############################################################################