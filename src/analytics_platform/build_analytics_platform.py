###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/build_analytics_platform.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Master orchestrator for the MedFabric Analytics Platform.
#
#     This file controls execution of Layer 2 analytics domains using:
#
#         config/analytics_platform/analytics_platform.yaml
#
#     It does not contain domain-specific business rules.
#
#     Domain business rules belong in domain-specific YAML files such as:
#
#         config/analytics_platform/population_health.yaml
#
# Responsibilities:
#     - Load Analytics Platform orchestration configuration.
#     - Validate enabled domain configuration.
#     - Build domain registry metadata.
#     - Build execution plan metadata.
#     - Execute enabled analytics domains in configured sequence.
#     - Capture audit records.
#     - Capture validation records.
#     - Write execution summary.
#     - Write Layer 2 metadata and audit outputs.
#
# Current Enabled Domain:
#     - Population Health
#
# Future Domains:
#     - Clinical Analytics
#     - Quality Analytics
#     - Predictive Analytics
#     - Provider Analytics
#     - Care Management
#     - Value-Based Care
#
# Inputs:
#     config/analytics_platform/analytics_platform.yaml
#
# Outputs:
#     data/analytics_platform/metadata/
#     data/analytics_platform/audit/
#
# Run:
#     python -m src.analytics_platform.build_analytics_platform
#
###############################################################################

from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "config/analytics_platform/analytics_platform.yaml"

DEFAULT_LAYER_NAME = "Layer 2 - Analytics Platform"

DEFAULT_PLATFORM_NAME = "MedFabric Analytics Platform"

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
class AnalyticsPlatformRuntime:
    """
    Runtime context for one Analytics Platform execution.
    """

    run_id: str
    project_root: Path
    config_path: Path
    start_time_utc: datetime
    config: Dict[str, Any]
    logger: logging.Logger
    layer_name: str
    platform_name: str
    audit_records: List[Dict[str, Any]] = field(default_factory=list)
    validation_records: List[Dict[str, Any]] = field(default_factory=list)
    domain_execution_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DomainExecutionResult:
    """
    Standardized result for one analytics domain execution.
    """

    domain_key: str
    domain_name: str
    status: str
    message: str
    start_time_utc: str
    end_time_utc: str
    duration_seconds: float


@dataclass
class BuildResult:
    """
    Standard result returned by the Layer 2 orchestrator.
    """

    name: str
    status: str
    message: str


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
    Resolve relative or absolute path.
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
    Safely convert value to string.
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
    Configure Analytics Platform logging.
    """

    logging_config = config.get("logging", {})

    log_level_name = logging_config.get("level", "INFO")
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)

    log_dir_raw = logging_config.get("module_log_dir", "logs/modules")
    log_dir = normalize_path(project_root, log_dir_raw)
    ensure_directory(log_dir)

    log_file_name = logging_config.get("log_file_name", "analytics_platform.log")
    log_file_path = log_dir / log_file_name

    logger = logging.getLogger("medfabric.analytics_platform")
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
    logger.info("MedFabric Analytics Platform execution started")
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
    Validate Analytics Platform master configuration.
    """

    errors: List[str] = []

    required_sections = [
        "analytics_platform",
        "logging",
        "execution",
        "domains",
        "paths",
        "validation",
        "metadata",
        "audit",
    ]

    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")

    platform_config = config.get("analytics_platform", {})
    output_format = platform_config.get("output_format", DEFAULT_OUTPUT_FORMAT)

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    execution_config = config.get("execution", {})

    if "sequence" not in execution_config:
        errors.append("Missing required configuration section: execution.sequence")

    if not isinstance(execution_config.get("sequence", []), list):
        errors.append("execution.sequence must be a list.")

    domains_config = config.get("domains", {})

    if not isinstance(domains_config, dict) or not domains_config:
        errors.append("domains must be a non-empty dictionary.")

    for domain_key, domain_config in domains_config.items():
        if not isinstance(domain_config, dict):
            errors.append(f"Domain configuration must be dictionary: {domain_key}")
            continue

        required_domain_fields = [
            "enabled",
            "required",
            "domain_name",
            "config_path",
            "module_path",
            "function_name",
            "expected_output_dir",
        ]

        for field_name in required_domain_fields:
            if field_name not in domain_config:
                errors.append(
                    f"Missing domain field '{field_name}' for domain '{domain_key}'"
                )

    paths_config = config.get("paths", {})

    if not isinstance(paths_config, dict):
        errors.append("paths must be a dictionary.")
    else:
        if "metadata_outputs" not in paths_config:
            errors.append("Missing required configuration section: paths.metadata_outputs")
        if "audit_outputs" not in paths_config:
            errors.append("Missing required configuration section: paths.audit_outputs")

    if errors:
        raise ValueError(
            "Analytics Platform configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> AnalyticsPlatformRuntime:
    """
    Initialize Analytics Platform runtime.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    run_id = generate_run_id()

    config = load_yaml_config(config_path)
    validate_config(config)

    platform_config = config.get("analytics_platform", {})

    layer_name = platform_config.get("layer_name", DEFAULT_LAYER_NAME)
    platform_name = platform_config.get("platform_name", DEFAULT_PLATFORM_NAME)

    logger = configure_logging(project_root, config, run_id)

    runtime = AnalyticsPlatformRuntime(
        run_id=run_id,
        project_root=project_root,
        config_path=config_path,
        start_time_utc=utc_now(),
        config=config,
        logger=logger,
        layer_name=layer_name,
        platform_name=platform_name,
    )

    add_audit_record(
        runtime=runtime,
        step_name="initialize_runtime",
        status=STATUS_SUCCESS,
        message="Analytics Platform runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit and Validation
###############################################################################

def add_audit_record(
    runtime: AnalyticsPlatformRuntime,
    step_name: str,
    status: str,
    message: str,
    domain_key: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Add an audit record.
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "platform_name": runtime.platform_name,
            "domain_key": domain_key,
            "step_name": step_name,
            "status": status,
            "message": message,
            "output_path": output_path,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


def add_validation_record(
    runtime: AnalyticsPlatformRuntime,
    rule_name: str,
    status: str,
    message: str,
    domain_key: Optional[str] = None,
    severity: str = "ERROR",
) -> None:
    """
    Add a validation record.
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "platform_name": runtime.platform_name,
            "domain_key": domain_key,
            "rule_name": rule_name,
            "status": status,
            "severity": severity,
            "message": message,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


###############################################################################
# Dataset IO
###############################################################################

def get_output_format(runtime: AnalyticsPlatformRuntime) -> str:
    """
    Return configured output format.
    """

    return (
        runtime.config.get("analytics_platform", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Ensure output path has the configured suffix.
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


def get_configured_output_path(
    runtime: AnalyticsPlatformRuntime,
    output_group: str,
    output_name: str,
) -> Path:
    """
    Resolve configured metadata or audit output path.
    """

    group_config = runtime.config.get("paths", {}).get(output_group, {})
    output_entry = group_config.get(output_name)

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        raw_path = f"data/analytics_platform/{output_group}/{output_name}"

    return normalize_path(runtime.project_root, raw_path)


###############################################################################
# Domain Registry and Execution Plan
###############################################################################

def build_domain_registry(runtime: AnalyticsPlatformRuntime) -> pd.DataFrame:
    """
    Build domain registry metadata.
    """

    rows: List[Dict[str, Any]] = []

    domains_config = runtime.config.get("domains", {})

    for domain_key, domain_config in domains_config.items():
        rows.append(
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "platform_name": runtime.platform_name,
                "domain_key": domain_key,
                "domain_name": domain_config.get("domain_name"),
                "enabled": bool(domain_config.get("enabled", False)),
                "required": bool(domain_config.get("required", False)),
                "config_path": domain_config.get("config_path"),
                "module_path": domain_config.get("module_path"),
                "function_name": domain_config.get("function_name"),
                "expected_output_dir": domain_config.get("expected_output_dir"),
                "created_at_utc": utc_now().isoformat(),
            }
        )

    return pd.DataFrame(rows)


def build_execution_plan(runtime: AnalyticsPlatformRuntime) -> pd.DataFrame:
    """
    Build configured execution plan.
    """

    rows: List[Dict[str, Any]] = []

    sequence = runtime.config.get("execution", {}).get("sequence", [])
    domains_config = runtime.config.get("domains", {})

    for sequence_number, domain_key in enumerate(sequence, start=1):
        domain_config = domains_config.get(domain_key, {})

        rows.append(
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "platform_name": runtime.platform_name,
                "sequence_number": sequence_number,
                "domain_key": domain_key,
                "domain_name": domain_config.get("domain_name"),
                "enabled": bool(domain_config.get("enabled", False)),
                "required": bool(domain_config.get("required", False)),
                "config_path": domain_config.get("config_path"),
                "module_path": domain_config.get("module_path"),
                "function_name": domain_config.get("function_name"),
                "expected_output_dir": domain_config.get("expected_output_dir"),
                "created_at_utc": utc_now().isoformat(),
            }
        )

    return pd.DataFrame(rows)


###############################################################################
# Domain Validation
###############################################################################

def validate_domain_config_file(
    runtime: AnalyticsPlatformRuntime,
    domain_key: str,
    domain_config: Dict[str, Any],
) -> bool:
    """
    Validate that enabled domain config file exists.
    """

    config_path_raw = domain_config.get("config_path")
    config_path = normalize_path(runtime.project_root, config_path_raw)

    if config_path.exists():
        add_validation_record(
            runtime=runtime,
            rule_name="config_file_exists",
            status=STATUS_SUCCESS,
            message=f"Config file exists: {config_path}",
            domain_key=domain_key,
        )
        return True

    add_validation_record(
        runtime=runtime,
        rule_name="config_file_exists",
        status=STATUS_FAILED,
        message=f"Config file missing: {config_path}",
        domain_key=domain_key,
    )
    return False


def validate_domain_module(
    runtime: AnalyticsPlatformRuntime,
    domain_key: str,
    domain_config: Dict[str, Any],
) -> bool:
    """
    Validate that enabled domain module and function are importable.
    """

    module_path = domain_config.get("module_path")
    function_name = domain_config.get("function_name")

    try:
        module = importlib.import_module(module_path)

        add_validation_record(
            runtime=runtime,
            rule_name="module_importable",
            status=STATUS_SUCCESS,
            message=f"Module import successful: {module_path}",
            domain_key=domain_key,
        )

    except Exception as exc:
        add_validation_record(
            runtime=runtime,
            rule_name="module_importable",
            status=STATUS_FAILED,
            message=f"Module import failed: {module_path} | {exc}",
            domain_key=domain_key,
        )
        return False

    if hasattr(module, function_name):
        add_validation_record(
            runtime=runtime,
            rule_name="function_exists",
            status=STATUS_SUCCESS,
            message=f"Function exists: {function_name}",
            domain_key=domain_key,
        )
        return True

    add_validation_record(
        runtime=runtime,
        rule_name="function_exists",
        status=STATUS_FAILED,
        message=f"Function missing: {function_name}",
        domain_key=domain_key,
    )
    return False


def validate_enabled_domains(runtime: AnalyticsPlatformRuntime) -> None:
    """
    Validate enabled analytics domains before execution.
    """

    logger = runtime.logger
    sequence = runtime.config.get("execution", {}).get("sequence", [])
    domains_config = runtime.config.get("domains", {})

    logger.info("START: Validate enabled analytics domains")

    validation_failed = False

    for domain_key in sequence:
        domain_config = domains_config.get(domain_key)

        if not domain_config:
            add_validation_record(
                runtime=runtime,
                rule_name="domain_exists_in_registry",
                status=STATUS_FAILED,
                message=f"Domain in execution sequence missing from registry: {domain_key}",
                domain_key=domain_key,
            )
            validation_failed = True
            continue

        enabled = bool(domain_config.get("enabled", False))

        if not enabled:
            add_validation_record(
                runtime=runtime,
                rule_name="domain_enabled",
                status=STATUS_SKIPPED,
                message="Domain disabled by configuration.",
                domain_key=domain_key,
                severity="INFO",
            )
            continue

        config_ok = validate_domain_config_file(runtime, domain_key, domain_config)
        module_ok = validate_domain_module(runtime, domain_key, domain_config)

        if not config_ok or not module_ok:
            validation_failed = True

    logger.info("COMPLETE: Validate enabled analytics domains")

    add_audit_record(
        runtime=runtime,
        step_name="validate_enabled_domains",
        status=STATUS_SUCCESS if not validation_failed else STATUS_FAILED,
        message="Enabled domain validation completed.",
    )

    if validation_failed:
        raise ValueError("Analytics Platform enabled domain validation failed.")


###############################################################################
# Domain Execution
###############################################################################

def execute_domain(
    runtime: AnalyticsPlatformRuntime,
    domain_key: str,
    domain_config: Dict[str, Any],
) -> DomainExecutionResult:
    """
    Execute one enabled analytics domain.
    """

    logger = runtime.logger

    domain_name = domain_config.get("domain_name", domain_key)
    module_path = domain_config.get("module_path")
    function_name = domain_config.get("function_name")
    config_path = domain_config.get("config_path")

    start_time = utc_now()

    logger.info("=" * 80)
    logger.info("START DOMAIN: %s", domain_name)
    logger.info("=" * 80)

    try:
        module = importlib.import_module(module_path)
        build_function = getattr(module, function_name)

        result = build_function(config_path=config_path)

        result_status = getattr(result, "status", STATUS_SUCCESS)
        result_message = getattr(result, "message", "Domain completed.")

        end_time = utc_now()
        duration_seconds = (end_time - start_time).total_seconds()

        domain_result = DomainExecutionResult(
            domain_key=domain_key,
            domain_name=domain_name,
            status=result_status,
            message=result_message,
            start_time_utc=start_time.isoformat(),
            end_time_utc=end_time.isoformat(),
            duration_seconds=duration_seconds,
        )

        runtime.domain_execution_records.append(domain_result.__dict__)

        add_audit_record(
            runtime=runtime,
            step_name=f"execute_domain:{domain_key}",
            status=result_status,
            message=result_message,
            domain_key=domain_key,
        )

        logger.info(
            "COMPLETE DOMAIN: %s | Status: %s | Duration: %.2f seconds",
            domain_name,
            result_status,
            duration_seconds,
        )

        return domain_result

    except Exception as exc:
        end_time = utc_now()
        duration_seconds = (end_time - start_time).total_seconds()

        message = f"Domain failed: {domain_name} | Error: {exc}"

        logger.error(message)
        logger.error("Traceback:\n%s", traceback.format_exc())

        domain_result = DomainExecutionResult(
            domain_key=domain_key,
            domain_name=domain_name,
            status=STATUS_FAILED,
            message=str(exc),
            start_time_utc=start_time.isoformat(),
            end_time_utc=end_time.isoformat(),
            duration_seconds=duration_seconds,
        )

        runtime.domain_execution_records.append(domain_result.__dict__)

        add_audit_record(
            runtime=runtime,
            step_name=f"execute_domain:{domain_key}",
            status=STATUS_FAILED,
            message=message,
            domain_key=domain_key,
        )

        return domain_result


def execute_enabled_domains(runtime: AnalyticsPlatformRuntime) -> List[DomainExecutionResult]:
    """
    Execute all enabled domains in configured sequence.
    """

    logger = runtime.logger
    sequence = runtime.config.get("execution", {}).get("sequence", [])
    domains_config = runtime.config.get("domains", {})

    fail_fast = bool(runtime.config.get("execution", {}).get("fail_fast", True))
    continue_on_domain_failure = bool(
        runtime.config.get("execution", {}).get("continue_on_domain_failure", False)
    )

    results: List[DomainExecutionResult] = []

    logger.info("START: Execute enabled Analytics Platform domains")

    for domain_key in sequence:
        domain_config = domains_config.get(domain_key, {})
        enabled = bool(domain_config.get("enabled", False))

        if not enabled:
            logger.info("SKIP DOMAIN: %s disabled by configuration", domain_key)

            add_audit_record(
                runtime=runtime,
                step_name=f"skip_domain:{domain_key}",
                status=STATUS_SKIPPED,
                message="Domain disabled by configuration.",
                domain_key=domain_key,
            )
            continue

        result = execute_domain(runtime, domain_key, domain_config)
        results.append(result)

        if result.status != STATUS_SUCCESS:
            if fail_fast and not continue_on_domain_failure:
                raise RuntimeError(
                    f"Analytics Platform stopped because domain failed: {domain_key}"
                )

    logger.info("COMPLETE: Execute enabled Analytics Platform domains")

    return results


###############################################################################
# Execution Summary
###############################################################################

def build_execution_summary(runtime: AnalyticsPlatformRuntime) -> pd.DataFrame:
    """
    Build one-row Analytics Platform execution summary.
    """

    end_time = utc_now()
    duration_seconds = (end_time - runtime.start_time_utc).total_seconds()

    failed_domains = [
        record for record in runtime.domain_execution_records
        if record.get("status") != STATUS_SUCCESS
    ]

    enabled_domain_count = sum(
        1
        for domain_config in runtime.config.get("domains", {}).values()
        if bool(domain_config.get("enabled", False))
    )

    summary = {
        "run_id": runtime.run_id,
        "layer_name": runtime.layer_name,
        "platform_name": runtime.platform_name,
        "config_path": str(runtime.config_path),
        "start_time_utc": runtime.start_time_utc.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "enabled_domain_count": enabled_domain_count,
        "executed_domain_count": len(runtime.domain_execution_records),
        "failed_domain_count": len(failed_domains),
        "audit_record_count": len(runtime.audit_records),
        "validation_record_count": len(runtime.validation_records),
        "status": STATUS_SUCCESS if len(failed_domains) == 0 else STATUS_FAILED,
    }

    return pd.DataFrame([summary])


###############################################################################
# Output Writing
###############################################################################

def write_metadata_outputs(runtime: AnalyticsPlatformRuntime) -> None:
    """
    Write Analytics Platform metadata outputs.
    """

    output_format = get_output_format(runtime)

    metadata_assets = {
        "analytics_platform_domain_registry": build_domain_registry(runtime),
        "analytics_platform_execution_plan": build_execution_plan(runtime),
    }

    for output_name, dataframe in metadata_assets.items():
        output_path = get_configured_output_path(
            runtime=runtime,
            output_group="metadata_outputs",
            output_name=output_name,
        )
        output_path = output_path_with_format(output_path, output_format)

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote metadata output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )

        add_audit_record(
            runtime=runtime,
            step_name=f"write_metadata:{output_name}",
            status=STATUS_SUCCESS,
            message="Metadata output written successfully.",
            output_path=str(output_path),
        )


def write_audit_outputs(runtime: AnalyticsPlatformRuntime) -> None:
    """
    Write Analytics Platform audit outputs.
    """

    output_format = get_output_format(runtime)

    audit_assets = {
        "analytics_platform_audit_records": pd.DataFrame(runtime.audit_records),
        "analytics_platform_validation_results": pd.DataFrame(runtime.validation_records),
        "analytics_platform_execution_summary": build_execution_summary(runtime),
    }

    for output_name, dataframe in audit_assets.items():
        output_path = get_configured_output_path(
            runtime=runtime,
            output_group="audit_outputs",
            output_name=output_name,
        )
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

def build_analytics_platform(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build the full enabled Analytics Platform sequence.
    """

    runtime: Optional[AnalyticsPlatformRuntime] = None

    try:
        runtime = initialize_runtime(config_path)

        runtime.logger.info("Configuration path: %s", runtime.config_path)

        write_metadata_outputs(runtime)

        validate_enabled_domains(runtime)

        execute_enabled_domains(runtime)

        add_audit_record(
            runtime=runtime,
            step_name="build_analytics_platform",
            status=STATUS_SUCCESS,
            message="Analytics Platform completed successfully.",
        )

        write_audit_outputs(runtime)

        runtime.logger.info("=" * 80)
        runtime.logger.info("MedFabric Analytics Platform completed successfully")
        runtime.logger.info("=" * 80)

        return BuildResult(
            name="analytics_platform",
            status=STATUS_SUCCESS,
            message="Analytics Platform completed successfully.",
        )

    except Exception as exc:
        if runtime is not None:
            runtime.logger.error("=" * 80)
            runtime.logger.error("MedFabric Analytics Platform failed")
            runtime.logger.error("Error: %s", exc)
            runtime.logger.error("Traceback:\n%s", traceback.format_exc())
            runtime.logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_analytics_platform",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_audit_outputs(runtime)
            except Exception as audit_exc:
                runtime.logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="analytics_platform",
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
        python -m src.analytics_platform.build_analytics_platform
    """

    config_path = os.environ.get(
        "MEDFABRIC_ANALYTICS_PLATFORM_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_analytics_platform(config_path=config_path)

    if result.status == STATUS_SUCCESS:
        print(result.message)
        return

    print(f"Analytics Platform failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()