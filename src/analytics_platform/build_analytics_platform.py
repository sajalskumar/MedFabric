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
# Architectural Standardization:
#     This orchestrator uses Layer 0 Foundation managers from src/common instead
#     of duplicating configuration, logging, path, storage, and validation logic.
#
# Run:
#     python -m src.analytics_platform.build_analytics_platform
#
###############################################################################

from __future__ import annotations

import importlib
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import PipelineContext, create_pipeline_context


###############################################################################
# Constants
###############################################################################

DEFAULT_CONFIG_PATH = "analytics_platform/analytics_platform.yaml"

DEFAULT_LAYER_NAME = "Layer 2 - Analytics Platform"
DEFAULT_PLATFORM_NAME = "MedFabric Analytics Platform"
DEFAULT_OUTPUT_FORMAT = "parquet"

SUPPORTED_FILE_FORMATS = {"parquet", "csv", "json"}

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_WARNING = "WARNING"
STATUS_SKIPPED = "SKIPPED"


###############################################################################
# Data Classes
###############################################################################

@dataclass
class DomainExecutionResult:
    """
    Purpose
    -------
    Stores standardized execution results for one Analytics Platform domain.

    Parameters
    ----------
    domain_key:
        Internal domain key from analytics_platform.yaml.

    domain_name:
        Human-readable domain name.

    status:
        Execution status.

    message:
        Execution message returned by the domain builder.

    start_time_utc:
        Domain start timestamp in UTC ISO format.

    end_time_utc:
        Domain end timestamp in UTC ISO format.

    duration_seconds:
        Domain execution duration in seconds.
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
    Purpose
    -------
    Stores standardized result returned by the Layer 2 orchestrator.

    Parameters
    ----------
    name:
        Build component name.

    status:
        Final execution status.

    message:
        Final execution message.
    """

    name: str
    status: str
    message: str


@dataclass
class AnalyticsPlatformRuntime:
    """
    Purpose
    -------
    Holds Analytics Platform runtime state.

    Parameters
    ----------
    context:
        MedFabric PipelineContext created from src/common.

    config:
        Loaded Analytics Platform orchestration configuration.

    config_file:
        Config file name passed to ConfigurationManager.

    layer_name:
        Logical layer name.

    platform_name:
        Platform name.

    start_time_utc:
        Runtime start timestamp.

    audit_records:
        In-memory audit records.

    validation_records:
        In-memory validation records.

    domain_execution_records:
        In-memory domain execution records.

    Notes
    -----
    This runtime intentionally stores only Analytics Platform orchestration
    records. Foundation services come from PipelineContext.
    """

    context: PipelineContext
    config: Dict[str, Any]
    config_file: str
    layer_name: str
    platform_name: str
    start_time_utc: datetime
    audit_records: List[Dict[str, Any]]
    validation_records: List[Dict[str, Any]]
    domain_execution_records: List[Dict[str, Any]]


###############################################################################
# Time Helpers
###############################################################################

def utc_now() -> datetime:
    """
    Purpose
    -------
    Return current timezone-aware UTC timestamp.

    Returns
    -------
    datetime
        Current UTC timestamp.
    """

    return datetime.now(timezone.utc)


###############################################################################
# Configuration Helpers
###############################################################################

def normalize_config_file(config_path: str) -> str:
    """
    Purpose
    -------
    Convert a repository-relative config path into a config-root-relative path
    expected by ConfigurationManager.

    Parameters
    ----------
    config_path:
        Config path. Examples:
        - config/analytics_platform/analytics_platform.yaml
        - analytics_platform/analytics_platform.yaml

    Returns
    -------
    str
        Config-root-relative path.

    Notes
    -----
    ConfigurationManager is initialized with config_root="config", so passing
    "analytics_platform/analytics_platform.yaml" is preferred.
    """

    normalized = str(config_path).strip()

    if normalized.startswith("config/"):
        normalized = normalized[len("config/"):]

    return normalized


def validate_analytics_platform_config(config: Dict[str, Any]) -> None:
    """
    Purpose
    -------
    Validate Analytics Platform master configuration.

    Parameters
    ----------
    config:
        Loaded analytics_platform.yaml configuration.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when required sections, domain fields, or output format are
        invalid.

    Notes
    -----
    This validation is orchestrator-specific. Generic YAML loading is handled
    by ConfigurationManager.
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

    output_format = (
        config.get("analytics_platform", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )

    if output_format not in SUPPORTED_FILE_FORMATS:
        errors.append(
            f"Unsupported output_format '{output_format}'. "
            f"Supported formats: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    execution_config = config.get("execution", {})
    sequence = execution_config.get("sequence")

    if sequence is None:
        errors.append("Missing required configuration section: execution.sequence")
    elif not isinstance(sequence, list):
        errors.append("execution.sequence must be a list.")

    domains_config = config.get("domains", {})

    if not isinstance(domains_config, dict) or not domains_config:
        errors.append("domains must be a non-empty dictionary.")
    else:
        required_domain_fields = [
            "enabled",
            "required",
            "domain_name",
            "config_path",
            "module_path",
            "function_name",
            "expected_output_dir",
        ]

        for domain_key, domain_config in domains_config.items():
            if not isinstance(domain_config, dict):
                errors.append(f"Domain configuration must be dictionary: {domain_key}")
                continue

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
        raise PipelineError(
            "Analytics Platform configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def initialize_runtime(config_path: str) -> AnalyticsPlatformRuntime:
    """
    Purpose
    -------
    Initialize Analytics Platform runtime using src/common PipelineContext.

    Parameters
    ----------
    config_path:
        Analytics Platform config path.

    Returns
    -------
    AnalyticsPlatformRuntime
        Initialized runtime.

    Raises
    ------
    PipelineError
        Raised when context creation or configuration validation fails.
    """

    config_file = normalize_config_file(config_path)

    context = create_pipeline_context(
        pipeline_name="Layer 2 - Analytics Platform",
    )

    config = context.configuration.load_yaml(config_file)
    validate_analytics_platform_config(config)

    platform_config = config.get("analytics_platform", {})

    runtime = AnalyticsPlatformRuntime(
        context=context,
        config=config,
        config_file=config_file,
        layer_name=platform_config.get("layer_name", DEFAULT_LAYER_NAME),
        platform_name=platform_config.get("platform_name", DEFAULT_PLATFORM_NAME),
        start_time_utc=utc_now(),
        audit_records=[],
        validation_records=[],
        domain_execution_records=[],
    )

    add_audit_record(
        runtime=runtime,
        step_name="initialize_runtime",
        status=STATUS_SUCCESS,
        message="Analytics Platform runtime initialized successfully.",
    )

    return runtime


###############################################################################
# Audit and Validation Records
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
    Purpose
    -------
    Add an Analytics Platform audit record.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    step_name:
        Step name.

    status:
        Step status.

    message:
        Human-readable audit message.

    domain_key:
        Optional domain key.

    output_path:
        Optional output path.

    Returns
    -------
    None
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.context.run_id,
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
    Purpose
    -------
    Add an Analytics Platform validation record.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    rule_name:
        Validation rule name.

    status:
        Validation status.

    message:
        Validation message.

    domain_key:
        Optional domain key.

    severity:
        Severity level.

    Returns
    -------
    None
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.context.run_id,
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
# Output Helpers
###############################################################################

def get_output_format(runtime: AnalyticsPlatformRuntime) -> str:
    """
    Purpose
    -------
    Return configured output format.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    str
        Output format.
    """

    return (
        runtime.config.get("analytics_platform", {})
        .get("output_format", DEFAULT_OUTPUT_FORMAT)
    )


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Purpose
    -------
    Add or replace output file suffix.

    Parameters
    ----------
    path:
        Base output path.

    output_format:
        Output format suffix.

    Returns
    -------
    Path
        Output path with file suffix.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


def get_configured_output_path(
    runtime: AnalyticsPlatformRuntime,
    output_group: str,
    output_name: str,
) -> Path:
    """
    Purpose
    -------
    Resolve configured metadata or audit output path using PathManager.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    output_group:
        Config output group, such as metadata_outputs or audit_outputs.

    output_name:
        Output asset name.

    Returns
    -------
    Path
        Resolved absolute output path.

    Raises
    ------
    PipelineError
        Raised when path resolution fails.
    """

    group_config = runtime.config.get("paths", {}).get(output_group, {})
    output_entry = group_config.get(output_name)

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        raw_path = f"data/analytics_platform/{output_group}/{output_name}"

    return runtime.context.paths.resolve_path(raw_path)


def write_dataframe(
    runtime: AnalyticsPlatformRuntime,
    dataframe: pd.DataFrame,
    output_path: Path,
    output_format: str,
) -> None:
    """
    Purpose
    -------
    Write dataframe using StorageManager.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    dataframe:
        Dataframe to write.

    output_path:
        Output file path.

    output_format:
        Output format.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised for unsupported output formats.
    """

    if output_format == "parquet":
        runtime.context.storage.write_parquet(dataframe, output_path, index=False)
        return

    if output_format == "csv":
        runtime.context.storage.write_csv(dataframe, output_path, index=False)
        return

    if output_format == "json":
        runtime.context.storage.write_json(
            dataframe.to_dict(orient="records"),
            output_path,
        )
        return

    raise PipelineError(f"Unsupported output file format: {output_format}")


###############################################################################
# Domain Registry and Execution Plan
###############################################################################

def build_domain_registry(runtime: AnalyticsPlatformRuntime) -> pd.DataFrame:
    """
    Purpose
    -------
    Build domain registry metadata from analytics_platform.yaml.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    pandas.DataFrame
        Domain registry metadata.
    """

    rows: List[Dict[str, Any]] = []
    domains_config = runtime.config.get("domains", {})

    for domain_key, domain_config in domains_config.items():
        rows.append(
            {
                "run_id": runtime.context.run_id,
                "layer_name": runtime.layer_name,
                "platform_name": runtime.platform_name,
                "source_layer": "Layer 2 - Analytics Platform",
                "source_dataset": "analytics_platform.yaml",
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
    Purpose
    -------
    Build configured execution plan metadata.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    pandas.DataFrame
        Execution plan metadata.
    """

    rows: List[Dict[str, Any]] = []

    sequence = runtime.config.get("execution", {}).get("sequence", [])
    domains_config = runtime.config.get("domains", {})

    for sequence_number, domain_key in enumerate(sequence, start=1):
        domain_config = domains_config.get(domain_key, {})

        rows.append(
            {
                "run_id": runtime.context.run_id,
                "layer_name": runtime.layer_name,
                "platform_name": runtime.platform_name,
                "source_layer": "Layer 2 - Analytics Platform",
                "source_dataset": "analytics_platform.yaml",
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
    Purpose
    -------
    Validate that enabled domain configuration file exists.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    domain_key:
        Domain key.

    domain_config:
        Domain configuration.

    Returns
    -------
    bool
        True when config file exists, otherwise False.
    """

    config_path_raw = domain_config.get("config_path")
    config_path = runtime.context.paths.resolve_path(config_path_raw)

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
    Purpose
    -------
    Validate that enabled domain module and build function are importable.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    domain_key:
        Domain key.

    domain_config:
        Domain configuration.

    Returns
    -------
    bool
        True when module and function are importable, otherwise False.
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
    Purpose
    -------
    Validate enabled domains before execution.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    None

    Raises
    ------
    PipelineError
        Raised when enabled domain validation fails.
    """

    logger = runtime.context.get_logger("medfabric.analytics_platform")
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

    add_audit_record(
        runtime=runtime,
        step_name="validate_enabled_domains",
        status=STATUS_SUCCESS if not validation_failed else STATUS_FAILED,
        message="Enabled domain validation completed.",
    )

    logger.info("COMPLETE: Validate enabled analytics domains")

    if validation_failed:
        raise PipelineError("Analytics Platform enabled domain validation failed.")


###############################################################################
# Domain Execution
###############################################################################

def execute_domain(
    runtime: AnalyticsPlatformRuntime,
    domain_key: str,
    domain_config: Dict[str, Any],
) -> DomainExecutionResult:
    """
    Purpose
    -------
    Execute one enabled analytics domain.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    domain_key:
        Domain key.

    domain_config:
        Domain configuration.

    Returns
    -------
    DomainExecutionResult
        Domain execution result.
    """

    logger = runtime.context.get_logger("medfabric.analytics_platform")

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
    Purpose
    -------
    Execute all enabled Analytics Platform domains in configured sequence.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    list[DomainExecutionResult]
        Results for executed domains.

    Raises
    ------
    RuntimeError
        Raised when a domain fails and fail-fast behavior is enabled.
    """

    logger = runtime.context.get_logger("medfabric.analytics_platform")

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
    Purpose
    -------
    Build one-row Analytics Platform execution summary.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    pandas.DataFrame
        Execution summary dataframe.
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
        "run_id": runtime.context.run_id,
        "layer_name": runtime.layer_name,
        "platform_name": runtime.platform_name,
        "source_layer": "Layer 2 - Analytics Platform",
        "source_dataset": "analytics_platform.yaml",
        "config_path": runtime.config_file,
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
# Metadata and Audit Output Writing
###############################################################################

def write_metadata_outputs(runtime: AnalyticsPlatformRuntime) -> None:
    """
    Purpose
    -------
    Write Analytics Platform metadata outputs using StorageManager.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    None
    """

    logger = runtime.context.get_logger("medfabric.analytics_platform")
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

        write_dataframe(runtime, dataframe, output_path, output_format)

        logger.info(
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
    Purpose
    -------
    Write Analytics Platform audit outputs using StorageManager.

    Parameters
    ----------
    runtime:
        Analytics Platform runtime.

    Returns
    -------
    None
    """

    logger = runtime.context.get_logger("medfabric.analytics_platform")
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

        write_dataframe(runtime, dataframe, output_path, output_format)

        logger.info(
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
    Purpose
    -------
    Build the full enabled Analytics Platform sequence.

    Parameters
    ----------
    config_path:
        Analytics Platform config path.

    Returns
    -------
    BuildResult
        Final build result.

    Raises
    ------
    None
        Exceptions are captured and converted into a failed BuildResult.

    Notes
    -----
    This orchestrator uses src/common PipelineContext. Domain builders are still
    executed through their configured module paths and function names.
    """

    runtime: Optional[AnalyticsPlatformRuntime] = None

    try:
        runtime = initialize_runtime(config_path)

        logger = runtime.context.get_logger("medfabric.analytics_platform")
        logger.info("Configuration file: %s", runtime.config_file)

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

        logger.info("=" * 80)
        logger.info("MedFabric Analytics Platform completed successfully")
        logger.info("=" * 80)

        return BuildResult(
            name="analytics_platform",
            status=STATUS_SUCCESS,
            message="Analytics Platform completed successfully.",
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.context.get_logger("medfabric.analytics_platform")

            logger.error("=" * 80)
            logger.error("MedFabric Analytics Platform failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_audit_record(
                runtime=runtime,
                step_name="build_analytics_platform",
                status=STATUS_FAILED,
                message=str(exc),
            )

            try:
                write_audit_outputs(runtime)
            except Exception as audit_exc:
                logger.error("Failed to write audit outputs: %s", audit_exc)

        return BuildResult(
            name="analytics_platform",
            status=STATUS_FAILED,
            message=str(exc),
        )

    finally:
        if runtime is not None:
            runtime.context.logging.close()


###############################################################################
# CLI Entry Point
###############################################################################

def main() -> None:
    """
    Purpose
    -------
    Command-line entry point.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Raised with exit code 1 when Analytics Platform execution fails.
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