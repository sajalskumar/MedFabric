###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/pipeline/common/validation.py
#
# Layer:
#     Enterprise Pipeline
#
# Purpose:
#     Provides shared validation helpers for the MedFabric master pipeline
#     orchestrator.
#
# Business Context:
#     The Pipeline layer coordinates the full MedFabric platform. Before and
#     after execution, the pipeline must validate configuration, execution order,
#     layer registry definitions, layer results, and output readiness.
#
# Architectural Rule:
#     This module contains Pipeline validation infrastructure only.
#
#     It does NOT contain:
#         - layer execution logic
#         - data transformation logic
#         - modeling logic
#         - analytics logic
#         - reporting logic
#
# Inputs:
#     config/pipeline/medfabric_platform.yaml
#     PipelineRuntime
#     PipelineBuildResult
#
# Outputs:
#     Validation records stored in runtime.validation_records.
#
# Used By:
#     src/pipeline/build_medfabric_platform.py
#
###############################################################################

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Optional

from src.pipeline.common.runtime import (
    PipelineBuildResult,
    PipelineRuntime,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    utc_now,
)


###############################################################################
# Validation Record Helpers
###############################################################################

def add_validation_record(
    runtime: PipelineRuntime,
    rule_name: str,
    status: str,
    message: str,
    failed_count: Optional[int] = None,
    layer_name: Optional[str] = None,
    source_dataset: Optional[str] = None,
) -> None:
    """
    Purpose
    -------
    Add one pipeline validation record.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    rule_name:
        Validation rule name.

    status:
        Validation status. Expected values are SUCCESS, WARNING, or FAILED.

    message:
        Human-readable validation message.

    failed_count:
        Optional count of failed items.

    layer_name:
        Optional layer associated with the validation.

    source_dataset:
        Optional source configuration or dataset name.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Validation records are written at the end of the pipeline by the audit
    output helpers.
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.run_id,
            "pipeline_name": runtime.pipeline_name,
            "layer_name": layer_name or runtime.layer_name,
            "rule_name": rule_name,
            "status": status,
            "message": message,
            "failed_count": failed_count,
            "source_dataset": source_dataset,
            "event_timestamp_utc": utc_now().isoformat(),
        }
    )


###############################################################################
# Generic Validation Helpers
###############################################################################

def require_sections(
    runtime: PipelineRuntime,
    config: Dict[str, Any],
    required_sections: Iterable[str],
) -> None:
    """
    Purpose
    -------
    Validate that required top-level configuration sections exist.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    config:
        Loaded pipeline configuration.

    required_sections:
        Required top-level section names.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when required configuration sections are missing.

    Notes
    -----
    This validates the core YAML contract for the master pipeline.
    """

    missing_sections = [
        section for section in required_sections if section not in config
    ]

    if missing_sections:
        message = f"Missing required pipeline configuration sections: {missing_sections}"

        add_validation_record(
            runtime=runtime,
            rule_name="required_config_sections",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(missing_sections),
            source_dataset=str(runtime.config_path),
        )

        raise ValueError(message)

    add_validation_record(
        runtime=runtime,
        rule_name="required_config_sections",
        status=STATUS_SUCCESS,
        message="All required pipeline configuration sections are present.",
        failed_count=0,
        source_dataset=str(runtime.config_path),
    )


def require_nested_sections(
    runtime: PipelineRuntime,
    config: Dict[str, Any],
    section_name: str,
    required_subsections: Iterable[str],
) -> None:
    """
    Purpose
    -------
    Validate that required subsections exist inside a top-level section.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    config:
        Loaded pipeline configuration.

    section_name:
        Top-level section name.

    required_subsections:
        Required keys within the top-level section.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when required subsections are missing.

    Notes
    -----
    Used for validating paths.outputs, paths.metadata_outputs, and
    paths.audit_outputs.
    """

    section = config.get(section_name, {})

    if not isinstance(section, dict):
        message = f"Pipeline configuration section '{section_name}' must be a mapping."

        add_validation_record(
            runtime=runtime,
            rule_name=f"{section_name}_is_mapping",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            source_dataset=str(runtime.config_path),
        )

        raise ValueError(message)

    missing_subsections = [
        subsection for subsection in required_subsections if subsection not in section
    ]

    if missing_subsections:
        message = (
            f"Missing required subsections under '{section_name}': "
            f"{missing_subsections}"
        )

        add_validation_record(
            runtime=runtime,
            rule_name=f"{section_name}_required_subsections",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(missing_subsections),
            source_dataset=str(runtime.config_path),
        )

        raise ValueError(message)

    add_validation_record(
        runtime=runtime,
        rule_name=f"{section_name}_required_subsections",
        status=STATUS_SUCCESS,
        message=f"All required subsections are present under '{section_name}'.",
        failed_count=0,
        source_dataset=str(runtime.config_path),
    )


###############################################################################
# Pipeline Configuration Validation
###############################################################################

def validate_pipeline_config(runtime: PipelineRuntime) -> None:
    """
    Purpose
    -------
    Validate the master pipeline configuration.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when configuration validation fails.

    Notes
    -----
    This validation is performed before executing any platform layer.
    """

    config = runtime.config

    required_sections = [
        "project",
        "pipeline",
        "layers",
        "dependencies",
        "runtime",
        "logging",
        "validation",
        "metadata",
        "audit",
        "paths",
    ]

    require_sections(
        runtime=runtime,
        config=config,
        required_sections=required_sections,
    )

    require_nested_sections(
        runtime=runtime,
        config=config,
        section_name="paths",
        required_subsections=[
            "outputs",
            "metadata_outputs",
            "audit_outputs",
        ],
    )

    layers = config.get("layers", [])

    if not isinstance(layers, list) or not layers:
        message = "Pipeline configuration must define a non-empty layers list."

        add_validation_record(
            runtime=runtime,
            rule_name="layers_list_not_empty",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            source_dataset=str(runtime.config_path),
        )

        raise ValueError(message)

    add_validation_record(
        runtime=runtime,
        rule_name="layers_list_not_empty",
        status=STATUS_SUCCESS,
        message="Pipeline layer registry is present and not empty.",
        failed_count=0,
        source_dataset=str(runtime.config_path),
    )


###############################################################################
# Layer Registry Validation
###############################################################################

def validate_layer_registry(runtime: PipelineRuntime) -> None:
    """
    Purpose
    -------
    Validate the configured layer execution registry.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when layer definitions are invalid.

    Notes
    -----
    Every configured layer must define:
        - name
        - enabled
        - module
        - description
    """

    layers = runtime.config.get("layers", [])
    errors: List[str] = []

    required_keys = ["name", "enabled", "module", "description"]

    for index, layer_config in enumerate(layers):
        if not isinstance(layer_config, dict):
            errors.append(f"Layer entry at index {index} must be a mapping.")
            continue

        for key in required_keys:
            if key not in layer_config:
                errors.append(
                    f"Layer entry at index {index} is missing required key: {key}"
                )

        layer_name = layer_config.get("name")

        if layer_name is not None and str(layer_name).strip() == "":
            errors.append(f"Layer entry at index {index} has blank name.")

        module_name = layer_config.get("module")

        if module_name is not None and str(module_name).strip() == "":
            errors.append(f"Layer entry at index {index} has blank module.")

    if errors:
        message = "Pipeline layer registry validation failed: " + "; ".join(errors)

        add_validation_record(
            runtime=runtime,
            rule_name="layer_registry",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(errors),
            source_dataset=str(runtime.config_path),
        )

        raise ValueError(message)

    add_validation_record(
        runtime=runtime,
        rule_name="layer_registry",
        status=STATUS_SUCCESS,
        message="Pipeline layer registry validation passed.",
        failed_count=0,
        source_dataset=str(runtime.config_path),
    )


def validate_unique_layer_names(runtime: PipelineRuntime) -> None:
    """
    Purpose
    -------
    Validate that configured layer names are unique.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when duplicate layer names are found.

    Notes
    -----
    Unique names are required so execution records and audit records can be
    joined and interpreted reliably.
    """

    layer_names = [
        layer_config.get("name")
        for layer_config in runtime.config.get("layers", [])
        if isinstance(layer_config, dict)
    ]

    duplicates = sorted(
        {
            layer_name
            for layer_name in layer_names
            if layer_names.count(layer_name) > 1
        }
    )

    if duplicates:
        message = f"Duplicate layer names configured: {duplicates}"

        add_validation_record(
            runtime=runtime,
            rule_name="unique_layer_names",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(duplicates),
            source_dataset=str(runtime.config_path),
        )

        raise ValueError(message)

    add_validation_record(
        runtime=runtime,
        rule_name="unique_layer_names",
        status=STATUS_SUCCESS,
        message="All configured layer names are unique.",
        failed_count=0,
        source_dataset=str(runtime.config_path),
    )


###############################################################################
# Module Import Validation
###############################################################################

def validate_layer_modules(runtime: PipelineRuntime) -> None:
    """
    Purpose
    -------
    Validate that configured layer modules can be imported.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        Raised when one or more enabled layer modules cannot be imported.

    Notes
    -----
    The master pipeline dynamically imports layer modules by name. This
    validation catches missing or misspelled modules before execution begins.
    """

    errors: List[str] = []

    for layer_config in runtime.config.get("layers", []):
        if not bool(layer_config.get("enabled", True)):
            continue

        layer_name = layer_config.get("name")
        module_name = layer_config.get("module")

        try:
            importlib.import_module(str(module_name))
        except Exception as exc:
            errors.append(f"{layer_name}: failed to import {module_name}: {exc}")

    if errors:
        message = "Layer module import validation failed: " + "; ".join(errors)

        add_validation_record(
            runtime=runtime,
            rule_name="layer_module_imports",
            status=STATUS_FAILED,
            message=message,
            failed_count=len(errors),
            source_dataset=str(runtime.config_path),
        )

        raise ImportError(message)

    add_validation_record(
        runtime=runtime,
        rule_name="layer_module_imports",
        status=STATUS_SUCCESS,
        message="All enabled layer modules imported successfully.",
        failed_count=0,
        source_dataset=str(runtime.config_path),
    )


###############################################################################
# Pre-Execution and Post-Execution Validation
###############################################################################

def validate_before_execution(runtime: PipelineRuntime) -> None:
    """
    Purpose
    -------
    Run all pre-execution pipeline validations.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when configuration or registry validation fails.

    ImportError
        Raised when configured modules cannot be imported.

    Notes
    -----
    This should be called before the first platform layer is executed.
    """

    runtime.logger.info("START: Pipeline pre-execution validation")

    validate_pipeline_config(runtime)
    validate_layer_registry(runtime)
    validate_unique_layer_names(runtime)

    if bool(runtime.config.get("dependencies", {}).get("verify_configuration_files", True)):
        validate_layer_modules(runtime)

    runtime.logger.info("COMPLETE: Pipeline pre-execution validation")


def validate_layer_result(
    runtime: PipelineRuntime,
    result: PipelineBuildResult,
) -> None:
    """
    Purpose
    -------
    Validate a single layer execution result.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    result:
        Normalized layer execution result.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when result status is invalid.

    Notes
    -----
    This validation does not necessarily fail the master pipeline. It records
    layer result health so the orchestrator can decide whether to continue.
    """

    valid_statuses = {
        STATUS_SUCCESS,
        STATUS_FAILED,
        STATUS_WARNING,
        STATUS_SKIPPED,
    }

    if result.status not in valid_statuses:
        message = (
            f"Layer '{result.name}' returned invalid status: {result.status}"
        )

        add_validation_record(
            runtime=runtime,
            rule_name="layer_result_status",
            status=STATUS_FAILED,
            message=message,
            failed_count=1,
            layer_name=result.name,
        )

        raise ValueError(message)

    validation_status = (
        STATUS_SUCCESS
        if result.status == STATUS_SUCCESS
        else STATUS_WARNING
        if result.status in {STATUS_WARNING, STATUS_SKIPPED}
        else STATUS_FAILED
    )

    add_validation_record(
        runtime=runtime,
        rule_name="layer_result_status",
        status=validation_status,
        message=(
            f"Layer '{result.name}' completed with status: "
            f"{result.status}. Message: {result.message}"
        ),
        failed_count=0 if result.status == STATUS_SUCCESS else 1,
        layer_name=result.name,
    )


def validate_after_execution(
    runtime: PipelineRuntime,
    layer_results: List[PipelineBuildResult],
) -> None:
    """
    Purpose
    -------
    Run post-execution validation across all layer results.

    Parameters
    ----------
    runtime:
        Pipeline runtime.

    layer_results:
        Normalized layer execution results.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    Post-execution validation records failures and warnings but does not raise.
    The orchestrator determines final pipeline status from result statuses.
    """

    runtime.logger.info("START: Pipeline post-execution validation")

    for result in layer_results:
        try:
            validate_layer_result(runtime, result)
        except Exception as exc:
            runtime.logger.error(
                "Layer result validation failed for %s: %s",
                result.name,
                exc,
            )

    failed_count = sum(
        1 for result in layer_results if result.status == STATUS_FAILED
    )
    warning_count = sum(
        1
        for result in layer_results
        if result.status in {STATUS_WARNING, STATUS_SKIPPED}
    )

    if failed_count > 0:
        status = STATUS_FAILED
        message = f"Pipeline post-execution validation found {failed_count} failures."
    elif warning_count > 0:
        status = STATUS_WARNING
        message = f"Pipeline post-execution validation found {warning_count} warnings."
    else:
        status = STATUS_SUCCESS
        message = "Pipeline post-execution validation passed."

    add_validation_record(
        runtime=runtime,
        rule_name="post_execution_summary",
        status=status,
        message=message,
        failed_count=failed_count,
    )

    runtime.logger.info("COMPLETE: Pipeline post-execution validation")


###############################################################################
# End of File
###############################################################################