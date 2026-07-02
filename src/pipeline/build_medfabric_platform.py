###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/pipeline/build_medfabric_platform.py
#
# Layer:
#     Enterprise Pipeline
#
# Purpose:
#     Master orchestrator for the complete MedFabric platform.
#
# Run:
#     python -m src.pipeline.build_medfabric_platform
#
###############################################################################

from __future__ import annotations

import importlib
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.pipeline.common.audit import (
    add_failed_audit,
    add_layer_completion_audit,
    add_layer_start_audit,
    add_running_audit,
    add_skipped_audit,
    add_success_audit,
    add_warning_audit,
    write_audit_outputs,
)
from src.pipeline.common.io import (
    get_output_format,
    prepare_pipeline_output_directories,
    write_business_outputs,
)
from src.pipeline.common.metadata import (
    add_layer_result_from_build_result,
    register_layer_execution_rule,
    register_output_asset,
    write_metadata_outputs,
    build_layer_execution_summary,
    build_pipeline_execution_summary,
)
from src.pipeline.common.runtime import (
    DEFAULT_CONFIG_PATH,
    PipelineBuildResult,
    PipelineRuntime,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_WARNING,
    initialize_pipeline_runtime,
    normalize_layer_result,
)
from src.pipeline.common.validation import (
    validate_after_execution,
    validate_before_execution,
)


###############################################################################
# Layer Execution Helpers
###############################################################################

def find_build_callable(module: Any) -> Any:
    """
    Find the most likely build function inside a layer module.
    """

    preferred_names = [
        "main",
        "build_data_generation",
        "build_data_platform",
        "build_feature_store",
        "build_modeling_layer",
        "build_semantic_layer",
        "build_analytics_platform",
        "build_insights_platform",
    ]

    for name in preferred_names:
        if hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"No supported build callable found in module: {module.__name__}")


def execute_layer_module(layer_config: Dict[str, Any]) -> Any:
    """
    Import and execute one configured layer module.
    """

    module_name = layer_config["module"]
    module = importlib.import_module(module_name)
    build_callable = find_build_callable(module)

    if build_callable.__name__ == "main":
        return build_callable()

    return build_callable()


def should_stop_after_failure(runtime: PipelineRuntime) -> bool:
    """
    Return whether the pipeline should stop after the first failed layer.
    """

    pipeline_config = runtime.config.get("pipeline", {})

    return bool(
        pipeline_config.get("fail_fast", True)
        or pipeline_config.get("stop_after_first_failure", True)
    )


def is_layer_enabled(layer_config: Dict[str, Any]) -> bool:
    """
    Return whether a configured layer is enabled.
    """

    return bool(layer_config.get("enabled", True))


###############################################################################
# Platform Status Helpers
###############################################################################

def determine_pipeline_status(layer_results: List[PipelineBuildResult]) -> str:
    """
    Determine final pipeline status from layer results.
    """

    statuses = [result.status for result in layer_results]

    if any(status == STATUS_FAILED for status in statuses):
        return STATUS_FAILED

    if any(status in {STATUS_WARNING, STATUS_SKIPPED} for status in statuses):
        return STATUS_WARNING

    return STATUS_SUCCESS


def build_pipeline_message(status: str) -> str:
    """
    Build final human-readable pipeline message.
    """

    if status == STATUS_SUCCESS:
        return "MedFabric platform completed successfully."

    if status == STATUS_WARNING:
        return "MedFabric platform completed with warnings."

    return "MedFabric platform failed."


###############################################################################
# Main Layer Orchestration
###############################################################################

def run_configured_layers(runtime: PipelineRuntime) -> List[PipelineBuildResult]:
    """
    Execute all configured platform layers in YAML order.
    """

    logger = runtime.logger
    layer_results: List[PipelineBuildResult] = []

    layers = runtime.config.get("layers", [])

    for layer_config in layers:
        layer_name = layer_config.get("name")
        module_name = layer_config.get("module")

        register_layer_execution_rule(runtime, layer_config)

        if not is_layer_enabled(layer_config):
            result = PipelineBuildResult(
                name=layer_name,
                status=STATUS_SKIPPED,
                message=f"Layer disabled in configuration: {layer_name}",
            )

            layer_results.append(result)

            add_skipped_audit(
                runtime=runtime,
                step_name=f"skip_layer:{layer_name}",
                message=result.message,
                layer_name=layer_name,
                source_layer="Enterprise Pipeline",
                source_dataset=module_name,
            )

            add_layer_result_from_build_result(
                runtime=runtime,
                result=result,
                module_name=module_name,
                duration_seconds=0.0,
            )

            continue

        logger.info("=" * 80)
        logger.info("START LAYER: %s", layer_name)
        logger.info("MODULE: %s", module_name)
        logger.info("=" * 80)

        add_layer_start_audit(
            runtime=runtime,
            layer_name=layer_name,
            module_name=module_name,
        )

        start_time = datetime.utcnow()

        try:
            raw_result = execute_layer_module(layer_config)
            result = normalize_layer_result(layer_name, raw_result)

        except SystemExit as exc:
            exit_code = int(exc.code or 0)

            if exit_code == 0:
                result = PipelineBuildResult(
                    name=layer_name,
                    status=STATUS_SUCCESS,
                    message=f"{layer_name} completed successfully.",
                )
            else:
                result = PipelineBuildResult(
                    name=layer_name,
                    status=STATUS_FAILED,
                    message=f"{layer_name} exited with code {exit_code}.",
                )

        except Exception as exc:
            result = PipelineBuildResult(
                name=layer_name,
                status=STATUS_FAILED,
                message=f"{layer_name} failed: {exc}",
            )

            logger.error("Layer failed: %s", layer_name)
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())

        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()

        layer_results.append(result)

        add_layer_completion_audit(
            runtime=runtime,
            layer_name=layer_name,
            module_name=module_name,
            result=result,
        )

        add_layer_result_from_build_result(
            runtime=runtime,
            result=result,
            module_name=module_name,
            duration_seconds=duration_seconds,
        )

        logger.info(
            "COMPLETE LAYER: %s | Status: %s | Duration: %.2f seconds",
            layer_name,
            result.status,
            duration_seconds,
        )

        if result.status == STATUS_FAILED and should_stop_after_failure(runtime):
            logger.error("Stopping pipeline after failed layer: %s", layer_name)
            break

    return layer_results


###############################################################################
# Output Builders
###############################################################################

def build_pipeline_output_assets(
    runtime: PipelineRuntime,
    layer_results: List[PipelineBuildResult],
) -> Dict[str, pd.DataFrame]:
    """
    Build pipeline-level output assets.
    """

    pipeline_execution_summary = build_pipeline_execution_summary(
        runtime=runtime,
        layer_results=layer_results,
    )

    layer_execution_summary = build_layer_execution_summary(runtime)

    return {
        "pipeline_execution_summary": pipeline_execution_summary,
        "layer_execution_summary": layer_execution_summary,
    }


def register_pipeline_outputs(
    runtime: PipelineRuntime,
    output_assets: Dict[str, pd.DataFrame],
) -> None:
    """
    Register pipeline outputs in dataset inventory.
    """

    for dataset_name, dataframe in output_assets.items():
        register_output_asset(
            runtime=runtime,
            dataset_name=dataset_name,
            dataframe=dataframe,
            dataset_type="pipeline_output",
            status=STATUS_SUCCESS,
            message=f"Pipeline output built successfully: {dataset_name}",
            source_layer="Enterprise Pipeline",
            source_dataset=str(runtime.config_path),
        )


###############################################################################
# Main Build Function
###############################################################################

def build_medfabric_platform(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> PipelineBuildResult:
    """
    Build the complete MedFabric platform end-to-end.
    """

    runtime: Optional[PipelineRuntime] = None
    layer_results: List[PipelineBuildResult] = []

    try:
        runtime = initialize_pipeline_runtime(config_path=config_path)
        logger = runtime.logger

        logger.info("=" * 80)
        logger.info("MedFabric Enterprise Pipeline started")
        logger.info("=" * 80)

        prepare_pipeline_output_directories(runtime)

        add_running_audit(
            runtime=runtime,
            step_name="build_medfabric_platform",
            message="MedFabric Enterprise Pipeline started.",
            layer_name=runtime.layer_name,
            source_layer="Enterprise Pipeline",
            source_dataset=str(runtime.config_path),
        )

        if bool(runtime.config.get("pipeline", {}).get("validate_before_execution", True)):
            validate_before_execution(runtime)

        layer_results = run_configured_layers(runtime)

        if bool(runtime.config.get("pipeline", {}).get("validate_after_execution", True)):
            validate_after_execution(runtime, layer_results)

        pipeline_status = determine_pipeline_status(layer_results)
        pipeline_message = build_pipeline_message(pipeline_status)

        output_format = get_output_format(runtime)

        output_assets = build_pipeline_output_assets(
            runtime=runtime,
            layer_results=layer_results,
        )

        register_pipeline_outputs(
            runtime=runtime,
            output_assets=output_assets,
        )

        write_business_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
        )

        if pipeline_status == STATUS_SUCCESS:
            add_success_audit(
                runtime=runtime,
                step_name="build_medfabric_platform",
                message=pipeline_message,
                layer_name=runtime.layer_name,
                row_count=sum(result.row_count for result in layer_results),
                column_count=sum(result.column_count for result in layer_results),
                source_layer="Enterprise Pipeline",
                source_dataset=str(runtime.config_path),
            )
        elif pipeline_status == STATUS_WARNING:
            add_warning_audit(
                runtime=runtime,
                step_name="build_medfabric_platform",
                message=pipeline_message,
                layer_name=runtime.layer_name,
                row_count=sum(result.row_count for result in layer_results),
                column_count=sum(result.column_count for result in layer_results),
                source_layer="Enterprise Pipeline",
                source_dataset=str(runtime.config_path),
            )
        else:
            add_failed_audit(
                runtime=runtime,
                step_name="build_medfabric_platform",
                message=pipeline_message,
                layer_name=runtime.layer_name,
                row_count=sum(result.row_count for result in layer_results),
                column_count=sum(result.column_count for result in layer_results),
                source_layer="Enterprise Pipeline",
                source_dataset=str(runtime.config_path),
            )

        write_metadata_outputs(
            runtime=runtime,
            output_assets=output_assets,
            output_format=output_format,
            dataset_inventory_name="pipeline_dataset_inventory",
            column_dictionary_name="pipeline_column_dictionary",
            rule_catalog_name="pipeline_rule_catalog",
        )

        write_audit_outputs(
            runtime=runtime,
            layer_results=layer_results,
            output_format=output_format,
            audit_records_name="pipeline_audit_records",
            validation_results_name="pipeline_validation_results",
            execution_history_name="pipeline_execution_history",
        )

        logger.info("=" * 80)
        logger.info("MedFabric Enterprise Pipeline completed")
        logger.info("Status: %s", pipeline_status)
        logger.info("=" * 80)

        return PipelineBuildResult(
            name="medfabric_platform",
            status=pipeline_status,
            message=pipeline_message,
            row_count=sum(result.row_count for result in layer_results),
            column_count=sum(result.column_count for result in layer_results),
        )

    except Exception as exc:
        if runtime is not None:
            logger = runtime.logger

            logger.error("=" * 80)
            logger.error("MedFabric Enterprise Pipeline failed")
            logger.error("Error: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            logger.error("=" * 80)

            add_failed_audit(
                runtime=runtime,
                step_name="build_medfabric_platform",
                message=str(exc),
                layer_name=runtime.layer_name,
                source_layer="Enterprise Pipeline",
                source_dataset=str(runtime.config_path),
            )

            try:
                output_format = get_output_format(runtime)

                write_audit_outputs(
                    runtime=runtime,
                    layer_results=layer_results,
                    output_format=output_format,
                    audit_records_name="pipeline_audit_records",
                    validation_results_name="pipeline_validation_results",
                    execution_history_name="pipeline_execution_history",
                )

            except Exception as audit_exc:
                logger.error("Failed to write pipeline audit outputs: %s", audit_exc)

        return PipelineBuildResult(
            name="medfabric_platform",
            status=STATUS_FAILED,
            message=str(exc),
        )


###############################################################################
# CLI Entry Point
###############################################################################

def main() -> None:
    """
    Command-line entry point.
    """

    config_path = os.environ.get(
        "MEDFABRIC_PIPELINE_CONFIG",
        DEFAULT_CONFIG_PATH,
    )

    result = build_medfabric_platform(config_path=config_path)

    if result.status in {STATUS_SUCCESS, STATUS_WARNING}:
        print(result.message)
        return

    print(f"MedFabric platform failed: {result.message}")
    sys.exit(1)


if __name__ == "__main__":
    main()