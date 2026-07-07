###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/build_modeling_layer.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Lightweight orchestrator for the MedFabric Modeling layer.
#
# Run:
#     python -m src.modeling.build_modeling_layer
#
###############################################################################

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Optional

from src.modeling.common.audit import add_audit_record
from src.modeling.common.constants import (
    DEFAULT_CONFIG_PATH,
    STATUS_FAILED,
    STATUS_SUCCESS,
)
from src.modeling.common.runtime import BuildResult, ModelingRuntime
from src.modeling.common.timing import add_step_timing_record
from src.modeling.execution.model_runner import run_models
from src.modeling.feature_matrix.builder import build_feature_matrix
from src.modeling.inputs.loader import load_input_datasets
from src.modeling.outputs.writer import (
    write_core_modeling_outputs,
    write_metadata_and_audit_outputs,
)
from src.modeling.runtime.initializer import initialize_runtime


###############################################################################
# Main Orchestration
###############################################################################

def build_modeling_layer(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build the complete MedFabric Modeling Framework.
    """

    runtime: Optional[ModelingRuntime] = None

    try:
        runtime = initialize_runtime(config_path)

        runtime.logger.info("Configuration path: %s", runtime.config_path)
        runtime.logger.info(
            "Architectural check: Modeling consumes Feature Store only."
        )

        step_start = time.perf_counter()
        datasets = load_input_datasets(runtime)
        add_step_timing_record(
            runtime=runtime,
            step_name="load_input_datasets",
            model_key=None,
            duration_seconds=time.perf_counter() - step_start,
        )

        step_start = time.perf_counter()
        feature_matrix = build_feature_matrix(runtime, datasets)
        add_step_timing_record(
            runtime=runtime,
            step_name="build_feature_matrix",
            model_key=None,
            duration_seconds=time.perf_counter() - step_start,
        )

        step_start = time.perf_counter()
        write_core_modeling_outputs(runtime, feature_matrix)
        add_step_timing_record(
            runtime=runtime,
            step_name="write_core_modeling_outputs",
            model_key=None,
            duration_seconds=time.perf_counter() - step_start,
        )

        step_start = time.perf_counter()
        scoring_outputs = run_models(runtime, feature_matrix)
        add_step_timing_record(
            runtime=runtime,
            step_name="run_models",
            model_key=None,
            duration_seconds=time.perf_counter() - step_start,
        )

        add_audit_record(
            runtime=runtime,
            step_name="build_modeling_layer",
            status=STATUS_SUCCESS,
            message="Modeling Framework completed successfully.",
        )

        step_start = time.perf_counter()

        add_step_timing_record(
            runtime=runtime,
            step_name="write_metadata_and_audit_outputs",
            model_key=None,
            duration_seconds=0.0,
            message=(
                "Timing record created before metadata write so it is included "
                "in the current output."
            ),
        )

        write_metadata_and_audit_outputs(runtime, scoring_outputs)

        runtime.step_timing_records[-1]["duration_seconds"] = float(
            time.perf_counter() - step_start
        )

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