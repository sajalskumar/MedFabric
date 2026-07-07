###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/outputs/writer.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Writes core Modeling Framework outputs, metadata outputs, audit outputs,
#     and consolidated modeling analysis artifacts.
#
###############################################################################

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.modeling.common.audit import add_audit_record
from src.modeling.common.constants import (
    DEFAULT_OUTPUT_FORMAT,
    STATUS_SUCCESS,
    STATUS_WARNING,
)
from src.modeling.common.io_utils import write_dataset
from src.modeling.common.output_paths import get_output_path, output_path_with_format
from src.modeling.common.runtime import ModelingRuntime
from src.modeling.common.timing import utc_now
from src.modeling.registry.model_registry import build_model_registry_dataframe


def get_output_format(runtime: ModelingRuntime) -> str:
    """
    Return configured Modeling output format.
    """

    return runtime.config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
    )


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
        1
        for record in runtime.validation_records
        if record.get("status") == "FAILED"
    )

    return pd.DataFrame(
        [
            {
                "run_id": runtime.run_id,
                "layer_name": runtime.layer_name,
                "capability_name": runtime.capability_name,
                "domain_name": runtime.domain_name,
                "config_path": str(runtime.config_path),
                "pipeline_config_path": str(runtime.pipeline_config_path),
                "start_time_utc": runtime.start_time_utc.isoformat(),
                "end_time_utc": end_time.isoformat(),
                "duration_seconds": duration_seconds,
                "parallel_execution": runtime.parallelism_config.get(
                    "parallel_execution"
                ),
                "max_parallel_workers": runtime.parallelism_config.get(
                    "max_parallel_workers"
                ),
                "parallel_strategy": runtime.parallelism_config.get(
                    "parallel_strategy"
                ),
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
    Write Modeling metadata, audit, and enterprise modeling outputs.
    """

    output_format = get_output_format(runtime)

    candidate_leaderboard_df = (
        pd.concat(runtime.candidate_leaderboard_frames, ignore_index=True)
        if runtime.candidate_leaderboard_frames
        else pd.DataFrame()
    )

    champion_summary_df = (
        pd.concat(runtime.champion_summary_frames, ignore_index=True)
        if runtime.champion_summary_frames
        else pd.DataFrame()
    )

    confusion_matrix_summary_df = (
        pd.concat(runtime.confusion_matrix_frames, ignore_index=True)
        if runtime.confusion_matrix_frames
        else pd.DataFrame()
    )

    cross_validation_fold_metrics_df = (
        pd.concat(runtime.cross_validation_fold_frames, ignore_index=True)
        if runtime.cross_validation_fold_frames
        else pd.DataFrame()
    )

    cross_validation_summary_df = (
        pd.concat(runtime.cross_validation_summary_frames, ignore_index=True)
        if runtime.cross_validation_summary_frames
        else pd.DataFrame()
    )

    target_leakage_report_df = (
        pd.concat(runtime.target_leakage_report_frames, ignore_index=True)
        if runtime.target_leakage_report_frames
        else pd.DataFrame()
    )

    target_quality_report_df = (
        pd.concat(runtime.target_quality_report_frames, ignore_index=True)
        if runtime.target_quality_report_frames
        else pd.DataFrame()
    )

    hyperparameter_search_results_df = (
        pd.concat(runtime.hyperparameter_search_frames, ignore_index=True)
        if runtime.hyperparameter_search_frames
        else pd.DataFrame()
    )

    model_explainability_summary_df = (
        pd.concat(runtime.model_explainability_frames, ignore_index=True)
        if runtime.model_explainability_frames
        else pd.DataFrame()
    )

    model_executive_explainability_summary_df = (
        pd.concat(runtime.model_executive_explainability_frames, ignore_index=True)
        if runtime.model_executive_explainability_frames
        else pd.DataFrame()
    )

    model_drift_baseline_df = (
        pd.concat(runtime.model_drift_baseline_frames, ignore_index=True)
        if runtime.model_drift_baseline_frames
        else pd.DataFrame()
    )

    lift_gain_decile_summary_df = (
        pd.concat(runtime.lift_gain_frames, ignore_index=True)
        if runtime.lift_gain_frames
        else pd.DataFrame()
    )

    permutation_importance_summary_df = (
        pd.concat(runtime.permutation_importance_frames, ignore_index=True)
        if runtime.permutation_importance_frames
        else pd.DataFrame()
    )

    model_monitoring_summary_df = (
        pd.concat(runtime.model_monitoring_summary_frames, ignore_index=True)
        if runtime.model_monitoring_summary_frames
        else pd.DataFrame()
    )

    shap_explainability_summary_df = (
        pd.concat(runtime.shap_explainability_frames, ignore_index=True)
        if runtime.shap_explainability_frames
        else pd.DataFrame()
    )

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
            runtime=runtime,
            scoring_outputs=scoring_outputs,
        ),
    }

    modeling_step_timings_df = pd.DataFrame(runtime.step_timing_records)

    modeling_output_assets = {
        "candidate_model_leaderboard": candidate_leaderboard_df,
        "modeling_step_timings": modeling_step_timings_df,
        "champion_model_summary": champion_summary_df,
        "cross_validation_fold_metrics": cross_validation_fold_metrics_df,
        "cross_validation_summary": cross_validation_summary_df,
        "hyperparameter_search_results": hyperparameter_search_results_df,
        "target_leakage_report": target_leakage_report_df,
        "target_quality_report": target_quality_report_df,
        "model_explainability_summary": model_explainability_summary_df,
        "model_explainability_executive_summary": model_executive_explainability_summary_df,
        "confusion_matrix_summary": confusion_matrix_summary_df,
        "lift_gain_decile_summary": lift_gain_decile_summary_df,
        "permutation_importance_summary": permutation_importance_summary_df,
        "model_monitoring_summary": model_monitoring_summary_df,
        "shap_explainability_summary": shap_explainability_summary_df,
    }

    if not model_drift_baseline_df.empty:
        modeling_output_assets["model_drift_baseline"] = model_drift_baseline_df

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

    for output_name, dataframe in modeling_output_assets.items():
        output_path = output_path_with_format(
            get_output_path(runtime, "outputs", output_name),
            output_format,
        )

        write_dataset(dataframe, output_path, output_format)

        runtime.logger.info(
            "Wrote modeling output: %s | Rows: %s | Path: %s",
            output_name,
            len(dataframe),
            output_path,
        )