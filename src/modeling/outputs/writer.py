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
#     consolidated modeling analysis artifacts, and experiment tracking outputs.
#
###############################################################################

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

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


def _json_safe(value: Any) -> Any:
    """
    Convert values into JSON-safe structures.
    """

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, set):
        return sorted([_json_safe(v) for v in value])

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "__fspath__"):
        return str(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _stable_json(value: Any) -> str:
    """
    Serialize a value into stable JSON text.
    """

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        default=str,
    )


def _sha256_text(value: str) -> str:
    """
    Build a deterministic SHA-256 hash from text.
    """

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def build_configuration_snapshot(runtime: ModelingRuntime) -> pd.DataFrame:
    """
    Build resolved configuration snapshot for one Modeling run.
    """

    config_json = _stable_json(runtime.config)
    pipeline_config_json = _stable_json(runtime.pipeline_config)
    parallelism_config_json = _stable_json(runtime.parallelism_config)

    configuration_snapshot_id = _sha256_text(
        "|".join(
            [
                runtime.run_id,
                config_json,
                pipeline_config_json,
                parallelism_config_json,
            ]
        )
    )

    models_config = runtime.config.get("models", {})
    training_config = runtime.config.get("training", {})
    algorithms_config = training_config.get("algorithms", {})

    enabled_models = [
        model_key
        for model_key, model_config in models_config.items()
        if bool(model_config.get("enabled", True))
    ]

    enabled_algorithms = [
        algorithm_key
        for algorithm_key, algorithm_config in algorithms_config.items()
        if bool(algorithm_config.get("enabled", True))
    ]

    return pd.DataFrame(
        [
            {
                "run_id": runtime.run_id,
                "configuration_snapshot_id": configuration_snapshot_id,
                "layer_name": runtime.layer_name,
                "capability_name": runtime.capability_name,
                "domain_name": runtime.domain_name,
                "config_path": str(runtime.config_path),
                "pipeline_config_path": str(runtime.pipeline_config_path),
                "output_format": get_output_format(runtime),
                "enabled_model_count": len(enabled_models),
                "enabled_models_json": _stable_json(enabled_models),
                "enabled_algorithm_count": len(enabled_algorithms),
                "enabled_algorithms_json": _stable_json(enabled_algorithms),
                "modeling_config_json": config_json,
                "pipeline_config_json": pipeline_config_json,
                "parallelism_config_json": parallelism_config_json,
                "modeling_config_hash": _sha256_text(config_json),
                "pipeline_config_hash": _sha256_text(pipeline_config_json),
                "parallelism_config_hash": _sha256_text(parallelism_config_json),
                "event_timestamp_utc": runtime.event_timestamp_utc,
            }
        ]
    )


def build_experiment_run_history(
    runtime: ModelingRuntime,
    scoring_outputs: Dict[str, pd.DataFrame],
    configuration_snapshot_df: pd.DataFrame,
    execution_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build experiment run history for one Modeling execution.
    """

    configuration_snapshot_id = (
        configuration_snapshot_df["configuration_snapshot_id"].iloc[0]
        if not configuration_snapshot_df.empty
        else None
    )

    experiment_id = _sha256_text(
        "|".join(
            [
                str(runtime.run_id),
                str(configuration_snapshot_id),
                runtime.start_time_utc.isoformat(),
            ]
        )
    )

    execution_row = (
        execution_summary_df.iloc[0].to_dict()
        if not execution_summary_df.empty
        else {}
    )

    return pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "run_id": runtime.run_id,
                "configuration_snapshot_id": configuration_snapshot_id,
                "layer_name": runtime.layer_name,
                "capability_name": runtime.capability_name,
                "domain_name": runtime.domain_name,
                "start_time_utc": execution_row.get("start_time_utc"),
                "end_time_utc": execution_row.get("end_time_utc"),
                "duration_seconds": execution_row.get("duration_seconds"),
                "model_count": execution_row.get("model_count"),
                "scored_dataset_count": execution_row.get("scored_dataset_count"),
                "audit_record_count": execution_row.get("audit_record_count"),
                "validation_record_count": execution_row.get(
                    "validation_record_count"
                ),
                "failed_validation_count": execution_row.get(
                    "failed_validation_count"
                ),
                "status": execution_row.get("status"),
                "event_timestamp_utc": runtime.event_timestamp_utc,
            }
        ]
    )


def build_candidate_parameter_snapshot(
    runtime: ModelingRuntime,
    candidate_leaderboard_df: pd.DataFrame,
    hyperparameter_search_results_df: pd.DataFrame,
    configuration_snapshot_df: pd.DataFrame,
    experiment_run_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build candidate parameter snapshot for every model/algorithm candidate.
    """

    if candidate_leaderboard_df.empty:
        return pd.DataFrame()

    configuration_snapshot_id = (
        configuration_snapshot_df["configuration_snapshot_id"].iloc[0]
        if not configuration_snapshot_df.empty
        else None
    )

    experiment_id = (
        experiment_run_history_df["experiment_id"].iloc[0]
        if not experiment_run_history_df.empty
        else None
    )

    training_config = runtime.config.get("training", {})
    algorithms_config = training_config.get("algorithms", {})
    modeling_defaults = runtime.config.get("modeling_defaults", {})

    candidate_keys = (
        candidate_leaderboard_df[
            [
                "run_id",
                "model_key",
                "model_name",
                "target_column",
                "algorithm_key",
                "algorithm_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    rows = []

    for _, candidate in candidate_keys.iterrows():
        model_key = candidate["model_key"]
        algorithm_key = candidate["algorithm_key"]

        algorithm_config = algorithms_config.get(algorithm_key, {})
        model_config = runtime.config.get("models", {}).get(model_key, {})

        search_params_df = pd.DataFrame()

        if not hyperparameter_search_results_df.empty:
            search_params_df = hyperparameter_search_results_df[
                (
                    hyperparameter_search_results_df["model_key"] == model_key
                )
                & (
                    hyperparameter_search_results_df["algorithm_key"]
                    == algorithm_key
                )
            ]

        best_search_params = None

        if not search_params_df.empty and "is_best" in search_params_df.columns:
            best_rows = search_params_df[
                search_params_df["is_best"].astype(bool)
            ]

            if not best_rows.empty:
                best_search_params = best_rows.iloc[0].get("params")

        candidate_snapshot_json = _stable_json(
            {
                "model_config": model_config,
                "algorithm_config": algorithm_config,
                "modeling_defaults": modeling_defaults,
                "training_config": training_config,
                "best_search_params": best_search_params,
            }
        )

        rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": runtime.run_id,
                "configuration_snapshot_id": configuration_snapshot_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "model_key": model_key,
                "model_name": candidate["model_name"],
                "target_column": candidate["target_column"],
                "algorithm_key": algorithm_key,
                "algorithm_name": candidate["algorithm_name"],
                "candidate_parameter_snapshot_id": _sha256_text(
                    "|".join(
                        [
                            runtime.run_id,
                            str(model_key),
                            str(algorithm_key),
                            candidate_snapshot_json,
                        ]
                    )
                ),
                "algorithm_enabled": bool(
                    algorithm_config.get("enabled", True)
                ),
                "best_search_params": (
                    None if best_search_params is None else str(best_search_params)
                ),
                "algorithm_config_json": _stable_json(algorithm_config),
                "model_config_json": _stable_json(model_config),
                "modeling_defaults_json": _stable_json(modeling_defaults),
                "training_config_json": _stable_json(training_config),
                "candidate_snapshot_json": candidate_snapshot_json,
                "event_timestamp_utc": runtime.event_timestamp_utc,
            }
        )

    return pd.DataFrame(rows)


def build_champion_challenger_history(
    runtime: ModelingRuntime,
    champion_summary_df: pd.DataFrame,
    candidate_leaderboard_df: pd.DataFrame,
    configuration_snapshot_df: pd.DataFrame,
    experiment_run_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build champion/challenger history from champion summary and leaderboard.
    """

    if champion_summary_df.empty:
        return pd.DataFrame()

    configuration_snapshot_id = (
        configuration_snapshot_df["configuration_snapshot_id"].iloc[0]
        if not configuration_snapshot_df.empty
        else None
    )

    experiment_id = (
        experiment_run_history_df["experiment_id"].iloc[0]
        if not experiment_run_history_df.empty
        else None
    )

    rows = []

    for _, champion in champion_summary_df.iterrows():
        model_key = champion.get("model_key")

        model_candidates = pd.DataFrame()

        if not candidate_leaderboard_df.empty:
            model_candidates = candidate_leaderboard_df[
                candidate_leaderboard_df["model_key"] == model_key
            ]

        challenger_algorithms = []

        if not model_candidates.empty:
            candidate_algorithms = (
                model_candidates[
                    ["algorithm_key", "algorithm_name", "is_champion"]
                ]
                .drop_duplicates()
                .copy()
            )

            challenger_algorithms = [
                {
                    "algorithm_key": row["algorithm_key"],
                    "algorithm_name": row["algorithm_name"],
                }
                for _, row in candidate_algorithms.iterrows()
                if not bool(row.get("is_champion", False))
            ]

        rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": runtime.run_id,
                "configuration_snapshot_id": configuration_snapshot_id,
                "layer_name": runtime.layer_name,
                "domain_name": runtime.domain_name,
                "model_key": model_key,
                "model_name": champion.get("model_name"),
                "target_column": champion.get("target_column"),
                "champion_algorithm_key": champion.get(
                    "champion_algorithm_key"
                ),
                "champion_algorithm_name": champion.get(
                    "champion_algorithm_name"
                ),
                "selection_metric": champion.get("selection_metric"),
                "selection_metric_value": champion.get(
                    "selection_metric_value"
                ),
                "prediction_threshold": champion.get("prediction_threshold"),
                "probability_calibration_enabled": champion.get(
                    "probability_calibration_enabled"
                ),
                "probability_calibration_applied": champion.get(
                    "probability_calibration_applied"
                ),
                "probability_calibration_method": champion.get(
                    "probability_calibration_method"
                ),
                "challenger_count": len(challenger_algorithms),
                "challenger_algorithms_json": _stable_json(
                    challenger_algorithms
                ),
                "champion_challenger_history_id": _sha256_text(
                    "|".join(
                        [
                            runtime.run_id,
                            str(model_key),
                            str(champion.get("champion_algorithm_key")),
                            str(champion.get("selection_metric")),
                            str(champion.get("selection_metric_value")),
                        ]
                    )
                ),
                "event_timestamp_utc": runtime.event_timestamp_utc,
            }
        )

    return pd.DataFrame(rows)


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

    member_level_explanations_summary_df = (
        pd.concat(runtime.member_level_explanations_frames, ignore_index=True)
        if runtime.member_level_explanations_frames
        else pd.DataFrame()
    )

    execution_summary_df = build_execution_summary(
        runtime=runtime,
        scoring_outputs=scoring_outputs,
    )

    configuration_snapshot_df = build_configuration_snapshot(runtime)

    experiment_run_history_df = build_experiment_run_history(
        runtime=runtime,
        scoring_outputs=scoring_outputs,
        configuration_snapshot_df=configuration_snapshot_df,
        execution_summary_df=execution_summary_df,
    )

    candidate_parameter_snapshot_df = build_candidate_parameter_snapshot(
        runtime=runtime,
        candidate_leaderboard_df=candidate_leaderboard_df,
        hyperparameter_search_results_df=hyperparameter_search_results_df,
        configuration_snapshot_df=configuration_snapshot_df,
        experiment_run_history_df=experiment_run_history_df,
    )

    champion_challenger_history_df = build_champion_challenger_history(
        runtime=runtime,
        champion_summary_df=champion_summary_df,
        candidate_leaderboard_df=candidate_leaderboard_df,
        configuration_snapshot_df=configuration_snapshot_df,
        experiment_run_history_df=experiment_run_history_df,
    )

    population_stability_index_summary_df = (
        pd.concat(runtime.population_stability_index_frames, ignore_index=True)
        if runtime.population_stability_index_frames
        else pd.DataFrame()
    )

    ks_drift_summary_df = (
        pd.concat(runtime.ks_drift_frames, ignore_index=True)
        if runtime.ks_drift_frames
        else pd.DataFrame()
    )   

    prediction_score_drift_summary_df = (
        pd.concat(runtime.prediction_score_drift_frames, ignore_index=True)
        if runtime.prediction_score_drift_frames
        else pd.DataFrame()
    )

    model_performance_drift_summary_df = (
        pd.concat(runtime.model_performance_drift_frames, ignore_index=True)
        if runtime.model_performance_drift_frames
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
        "modeling_execution_summary": execution_summary_df,
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
        "member_level_explanations_summary": member_level_explanations_summary_df,
        "configuration_snapshot": configuration_snapshot_df,
        "experiment_run_history": experiment_run_history_df,
        "candidate_parameter_snapshot": candidate_parameter_snapshot_df,
        "champion_challenger_history": champion_challenger_history_df,
        "population_stability_index_summary": population_stability_index_summary_df,
        "ks_drift_summary": ks_drift_summary_df,
        "prediction_score_drift_summary": prediction_score_drift_summary_df,
        "model_performance_drift_summary": model_performance_drift_summary_df,
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


###############################################################################
# End of File
###############################################################################