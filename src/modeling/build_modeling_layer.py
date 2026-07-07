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
#     This builder coordinates:
#       - Feature Store input loading
#       - Modeling feature matrix construction
#       - Target generation
#       - Multi-algorithm training
#       - Champion model selection
#       - Population scoring
#       - Feature importance extraction
#       - Model registry creation
#       - Metadata and audit output writing
#
# Parallelism Scope:
#     - Project-wide parallel configuration is read from config/pipeline.yaml.
#     - Modeling currently uses parallelism only for candidate algorithm training.
#     - Scoring remains sequential until Modeling is fully stabilized.
#
# Run:
#     python -m src.modeling.build_modeling_layer
#
###############################################################################

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import time

from src.common.parallel_utils import resolve_parallelism_config
from src.common.pipeline_context import create_pipeline_context
from src.modeling.evaluation.feature_importance import build_feature_importance_output
from src.modeling.registry.model_registry import (
    build_model_registry_dataframe,
    build_model_registry_record,
)
from src.modeling.scoring.scorer import score_population
from src.modeling.targets.leakage_detection import (
    build_target_leakage_report_for_models,
    get_model_specific_safe_feature_columns,
    get_target_output_column,
)
from src.modeling.targets.target_builder import build_targets_for_enabled_models
from src.modeling.training.trainer import train_model_candidates
from src.modeling.evaluation.build_feature_baseline_statistics import (
    build_feature_baseline_statistics,
)
from src.modeling.evaluation.build_target_quality_report import (
    build_target_quality_report,
)

from src.modeling.evaluation.build_model_explainability_summary import (
    build_model_explainability_summary,
    build_model_explainability_executive_summary,
)

from src.modeling.evaluation.build_model_drift_baseline import (
    build_model_drift_baseline,
)

from src.modeling.evaluation.confusion_matrix import build_confusion_matrix_output

from src.modeling.evaluation.build_lift_gain_analysis import build_lift_gain_analysis

from src.modeling.evaluation.build_permutation_importance import build_permutation_importance

from src.modeling.evaluation.build_model_monitoring_summary import (
    build_model_monitoring_summary,
)

from src.modeling.evaluation.build_shap_explainability import build_shap_explainability

from src.modeling.common.audit import add_audit_record, add_dataset_record, add_validation_record
from src.modeling.common.configuration import load_optional_yaml_config, load_runtime_modeling_config, validate_config
from src.modeling.common.io_utils import read_dataset, save_pickle_object, write_dataset
from src.modeling.common.logging_utils import configure_logging
from src.modeling.common.output_paths import get_model_output_config, get_output_path, normalize_path, output_path_with_format
from src.modeling.common.runtime import BuildResult, ModelingRuntime
from src.modeling.common.timing import add_step_timing_record, utc_now
from src.modeling.common.constants import (
    DEFAULT_CAPABILITY_NAME,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_NAME,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PIPELINE_CONFIG_PATH,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_WARNING,
)

from src.modeling.feature_matrix.builder import build_feature_matrix

from src.modeling.inputs.loader import load_input_datasets
###############################################################################
# Runtime Initialization
###############################################################################


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> ModelingRuntime:
    """
    Initialize Modeling runtime using global PipelineContext.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    pipeline_config_path = normalize_path(project_root, DEFAULT_PIPELINE_CONFIG_PATH)

    pipeline_context = create_pipeline_context(
        pipeline_name="MedFabric Modeling Framework",
    )

    run_id = pipeline_context.run_id

    config = load_runtime_modeling_config(config_path)
    pipeline_config = load_optional_yaml_config(pipeline_config_path)

    validate_config(config)

    modeling_config = config.get("modeling", {})

    capability_name = modeling_config.get(
        "capability_name",
        modeling_config.get("layer_name", DEFAULT_CAPABILITY_NAME),
    )

    layer_name = modeling_config.get("layer_name", capability_name)
    domain_name = modeling_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(
        project_root=project_root,
        config=config,
        run_id=run_id,
    )

    start_time_utc = utc_now()
    event_timestamp_utc = start_time_utc.isoformat()

    parallelism_config = resolve_parallelism_config(
        pipeline_config=pipeline_config,
    )

    runtime = ModelingRuntime(
        run_id=run_id,
        project_root=project_root,
        config_path=config_path,
        pipeline_config_path=pipeline_config_path,
        start_time_utc=start_time_utc,
        event_timestamp_utc=event_timestamp_utc,
        config=config,
        pipeline_config=pipeline_config,
        parallelism_config=parallelism_config,
        logger=logger,
        layer_name=layer_name,
        domain_name=domain_name,
        capability_name=capability_name
    )

    logger.info("Global pipeline run ID resolved from PipelineContext: %s", run_id)
    logger.info("Runtime event timestamp UTC: %s", event_timestamp_utc)
    logger.info("Pipeline config path: %s", pipeline_config_path)
    logger.info("Parallelism config: %s", parallelism_config)

    add_audit_record(
        runtime=runtime,
        step_name="initialize_runtime",
        status=STATUS_SUCCESS,
        message="Modeling runtime initialized successfully using PipelineContext.",
    )

    return runtime



def get_output_format(runtime: ModelingRuntime) -> str:
    """
    Return configured Modeling output format.
    """

    return runtime.config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
    )







def get_selection_metric(
    modeling_defaults: Dict[str, Any],
    training_config: Dict[str, Any],
) -> str:
    """
    Resolve champion selection metric.
    """

    return (
        training_config
        .get("metrics", {})
        .get(
            "primary_metric",
            modeling_defaults.get("selection_metric", "roc_auc"),
        )
    )


###############################################################################
# Modeling Execution
###############################################################################

def get_algorithms_config(runtime: ModelingRuntime) -> Dict[str, Any]:
    """
    Return algorithm configuration from modeling.yaml.
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
    Execute Target, Target Quality, Leakage, Training, Scoring, Feature Importance,
    Feature Baseline, Drift Baseline, and Registry workflows.
    """

    models_config = runtime.config.get("models", {})
    modeling_defaults = dict(runtime.config.get("modeling_defaults", {}))
    training_config = runtime.config.get("training", {})
    member_key = runtime.config.get("join_keys", {}).get("member_key", "member_id")
    risk_tiers_config = runtime.config.get("risk_tiers", {})
    leakage_config = runtime.config.get("leakage_detection", {})
    drift_config = runtime.config.get("drift", {})
    output_format = get_output_format(runtime)

    selection_metric = get_selection_metric(
        modeling_defaults=modeling_defaults,
        training_config=training_config,
    )

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

    target_quality_targets = []

    for model_key, model_config in models_config.items():
        if not bool(model_config.get("enabled", True)):
            continue

        target_quality_targets.append(
            {
                "target_name": model_key,
                "target_column": get_target_output_column(model_config),
                "problem_type": model_config.get("model_type", "classification"),
            }
        )

    target_quality_report_df = build_target_quality_report(
        dataframe=modeling_frame,
        targets=target_quality_targets,
        run_id=runtime.run_id,
        layer_name=runtime.layer_name,
        domain_name=runtime.domain_name,
    )

    runtime.target_quality_report_frames.append(target_quality_report_df)

    add_dataset_record(
        runtime=runtime,
        dataset_name="target_quality_report",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(target_quality_report_df),
        column_count=len(target_quality_report_df.columns),
        message="Target quality report built successfully.",
    )

    add_validation_record(
        runtime=runtime,
        dataset_name="modeling_targets",
        rule_name="target_quality_report",
        status=STATUS_SUCCESS,
        message=f"Target quality report built for {len(target_quality_report_df)} targets.",
    )

    target_leakage_report_df = build_target_leakage_report_for_models(
        dataframe=modeling_frame,
        models_config=models_config,
        member_key=member_key,
        target_columns=target_columns,
        leakage_config=leakage_config,
    )

    runtime.target_leakage_report_frames.append(target_leakage_report_df)

    add_validation_record(
        runtime=runtime,
        dataset_name="modeling_feature_matrix",
        rule_name="target_leakage_hardening",
        status=STATUS_SUCCESS,
        message=(
            "Model-specific target leakage report built. "
            f"Excluded records: {len(target_leakage_report_df)}"
        ),
    )

    scoring_outputs: Dict[str, pd.DataFrame] = {}

    get_algorithms_config(runtime)

    all_safe_feature_columns: List[str] = []

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

        model_start_time = time.perf_counter()

        model_feature_columns = get_model_specific_safe_feature_columns(
            dataframe=modeling_frame,
            model_key=model_key,
            model_config=model_config,
            member_key=member_key,
            target_columns=target_columns,
            leakage_config=leakage_config,
        )

        if not model_feature_columns:
            raise ValueError(
                f"No eligible training feature columns found for model: {model_key}"
            )

        all_safe_feature_columns.extend(model_feature_columns)

        runtime.logger.info(
            "Model-specific training feature count | Model: %s | Features: %s",
            model_key,
            len(model_feature_columns),
        )

        training_result = train_model_candidates(
            dataframe=modeling_frame,
            feature_columns=model_feature_columns,
            target_column=target_column,
            model_key=model_key,
            model_name=model_name,
            modeling_defaults=modeling_defaults,
            training_config=training_config,
            run_id=runtime.run_id,
            event_timestamp_utc=runtime.event_timestamp_utc,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            parallelism_config=runtime.parallelism_config,
            logger=runtime.logger,
        )

        runtime.candidate_leaderboard_frames.append(training_result.metrics_dataframe)
        runtime.champion_summary_frames.append(
            training_result.champion_summary_dataframe
        )

        if getattr(training_result, "cross_validation_fold_metrics_dataframe", None) is not None:
            runtime.cross_validation_fold_frames.append(
                training_result.cross_validation_fold_metrics_dataframe
            )

        if getattr(training_result, "hyperparameter_search_results_dataframe", None) is not None:
            runtime.hyperparameter_search_frames.append(
                training_result.hyperparameter_search_results_dataframe
            )

        if getattr(training_result, "cross_validation_summary_dataframe", None) is not None:
            runtime.cross_validation_summary_frames.append(
                training_result.cross_validation_summary_dataframe
            )

        runtime.logger.info(
            "Champion selected | Model: %s | Algorithm: %s | Metrics: %s",
            model_key,
            training_result.champion_algorithm_key,
            training_result.champion_metrics,
        )

        scoring_start_time = time.perf_counter()
        runtime.logger.info("START Population Scoring | Model: %s", model_key)

        scoring_result = score_population(
            dataframe=modeling_frame,
            feature_columns=model_feature_columns,
            member_key=member_key,
            model_key=model_key,
            model_name=model_name,
            pipeline=training_result.champion_pipeline,
            model_config=model_config,
            risk_tiers_config=risk_tiers_config,
            run_id=runtime.run_id,
        )

        runtime.scoring_results.append(scoring_result)
        runtime.logger.info(
            "COMPLETE Population Scoring | Model: %s | %.2f sec | Rows: %s",
            model_key,
            time.perf_counter() - scoring_start_time,
            scoring_result.row_count,
        )
        confusion_matrix_df = build_confusion_matrix_output(
            scoring_dataframe=scoring_result.scoring_dataframe,
            source_dataframe=modeling_frame,
            target_column=target_column,
            prediction_column=model_config.get("prediction_column"),
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
        )

        lift_gain_df = build_lift_gain_analysis(
            scoring_dataframe=scoring_result.scoring_dataframe,
            source_dataframe=modeling_frame,
            target_column=target_column,
            score_column=model_config.get("score_column"),
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
        )

        runtime.lift_gain_frames.append(lift_gain_df)

        runtime.confusion_matrix_frames.append(confusion_matrix_df)        

        feature_importance_df = build_feature_importance_output(
            pipeline=training_result.champion_pipeline,
            feature_columns=model_feature_columns,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
        )
        
        permutation_importance_df = build_permutation_importance(
            dataframe=modeling_frame,
            feature_columns=model_feature_columns,
            target_column=target_column,
            pipeline=training_result.champion_pipeline,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
            scoring_metric=runtime.config.get("explainability", {})
                .get("permutation_importance", {})
                .get("scoring_metric", "roc_auc"),
            n_repeats=runtime.config.get("explainability", {})
                .get("permutation_importance", {})
                .get("n_repeats", 5),
            max_rows=runtime.config.get("explainability", {})
                .get("permutation_importance", {})
                .get("sample_row_count", 300),
            random_state=runtime.config.get("explainability", {})
                .get("permutation_importance", {})
                .get("random_state", 42),
        )
        runtime.permutation_importance_frames.append(permutation_importance_df)

        shap_explainability_df = build_shap_explainability(
            dataframe=modeling_frame,
            feature_columns=model_feature_columns,
            pipeline=training_result.champion_pipeline,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
            max_rows=runtime.config.get("explainability", {})
                .get("shap", {})
                .get("sample_row_count", 300),
            background_rows=runtime.config.get("explainability", {})
                .get("shap", {})
                .get("background_row_count", 50),
            random_state=runtime.config.get("explainability", {})
                .get("shap", {})
                .get("random_state", 42),
        )
        runtime.shap_explainability_frames.append(shap_explainability_df)
        explainability_df = build_model_explainability_summary(
            feature_importance_dataframe=feature_importance_df,
            run_id=runtime.run_id,
            top_n_features=10,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
        )

        executive_explainability_df = build_model_explainability_executive_summary(
            explainability_summary_dataframe=explainability_df,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
        )

        runtime.model_explainability_frames.append(explainability_df)
        runtime.model_executive_explainability_frames.append(
            executive_explainability_df
        )

        output_config = get_model_output_config(runtime, model_key)

        model_path = normalize_path(
            runtime.project_root,
            output_config.get("model_path"),
        )

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
            selection_metric=selection_metric,
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
            message=f"Model completed. Champion={training_result.champion_algorithm_key}",
            row_count=scoring_result.row_count,
            output_path=str(scoring_path),
        )

        model_duration_seconds = time.perf_counter() - model_start_time
        add_step_timing_record(
            runtime=runtime,
            step_name="model_complete",
            model_key=model_key,
            duration_seconds=model_duration_seconds,
        )
        
        runtime.logger.info(
            "COMPLETE MODEL: %s | Total Time: %.2f sec",
            model_key,
            model_duration_seconds,
            )
    model_monitoring_summary_df = build_model_monitoring_summary(
        scoring_results=runtime.scoring_results,
        run_id=runtime.run_id,
        layer_name=runtime.layer_name,
        domain_name=runtime.domain_name,
    )
    runtime.model_monitoring_summary_frames.append(model_monitoring_summary_df)
    baseline_feature_columns = sorted(set(all_safe_feature_columns))

    feature_baseline_df = build_feature_baseline_statistics(
        dataframe=modeling_frame,
        feature_columns=baseline_feature_columns,
        run_id=runtime.run_id,
        layer_name=runtime.layer_name,
        domain_name=runtime.domain_name,
    )

    feature_baseline_path = output_path_with_format(
        get_output_path(runtime, "outputs", "feature_baseline_statistics"),
        output_format,
    )

    write_dataset(feature_baseline_df, feature_baseline_path, output_format)

    add_dataset_record(
        runtime=runtime,
        dataset_name="feature_baseline_statistics",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=str(feature_baseline_path),
        row_count=len(feature_baseline_df),
        column_count=len(feature_baseline_df.columns),
        message="Feature baseline statistics written successfully.",
    )

    if bool(drift_config.get("enabled", False)) and bool(
        drift_config.get("build_baseline", True)
    ):
        model_drift_baseline_df = build_model_drift_baseline(
            dataframe=modeling_frame,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
        )

        runtime.model_drift_baseline_frames.append(model_drift_baseline_df)

        model_drift_baseline_path = output_path_with_format(
            get_output_path(runtime, "outputs", "model_drift_baseline"),
            output_format,
        )

        write_dataset(
            model_drift_baseline_df,
            model_drift_baseline_path,
            output_format,
        )

        add_dataset_record(
            runtime=runtime,
            dataset_name="model_drift_baseline",
            dataset_type="modeling_output",
            status=STATUS_SUCCESS,
            path=str(model_drift_baseline_path),
            row_count=len(model_drift_baseline_df),
            column_count=len(model_drift_baseline_df.columns),
            message="Model drift baseline written successfully.",
        )

        runtime.logger.info(
            "COMPLETE: Model Drift Baseline | Rows: %s | Path: %s",
            len(model_drift_baseline_df),
            model_drift_baseline_path,
        )

    else:
        runtime.logger.info(
            "SKIP: Model Drift Baseline | drift.enabled=false or build_baseline=false"
        )

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
###############################################################################
# Main Orchestration
###############################################################################

def build_modeling_layer(config_path: str = DEFAULT_CONFIG_PATH) -> BuildResult:
    """
    Build complete Modeling Framework.
    """

    runtime: Optional[ModelingRuntime] = None

    try:
        runtime = initialize_runtime(config_path)

        runtime.logger.info("Configuration path: %s", runtime.config_path)
        runtime.logger.info("Architectural check: Modeling consumes Feature Store only.")


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
            message="Timing record created before metadata write so it is included in the current output.",
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