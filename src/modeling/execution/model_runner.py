###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/execution/model_runner.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Executes enabled Modeling objectives using the enterprise modeling runtime.
#
# Run:
#     python -m src.modeling.build_modeling_layer
#
###############################################################################

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

from src.modeling.common.audit import (
    add_audit_record,
    add_dataset_record,
    add_validation_record,
)
from src.modeling.common.constants import STATUS_SUCCESS
from src.modeling.common.io_utils import save_pickle_object, write_dataset
from src.modeling.common.output_paths import (
    get_model_output_config,
    get_output_path,
    normalize_path,
    output_path_with_format,
)
from src.modeling.common.runtime import ModelingRuntime
from src.modeling.common.timing import add_step_timing_record
from src.modeling.evaluation.build_feature_baseline_statistics import (
    build_feature_baseline_statistics,
)
from src.modeling.evaluation.build_lift_gain_analysis import build_lift_gain_analysis
from src.modeling.evaluation.build_model_drift_baseline import (
    build_model_drift_baseline,
)
from src.modeling.evaluation.build_model_explainability_summary import (
    build_model_explainability_executive_summary,
    build_model_explainability_summary,
)
from src.modeling.evaluation.build_model_monitoring_summary import (
    build_model_monitoring_summary,
)
from src.modeling.evaluation.build_permutation_importance import (
    build_permutation_importance,
)
from src.modeling.evaluation.build_shap_explainability import build_shap_explainability
from src.modeling.evaluation.build_target_quality_report import (
    build_target_quality_report,
)
from src.modeling.evaluation.confusion_matrix import build_confusion_matrix_output
from src.modeling.evaluation.feature_importance import build_feature_importance_output
from src.modeling.outputs.writer import get_output_format
from src.modeling.registry.model_registry import build_model_registry_record
from src.modeling.scoring.scorer import score_population
from src.modeling.targets.leakage_detection import (
    build_target_leakage_report_for_models,
    get_model_specific_safe_feature_columns,
    get_target_output_column,
)
from src.modeling.targets.target_builder import build_targets_for_enabled_models
from src.modeling.training.trainer import train_model_candidates
from src.modeling.evaluation.build_member_level_explanations import (
    build_member_level_explanations,
)

###############################################################################
# Configuration Helpers
###############################################################################

def get_selection_metric(
    modeling_defaults: Dict[str, Any],
    training_config: Dict[str, Any],
) -> str:
    """
    Resolve champion model selection metric.
    """

    return (
        training_config
        .get("metrics", {})
        .get(
            "primary_metric",
            modeling_defaults.get("selection_metric", "roc_auc"),
        )
    )


def get_algorithms_config(runtime: ModelingRuntime) -> Dict[str, Any]:
    """
    Validate and return enabled algorithm configuration.
    """

    training_config = runtime.config.get("training", {})
    algorithms_config = training_config.get("algorithms")

    if not isinstance(algorithms_config, dict):
        raise ValueError("training.algorithms must be configured in Modeling YAML.")

    enabled_algorithms = [
        key
        for key, value in algorithms_config.items()
        if bool(value.get("enabled", True))
    ]

    if not enabled_algorithms:
        raise ValueError("No enabled algorithms found in training.algorithms.")

    runtime.logger.info("Enabled algorithms: %s", enabled_algorithms)

    return algorithms_config


###############################################################################
# Modeling Execution
###############################################################################

def run_models(
    runtime: ModelingRuntime,
    feature_matrix: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Execute model-level workflows for all enabled Modeling objectives.
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

    get_algorithms_config(runtime)

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

    target_quality_targets: List[Dict[str, Any]] = []

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

    add_dataset_record(
        runtime=runtime,
        dataset_name="target_leakage_report",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(target_leakage_report_df),
        column_count=len(target_leakage_report_df.columns),
        message="Target leakage report built successfully.",
    )

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

        runtime.confusion_matrix_frames.append(confusion_matrix_df)

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

        permutation_importance_config = (
            runtime.config.get("explainability", {}).get("permutation_importance", {})
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
            scoring_metric=permutation_importance_config.get("scoring_metric", "roc_auc"),
            n_repeats=permutation_importance_config.get("n_repeats", 5),
            max_rows=permutation_importance_config.get("sample_row_count", 300),
            random_state=permutation_importance_config.get("random_state", 42),
        )

        runtime.permutation_importance_frames.append(permutation_importance_df)

        shap_config = runtime.config.get("explainability", {}).get("shap", {})

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
            max_rows=shap_config.get("sample_row_count", 300),
            background_rows=shap_config.get("background_row_count", 50),
            random_state=shap_config.get("random_state", 42),
        )

        runtime.shap_explainability_frames.append(shap_explainability_df)

        member_level_config = (
            runtime.config
            .get("explainability", {})
            .get("member_level_explanations", {})
        )

        member_level_explanations_df = build_member_level_explanations(
            dataframe=modeling_frame,
            feature_columns=model_feature_columns,
            member_key=member_key,
            pipeline=training_result.champion_pipeline,
            run_id=runtime.run_id,
            layer_name=runtime.layer_name,
            domain_name=runtime.domain_name,
            model_key=model_key,
            model_name=model_name,
            algorithm_key=training_result.champion_algorithm_key,
            algorithm_name=training_result.champion_algorithm_name,
            max_members=member_level_config.get("max_members", 100),
            top_n_features=member_level_config.get("top_n_features", 5),
            background_row_count=member_level_config.get("background_row_count", 50),
            random_state=member_level_config.get("random_state", 42),
        )

        runtime.member_level_explanations_frames.append(member_level_explanations_df)

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
            message=(
                "Model completed. "
                f"Champion={training_result.champion_algorithm_key}"
            ),
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

    if not baseline_feature_columns:
        runtime.logger.warning(
            "SKIP: Feature baseline statistics because no enabled models produced safe features."
        )
        return scoring_outputs

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
        drift_columns = [member_key] + baseline_feature_columns
        drift_columns = [
            column for column in drift_columns if column in modeling_frame.columns
        ]

        model_drift_baseline_df = build_model_drift_baseline(
            dataframe=modeling_frame[drift_columns].copy(),
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