###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/runtime.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Defines runtime dataclasses used by the Modeling Framework.
#
# Responsibilities:
#     - Store run-level execution context
#     - Carry shared configuration and logger objects
#     - Hold intermediate output frames for consolidated writing
#     - Return standardized build results
#
# Notes:
#     This module contains structure only.
#     It should not contain orchestration or business logic.
#
###############################################################################

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ModelingRuntime:
    """
    Runtime context for one Modeling Framework execution.
    """

    run_id: str
    project_root: Path
    config_path: Path
    pipeline_config_path: Path
    start_time_utc: datetime
    event_timestamp_utc: str
    config: Dict[str, Any]
    pipeline_config: Dict[str, Any]
    parallelism_config: Dict[str, Any]
    logger: logging.Logger
    capability_name: str
    layer_name: str
    domain_name: str

    audit_records: List[Dict[str, Any]] = field(default_factory=list)
    validation_records: List[Dict[str, Any]] = field(default_factory=list)
    dataset_records: List[Dict[str, Any]] = field(default_factory=list)
    model_registry_records: List[Dict[str, Any]] = field(default_factory=list)

    candidate_leaderboard_frames: List[pd.DataFrame] = field(default_factory=list)
    champion_summary_frames: List[pd.DataFrame] = field(default_factory=list)
    cross_validation_fold_frames: List[pd.DataFrame] = field(default_factory=list)
    cross_validation_summary_frames: List[pd.DataFrame] = field(default_factory=list)
    target_leakage_report_frames: List[pd.DataFrame] = field(default_factory=list)
    target_quality_report_frames: List[pd.DataFrame] = field(default_factory=list)
    hyperparameter_search_frames: List[pd.DataFrame] = field(default_factory=list)

    model_explainability_frames: List[pd.DataFrame] = field(default_factory=list)
    model_executive_explainability_frames: List[pd.DataFrame] = field(default_factory=list)
    model_drift_baseline_frames: List[pd.DataFrame] = field(default_factory=list)
    step_timing_records: List[Dict[str, Any]] = field(default_factory=list)

    confusion_matrix_frames: List[pd.DataFrame] = field(default_factory=list)
    lift_gain_frames: List[pd.DataFrame] = field(default_factory=list)
    permutation_importance_frames: List[pd.DataFrame] = field(default_factory=list)
    model_monitoring_summary_frames: List[pd.DataFrame] = field(default_factory=list)
    shap_explainability_frames: List[pd.DataFrame] = field(default_factory=list)
    member_level_explanations_frames: List[pd.DataFrame] = field(default_factory=list)
    population_stability_index_frames: List[pd.DataFrame] = field(default_factory=list)

    scoring_results: List[Any] = field(default_factory=list)


@dataclass
class BuildResult:
    """
    Standard result returned by the Modeling Framework builder.
    """

    name: str
    status: str
    message: str
    row_count: int = 0
    column_count: int = 0


###############################################################################
# End of File
###############################################################################