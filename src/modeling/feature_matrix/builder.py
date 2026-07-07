###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/feature_matrix/builder.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Builds the unified Modeling feature matrix from configured Feature Store
#     input datasets.
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.modeling.common.audit import add_dataset_record
from src.modeling.common.constants import STATUS_SUCCESS
from src.modeling.common.runtime import ModelingRuntime


def clean_duplicate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated column names after joins.
    """

    return dataframe.loc[:, ~dataframe.columns.duplicated()].copy()


def prepare_dataset_for_join(
    dataframe: pd.DataFrame,
    dataset_name: str,
    member_key: str,
    exclude_columns: List[str],
    existing_columns: List[str],
) -> pd.DataFrame:
    """
    Prepare one Feature Store dataset for member-level joining.
    """

    if member_key not in dataframe.columns:
        raise ValueError(f"Dataset {dataset_name} missing member key: {member_key}")

    prepared = dataframe.drop_duplicates(subset=[member_key]).copy()

    drop_columns = [
        column for column in exclude_columns
        if column in prepared.columns
    ]

    if drop_columns:
        prepared = prepared.drop(columns=drop_columns)

    rename_map: Dict[str, str] = {}

    for column in prepared.columns:
        if column == member_key:
            continue

        if column in existing_columns:
            rename_map[column] = f"{dataset_name}__{column}"

    if rename_map:
        prepared = prepared.rename(columns=rename_map)

    return prepared


def build_feature_matrix(
    runtime: ModelingRuntime,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build unified Modeling feature matrix from configured Feature Store datasets.
    """

    config: Dict[str, Any] = runtime.config.get("feature_matrix", {})

    if not bool(config.get("enabled", True)):
        raise ValueError("feature_matrix.enabled must be true.")

    member_key = config.get("member_key", "member_id")
    base_dataset_name = config.get("base_dataset", "risk_features")
    join_datasets = config.get("join_datasets", [])
    exclude_columns = config.get("exclude_columns", [])

    if base_dataset_name not in datasets:
        raise ValueError(f"Feature matrix base dataset missing: {base_dataset_name}")

    matrix = prepare_dataset_for_join(
        dataframe=datasets[base_dataset_name],
        dataset_name=base_dataset_name,
        member_key=member_key,
        exclude_columns=exclude_columns,
        existing_columns=[],
    )

    runtime.logger.info(
        "START: Build feature matrix | Base: %s | Rows: %s | Columns: %s",
        base_dataset_name,
        len(matrix),
        len(matrix.columns),
    )

    for dataset_name in join_datasets:
        if dataset_name not in datasets:
            runtime.logger.warning("Skipping missing feature dataset: %s", dataset_name)
            continue

        join_df = prepare_dataset_for_join(
            dataframe=datasets[dataset_name],
            dataset_name=dataset_name,
            member_key=member_key,
            exclude_columns=exclude_columns,
            existing_columns=list(matrix.columns),
        )

        matrix = matrix.merge(join_df, on=member_key, how="left")
        matrix = clean_duplicate_columns(matrix)

        runtime.logger.info(
            "Joined %s | Rows: %s | Columns: %s",
            dataset_name,
            len(matrix),
            len(matrix.columns),
        )

    matrix["modeling_layer_run_id"] = runtime.run_id
    matrix["modeling_layer_built_at_utc"] = runtime.event_timestamp_utc

    add_dataset_record(
        runtime=runtime,
        dataset_name="modeling_feature_matrix",
        dataset_type="modeling_output",
        status=STATUS_SUCCESS,
        path=None,
        row_count=len(matrix),
        column_count=len(matrix.columns),
        message="Modeling feature matrix built successfully.",
    )

    runtime.logger.info(
        "COMPLETE: Build feature matrix | Rows: %s | Columns: %s",
        len(matrix),
        len(matrix.columns),
    )

    return matrix