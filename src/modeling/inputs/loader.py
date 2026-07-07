###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/inputs/loader.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Loads configured Feature Store input datasets for the Modeling Framework.
#
###############################################################################

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.modeling.common.audit import add_audit_record, add_dataset_record
from src.modeling.common.constants import STATUS_FAILED, STATUS_SKIPPED, STATUS_SUCCESS
from src.modeling.common.io_utils import read_dataset
from src.modeling.common.output_paths import normalize_path
from src.modeling.common.runtime import ModelingRuntime


def load_input_datasets(runtime: ModelingRuntime) -> Dict[str, pd.DataFrame]:
    """
    Load configured Feature Store inputs.
    """

    inputs_config = runtime.config.get("paths", {}).get("inputs", {})
    datasets: Dict[str, pd.DataFrame] = {}

    runtime.logger.info("START: Load Modeling Feature Store inputs")

    for dataset_name, dataset_config in inputs_config.items():
        raw_path = dataset_config.get("path")
        file_format = dataset_config.get("format", "parquet")
        required = bool(dataset_config.get("required", True))

        if not raw_path:
            message = f"No path configured for dataset: {dataset_name}"

            if required:
                raise ValueError(message)

            add_audit_record(
                runtime=runtime,
                step_name=f"load_input:{dataset_name}",
                status=STATUS_SKIPPED,
                message=message,
            )
            continue

        dataset_path = normalize_path(runtime.project_root, raw_path)

        try:
            dataframe = read_dataset(dataset_path, file_format)
            datasets[dataset_name] = dataframe

            add_dataset_record(
                runtime=runtime,
                dataset_name=dataset_name,
                dataset_type="feature_store_input",
                status=STATUS_SUCCESS,
                path=str(dataset_path),
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                message="Feature Store input loaded successfully.",
            )

            runtime.logger.info(
                "Loaded %s | Rows: %s | Columns: %s | Path: %s",
                dataset_name,
                len(dataframe),
                len(dataframe.columns),
                dataset_path,
            )

        except Exception as exc:
            if required:
                add_audit_record(
                    runtime=runtime,
                    step_name=f"load_input:{dataset_name}",
                    status=STATUS_FAILED,
                    message=str(exc),
                    output_path=str(dataset_path),
                )
                raise

            add_audit_record(
                runtime=runtime,
                step_name=f"load_input:{dataset_name}",
                status=STATUS_SKIPPED,
                message=str(exc),
                output_path=str(dataset_path),
            )

            runtime.logger.warning(
                "Skipped optional dataset: %s | Reason: %s",
                dataset_name,
                exc,
            )

    runtime.logger.info("COMPLETE: Load inputs | Count: %s", len(datasets))

    return datasets