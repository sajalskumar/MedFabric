###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/audit.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Provides shared audit, validation, and dataset inventory record helpers
#     for the Enterprise Modeling Framework.
#
# Responsibilities:
#     - Add step-level audit records.
#     - Add validation result records.
#     - Add dataset inventory records.
#
###############################################################################

from __future__ import annotations

from typing import Optional

from src.modeling.common.runtime import ModelingRuntime


def add_audit_record(
    runtime: ModelingRuntime,
    step_name: str,
    status: str,
    message: str,
    row_count: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Add a Modeling Framework audit record.
    """

    runtime.audit_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "capability_name": runtime.capability_name,
            "domain_name": runtime.domain_name,
            "step_name": step_name,
            "status": status,
            "message": message,
            "row_count": row_count,
            "output_path": output_path,
            "event_timestamp_utc": runtime.event_timestamp_utc,
        }
    )


def add_validation_record(
    runtime: ModelingRuntime,
    dataset_name: str,
    rule_name: str,
    status: str,
    message: str,
    failed_count: int = 0,
) -> None:
    """
    Add a Modeling Framework validation result record.
    """

    runtime.validation_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "capability_name": runtime.capability_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "rule_name": rule_name,
            "status": status,
            "message": message,
            "failed_count": failed_count,
            "event_timestamp_utc": runtime.event_timestamp_utc,
        }
    )


def add_dataset_record(
    runtime: ModelingRuntime,
    dataset_name: str,
    dataset_type: str,
    status: str,
    path: Optional[str],
    row_count: int,
    column_count: int,
    message: str,
) -> None:
    """
    Add a Modeling Framework dataset inventory record.
    """

    runtime.dataset_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "capability_name": runtime.capability_name,
            "domain_name": runtime.domain_name,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "status": status,
            "path": path,
            "row_count": row_count,
            "column_count": column_count,
            "message": message,
            "event_timestamp_utc": runtime.event_timestamp_utc,
        }
    )


###############################################################################
# End of File
###############################################################################