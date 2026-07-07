###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/timing.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Provides shared timing utilities for the Enterprise Modeling Framework.
#
# Responsibilities:
#     - Return timezone-aware UTC timestamps.
#     - Add step-level runtime timing records.
#
###############################################################################

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.modeling.common.runtime import ModelingRuntime


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


def add_step_timing_record(
    runtime: ModelingRuntime,
    step_name: str,
    model_key: Optional[str],
    duration_seconds: float,
    status: str = "SUCCESS",
    message: Optional[str] = None,
) -> None:
    """
    Add a Modeling Framework step timing record.
    """

    runtime.step_timing_records.append(
        {
            "run_id": runtime.run_id,
            "layer_name": runtime.layer_name,
            "capability_name": runtime.capability_name,
            "domain_name": runtime.domain_name,
            "model_key": model_key,
            "step_name": step_name,
            "duration_seconds": float(duration_seconds),
            "status": status,
            "message": message,
            "event_timestamp_utc": runtime.event_timestamp_utc,
        }
    )


###############################################################################
# End of File
###############################################################################