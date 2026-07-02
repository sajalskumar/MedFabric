###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/targets/target_metadata.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds target metadata outputs for the MedFabric Modeling Target Framework.
#
# Run:
#     python -m src.modeling.targets.target_metadata
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TargetMetadataBuilder:
    run_id: str
    layer_name: str
    domain_name: str
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add_target_record(
        self,
        model_key: str,
        model_name: str,
        target_column: str,
        strategy: str,
        source_column: Optional[str],
        resolved_source_column: Optional[str],
        threshold_value: Optional[float],
        positive_count: int,
        negative_count: int,
        positive_rate: float,
        validation_status: str,
    ) -> None:
        self.records.append(
            {
                "run_id": self.run_id,
                "layer_name": self.layer_name,
                "domain_name": self.domain_name,
                "model_key": model_key,
                "model_name": model_name,
                "target_column": target_column,
                "target_strategy": strategy,
                "source_column": source_column,
                "resolved_source_column": resolved_source_column,
                "threshold_value": threshold_value,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": positive_rate,
                "validation_status": validation_status,
                "event_timestamp_utc": utc_now_iso(),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def build_target_metadata_from_summary(target_summary: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "run_id",
        "layer_name",
        "domain_name",
        "model_key",
        "model_name",
        "target_column",
        "target_strategy",
        "source_column",
        "resolved_source_column",
        "threshold_value",
        "positive_count",
        "negative_count",
        "positive_rate",
        "status",
        "event_timestamp_utc",
    ]

    if target_summary.empty:
        return pd.DataFrame(columns=required_columns)

    metadata = target_summary.copy()

    for column in required_columns:
        if column not in metadata.columns:
            metadata[column] = None

    return metadata[required_columns]


def main() -> None:
    builder = TargetMetadataBuilder(
        run_id="TEST_RUN",
        layer_name="Layer 2D - Enterprise Modeling Framework",
        domain_name="Modeling",
    )

    builder.add_target_record(
        model_key="high_cost",
        model_name="High Cost Model",
        target_column="high_cost_target",
        strategy="quantile",
        source_column="total_paid_amount",
        resolved_source_column="total_paid_amount",
        threshold_value=420.0,
        positive_count=1,
        negative_count=4,
        positive_rate=0.20,
        validation_status="SUCCESS",
    )

    df = builder.to_dataframe()

    print("Target metadata module validation successful.")
    print(df)


if __name__ == "__main__":
    main()