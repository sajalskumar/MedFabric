###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/check_data_integrity.py
#
# Purpose:
#     Validates generated MedFabric data artifacts after pipeline execution.
#
# Business Context:
#     This utility provides a lightweight data integrity report across generated
#     platform outputs. It helps confirm that datasets exist, are readable,
#     contain rows, and do not have obvious key-quality problems.
#
# Safety:
#     Read-only script. It does not modify, delete, or regenerate any data.
#
# Run:
#     python scripts/check_data_integrity.py
#     python scripts/check_data_integrity.py --layer raw
#     python scripts/check_data_integrity.py --layer bronze
#     python scripts/check_data_integrity.py --fail-on-error
#
###############################################################################

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


###############################################################################
# Dataset Registry
###############################################################################

DATASET_LAYERS: Dict[str, str] = {
    "raw": "data/raw",
    "bronze": "data/bronze",
    "silver": "data/silver",
    "gold": "data/gold",
    "features": "data/features",
    "scoring": "data/scoring",
    "semantic_layer": "data/semantic_layer",
    "analytics_platform": "data/analytics_platform",
    "insights": "data/insights",
    "pipeline": "data/pipeline",
}


KEY_CANDIDATES = [
    "member_id",
    "provider_id",
    "facility_id",
    "claim_id",
    "claim_line_id",
    "encounter_id",
    "enrollment_id",
    "organization_id",
    "model_id",
    "run_id",
]


###############################################################################
# Data Classes
###############################################################################

@dataclass
class DatasetIntegrityResult:
    layer: str
    dataset_name: str
    path: str
    status: str
    row_count: int
    column_count: int
    key_column: Optional[str]
    null_key_count: int
    duplicate_key_count: int
    file_size_mb: float
    message: str


###############################################################################
# Helpers
###############################################################################

def get_project_root() -> Path:
    """
    Return the current MedFabric project root.
    """

    return Path.cwd()


def validate_project_root(project_root: Path) -> None:
    """
    Ensure the script is run from the MedFabric project root.
    """

    required_paths = [
        project_root / "src",
        project_root / "config",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise RuntimeError(
            "Run this script from the MedFabric project root. "
            f"Missing required paths: {missing}"
        )


def file_size_mb(path: Path) -> float:
    """
    Return file size in megabytes.
    """

    if not path.exists() or not path.is_file():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def find_key_column(dataframe: pd.DataFrame) -> Optional[str]:
    """
    Find the first likely key column in a dataframe.
    """

    for column in KEY_CANDIDATES:
        if column in dataframe.columns:
            return column

    sk_columns = [column for column in dataframe.columns if column.endswith("_sk")]
    if sk_columns:
        return sk_columns[0]

    id_columns = [column for column in dataframe.columns if column.endswith("_id")]
    if id_columns:
        return id_columns[0]

    return None


def read_parquet_safely(path: Path) -> pd.DataFrame:
    """
    Read a parquet file.
    """

    return pd.read_parquet(path)


def check_dataset(layer: str, path: Path, project_root: Path) -> DatasetIntegrityResult:
    """
    Check one parquet dataset.
    """

    relative_path = str(path.relative_to(project_root))
    dataset_name = path.stem

    try:
        dataframe = read_parquet_safely(path)

        row_count = len(dataframe)
        column_count = len(dataframe.columns)
        key_column = find_key_column(dataframe)

        null_key_count = 0
        duplicate_key_count = 0

        if key_column:
            null_key_count = int(dataframe[key_column].isna().sum())
            duplicate_key_count = int(dataframe.duplicated(subset=[key_column]).sum())

        status = "PASS"
        messages: List[str] = []

        if row_count == 0:
            status = "FAIL"
            messages.append("Dataset has zero rows.")

        if column_count == 0:
            status = "FAIL"
            messages.append("Dataset has zero columns.")

        if key_column and null_key_count > 0:
            status = "FAIL"
            messages.append(f"Key column '{key_column}' contains {null_key_count} nulls.")

        if key_column and duplicate_key_count > 0:
            status = "WARNING"
            messages.append(
                f"Key column '{key_column}' contains {duplicate_key_count} duplicates."
            )

        if not messages:
            messages.append("Dataset passed basic integrity checks.")

        return DatasetIntegrityResult(
            layer=layer,
            dataset_name=dataset_name,
            path=relative_path,
            status=status,
            row_count=row_count,
            column_count=column_count,
            key_column=key_column,
            null_key_count=null_key_count,
            duplicate_key_count=duplicate_key_count,
            file_size_mb=file_size_mb(path),
            message=" ".join(messages),
        )

    except Exception as error:
        return DatasetIntegrityResult(
            layer=layer,
            dataset_name=dataset_name,
            path=relative_path,
            status="FAIL",
            row_count=0,
            column_count=0,
            key_column=None,
            null_key_count=0,
            duplicate_key_count=0,
            file_size_mb=file_size_mb(path),
            message=f"Failed to read dataset: {error}",
        )


def discover_parquet_files(project_root: Path, layer_filter: Optional[str]) -> List[tuple[str, Path]]:
    """
    Discover parquet files to validate.
    """

    discovered: List[tuple[str, Path]] = []

    for layer, relative_dir in DATASET_LAYERS.items():
        if layer_filter and layer != layer_filter:
            continue

        directory = project_root / relative_dir

        if not directory.exists():
            continue

        for path in sorted(directory.rglob("*.parquet")):
            discovered.append((layer, path))

    return discovered


def write_report(
    project_root: Path,
    results: List[DatasetIntegrityResult],
) -> Path:
    """
    Write integrity results to data/pipeline/audit.
    """

    output_dir = project_root / "data/pipeline/audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "data_integrity_results.csv"

    dataframe = pd.DataFrame([result.__dict__ for result in results])
    dataframe.to_csv(output_path, index=False)

    return output_path


def print_report(results: List[DatasetIntegrityResult], duration_seconds: float) -> None:
    """
    Print readable integrity report.
    """

    pass_count = sum(1 for result in results if result.status == "PASS")
    warning_count = sum(1 for result in results if result.status == "WARNING")
    fail_count = sum(1 for result in results if result.status == "FAIL")

    print("")
    print("=" * 80)
    print("MedFabric Data Integrity Report")
    print("=" * 80)
    print(f"Datasets Checked : {len(results)}")
    print(f"Passed           : {pass_count}")
    print(f"Warnings         : {warning_count}")
    print(f"Failed           : {fail_count}")
    print(f"Duration         : {duration_seconds:.2f} seconds")
    print("=" * 80)

    for result in results:
        print(
            f"{result.status:<8} "
            f"{result.layer:<20} "
            f"{result.dataset_name:<45} "
            f"rows={result.row_count:<10} "
            f"cols={result.column_count:<5} "
            f"key={result.key_column or '-'}"
        )

        if result.status != "PASS":
            print(f"         Message: {result.message}")

    print("=" * 80)


###############################################################################
# CLI
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Check MedFabric generated data integrity."
    )

    parser.add_argument(
        "--layer",
        choices=DATASET_LAYERS.keys(),
        help="Optional layer filter.",
    )

    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero status if any dataset fails.",
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write CSV audit output.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_parser()
    args = parser.parse_args()

    project_root = get_project_root()
    validate_project_root(project_root)

    start_time = time.time()

    parquet_files = discover_parquet_files(
        project_root=project_root,
        layer_filter=args.layer,
    )

    results = [
        check_dataset(layer=layer, path=path, project_root=project_root)
        for layer, path in parquet_files
    ]

    duration_seconds = time.time() - start_time

    print_report(results, duration_seconds)

    if not args.no_write:
        output_path = write_report(project_root, results)
        print(f"Report written to: {output_path}")

    fail_count = sum(1 for result in results if result.status == "FAIL")

    if args.fail_on_error and fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()