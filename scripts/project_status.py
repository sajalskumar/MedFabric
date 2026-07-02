###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/project_status.py
#
# Purpose:
#     Reports current MedFabric project health, generated artifacts, dataset
#     counts, logs, models, metadata, and storage usage.
#
# Run:
#     python scripts/project_status.py
#     python scripts/project_status.py --data
#     python scripts/project_status.py --logs
#     python scripts/project_status.py --models
#     python scripts/project_status.py --storage
#
###############################################################################

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DATA_LAYER_DIRS = {
    "Raw": "data/raw",
    "Bronze": "data/bronze",
    "Silver": "data/silver",
    "Gold": "data/gold",
    "Features": "data/features",
    "Scoring": "data/scoring",
    "Semantic Layer": "data/semantic_layer",
    "Analytics Platform": "data/analytics_platform",
    "Insights": "data/insights",
    "Pipeline": "data/pipeline",
}

LOG_DIRS = {
    "Pipeline Logs": "logs/pipeline",
    "Module Logs": "logs/modules",
    "Audit Logs": "logs/audit",
    "Error Logs": "logs/errors",
}

CONFIG_DIRS = {
    "Data Generation": "config/data_generation",
    "Ingestion": "config/ingestion",
    "Silver": "config/silver",
    "Gold": "config/gold",
    "Feature Store": "config/feature_store",
    "Modeling": "config/modeling",
    "Semantic Layer": "config/semantic_layer",
    "Analytics Platform": "config/analytics_platform",
    "Insights": "config/insights",
    "Pipeline": "config/pipeline",
}

SRC_DIRS = {
    "Common": "src/common",
    "Data Generation": "src/data_generation",
    "Ingestion": "src/ingestion",
    "Silver": "src/silver",
    "Gold": "src/gold",
    "Feature Store": "src/feature_store",
    "Modeling": "src/modeling",
    "Semantic Layer": "src/semantic_layer",
    "Analytics Platform": "src/analytics_platform",
    "Insights": "src/insights",
    "Pipeline": "src/pipeline",
}


@dataclass
class DirectoryStatus:
    name: str
    path: Path
    exists: bool
    file_count: int
    parquet_count: int
    size_bytes: int


def get_project_root() -> Path:
    """
    Return the current project root.
    """

    return Path.cwd()


def validate_project_root(project_root: Path) -> None:
    """
    Ensure the script is being run from the MedFabric project root.
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


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes into readable units.
    """

    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"

    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"

    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"

    return f"{size_bytes} bytes"


def directory_size(path: Path) -> int:
    """
    Calculate directory size.
    """

    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    total = 0

    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass

    return total


def count_files(path: Path, suffix: Optional[str] = None) -> int:
    """
    Count files in a directory.
    """

    if not path.exists():
        return 0

    if path.is_file():
        if suffix is None:
            return 1
        return 1 if path.name.endswith(suffix) else 0

    count = 0

    for item in path.rglob("*"):
        if not item.is_file():
            continue

        if suffix is None or item.name.endswith(suffix):
            count += 1

    return count


def build_directory_status(
    project_root: Path,
    name: str,
    relative_path: str,
) -> DirectoryStatus:
    """
    Build status for one directory.
    """

    path = project_root / relative_path

    return DirectoryStatus(
        name=name,
        path=path,
        exists=path.exists(),
        file_count=count_files(path),
        parquet_count=count_files(path, ".parquet"),
        size_bytes=directory_size(path),
    )


def status_symbol(condition: bool) -> str:
    """
    Return status symbol.
    """

    return "✓" if condition else "✗"


def print_header(title: str) -> None:
    """
    Print report header.
    """

    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_section(title: str) -> None:
    """
    Print section header.
    """

    print("")
    print(title)
    print("-" * 80)


def print_project_summary(project_root: Path) -> None:
    """
    Print high-level project summary.
    """

    print_section("Project")

    print(f"{'Project Root':<30}: {project_root}")
    print(f"{'Source Exists':<30}: {status_symbol((project_root / 'src').exists())}")
    print(f"{'Config Exists':<30}: {status_symbol((project_root / 'config').exists())}")
    print(f"{'Docs Exists':<30}: {status_symbol((project_root / 'docs').exists())}")
    print(f"{'Scripts Exists':<30}: {status_symbol((project_root / 'scripts').exists())}")
    print(f"{'Reference Exists':<30}: {status_symbol((project_root / 'reference').exists())}")
    print(f"{'Tests Exists':<30}: {status_symbol((project_root / 'tests').exists())}")


def print_config_status(project_root: Path) -> None:
    """
    Print configuration folder status.
    """

    print_section("Configuration")

    for name, relative_path in CONFIG_DIRS.items():
        path = project_root / relative_path
        yaml_count = count_files(path, ".yaml")
        print(
            f"{name:<30}: {status_symbol(path.exists())} "
            f"{relative_path:<40} YAML files: {yaml_count}"
        )


def print_source_status(project_root: Path) -> None:
    """
    Print source package status.
    """

    print_section("Source Packages")

    for name, relative_path in SRC_DIRS.items():
        path = project_root / relative_path
        py_count = count_files(path, ".py")
        print(
            f"{name:<30}: {status_symbol(path.exists())} "
            f"{relative_path:<40} Python files: {py_count}"
        )


def print_data_status(project_root: Path) -> None:
    """
    Print generated data status.
    """

    print_section("Generated Data")

    total_size = 0
    total_files = 0
    total_parquet = 0

    for name, relative_path in DATA_LAYER_DIRS.items():
        status = build_directory_status(project_root, name, relative_path)

        total_size += status.size_bytes
        total_files += status.file_count
        total_parquet += status.parquet_count

        print(
            f"{name:<30}: {status_symbol(status.exists)} "
            f"files={status.file_count:<6} "
            f"parquet={status.parquet_count:<6} "
            f"size={format_bytes(status.size_bytes)}"
        )

    print("-" * 80)
    print(
        f"{'Total Generated Data':<30}: "
        f"files={total_files:<6} parquet={total_parquet:<6} "
        f"size={format_bytes(total_size)}"
    )


def print_log_status(project_root: Path) -> None:
    """
    Print log status.
    """

    print_section("Logs")

    for name, relative_path in LOG_DIRS.items():
        status = build_directory_status(project_root, name, relative_path)

        print(
            f"{name:<30}: {status_symbol(status.exists)} "
            f"files={status.file_count:<6} "
            f"size={format_bytes(status.size_bytes)}"
        )


def print_model_status(project_root: Path) -> None:
    """
    Print model artifact status.
    """

    print_section("Models")

    model_path = project_root / "models"

    if not model_path.exists():
        print(f"{'Models Directory':<30}: ✗ models")
        return

    pkl_count = count_files(model_path, ".pkl")
    csv_count = count_files(model_path, ".csv")
    total_files = count_files(model_path)
    size = directory_size(model_path)

    print(f"{'Models Directory':<30}: ✓ models")
    print(f"{'Model Files (.pkl)':<30}: {pkl_count}")
    print(f"{'Metric/Importance CSVs':<30}: {csv_count}")
    print(f"{'Total Model Files':<30}: {total_files}")
    print(f"{'Model Size':<30}: {format_bytes(size)}")


def print_metadata_status(project_root: Path) -> None:
    """
    Print metadata and audit output status.
    """

    print_section("Metadata and Audit")

    metadata_paths = {
        "Pipeline Metadata": "data/pipeline/metadata",
        "Pipeline Audit": "data/pipeline/audit",
        "Analytics Metadata": "data/analytics_platform/metadata",
        "Analytics Audit": "data/analytics_platform/audit",
        "Insights Metadata": "data/insights/metadata",
        "Insights Audit": "data/insights/audit",
    }

    for name, relative_path in metadata_paths.items():
        status = build_directory_status(project_root, name, relative_path)

        print(
            f"{name:<30}: {status_symbol(status.exists)} "
            f"files={status.file_count:<6} "
            f"parquet={status.parquet_count:<6} "
            f"size={format_bytes(status.size_bytes)}"
        )


def print_storage_status(project_root: Path) -> None:
    """
    Print storage usage.
    """

    print_section("Storage")

    storage_paths = {
        "Project Total": ".",
        "Data": "data",
        "Models": "models",
        "Logs": "logs",
        "Reference": "reference",
        "Source": "src",
        "Config": "config",
        "Docs": "docs",
    }

    for name, relative_path in storage_paths.items():
        path = project_root / relative_path
        print(f"{name:<30}: {format_bytes(directory_size(path))}")


def calculate_platform_ready(project_root: Path) -> bool:
    """
    Determine whether major platform outputs appear ready.
    """

    required_paths = [
        project_root / "data/raw",
        project_root / "data/bronze",
        project_root / "data/silver",
        project_root / "data/gold",
        project_root / "data/features",
        project_root / "data/analytics_platform",
        project_root / "data/insights",
        project_root / "data/pipeline",
    ]

    return all(path.exists() and count_files(path, ".parquet") > 0 for path in required_paths)


def print_overall_status(project_root: Path) -> None:
    """
    Print overall platform readiness.
    """

    print_section("Overall Status")

    platform_ready = calculate_platform_ready(project_root)

    print(f"{'Platform Ready':<30}: {'YES' if platform_ready else 'NO'}")

    if not platform_ready:
        print("")
        print("Note:")
        print("  Platform Ready = YES only when major generated output folders exist")
        print("  and contain parquet outputs.")


def print_full_status(project_root: Path) -> None:
    """
    Print full project status.
    """

    print_header("MedFabric Project Status")
    print_project_summary(project_root)
    print_config_status(project_root)
    print_source_status(project_root)
    print_data_status(project_root)
    print_model_status(project_root)
    print_log_status(project_root)
    print_metadata_status(project_root)
    print_storage_status(project_root)
    print_overall_status(project_root)
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Show MedFabric project status."
    )

    parser.add_argument(
        "--data",
        action="store_true",
        help="Show generated data status only.",
    )

    parser.add_argument(
        "--logs",
        action="store_true",
        help="Show log status only.",
    )

    parser.add_argument(
        "--models",
        action="store_true",
        help="Show model status only.",
    )

    parser.add_argument(
        "--storage",
        action="store_true",
        help="Show storage status only.",
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Show metadata and audit status only.",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration status only.",
    )

    parser.add_argument(
        "--source",
        action="store_true",
        help="Show source package status only.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """

    project_root = get_project_root()
    validate_project_root(project_root)

    parser = build_parser()
    args = parser.parse_args()

    if args.data:
        print_header("MedFabric Generated Data Status")
        print_data_status(project_root)
        return

    if args.logs:
        print_header("MedFabric Log Status")
        print_log_status(project_root)
        return

    if args.models:
        print_header("MedFabric Model Status")
        print_model_status(project_root)
        return

    if args.storage:
        print_header("MedFabric Storage Status")
        print_storage_status(project_root)
        return

    if args.metadata:
        print_header("MedFabric Metadata and Audit Status")
        print_metadata_status(project_root)
        return

    if args.config:
        print_header("MedFabric Configuration Status")
        print_config_status(project_root)
        return

    if args.source:
        print_header("MedFabric Source Status")
        print_source_status(project_root)
        return

    print_full_status(project_root)


if __name__ == "__main__":
    main()