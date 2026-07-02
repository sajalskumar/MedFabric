###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/clean_medfabric.py
#
# Purpose:
#     Cleans generated MedFabric artifacts, logs, caches, and temporary files.
#
# Safety:
#     This script does NOT delete source code, configuration, documentation,
#     reference data, tests, notebooks, or scripts.
#
# Run Examples:
#     python scripts/clean_medfabric.py --dev
#     python scripts/clean_medfabric.py --data
#     python scripts/clean_medfabric.py --logs
#     python scripts/clean_medfabric.py --cache
#     python scripts/clean_medfabric.py --all
#
###############################################################################

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


GENERATED_DATA_DIRS = [
    "data/raw",
    "data/bronze",
    "data/silver",
    "data/gold",
    "data/features",
    "data/scoring",
    "data/semantic_layer",
    "data/analytics_platform",
    "data/insights",
    "data/pipeline",
]

LOG_DIRS = [
    "logs/pipeline",
    "logs/modules",
    "logs/audit",
    "logs/errors",
]

MODEL_DIRS = [
    "models",
]

PROTECTED_DIRS = {
    "src",
    "config",
    "reference",
    "docs",
    "tests",
    "scripts",
    "notebooks",
}


@dataclass
class CleanupSummary:
    mode: str
    directories_cleaned: int = 0
    files_removed: int = 0
    bytes_removed: int = 0


def get_project_root() -> Path:
    """
    Return the MedFabric project root.
    """

    return Path.cwd()


def format_bytes(size_bytes: int) -> str:
    """
    Format byte count into readable size.
    """

    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"

    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"

    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"

    return f"{size_bytes} bytes"


def path_size(path: Path) -> int:
    """
    Return total size of file or directory.
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


def ensure_directory(path: Path) -> None:
    """
    Ensure directory exists.
    """

    path.mkdir(parents=True, exist_ok=True)


def clean_directory_contents(path: Path, summary: CleanupSummary) -> None:
    """
    Remove contents of a directory but keep the directory itself.
    """

    if not path.exists():
        ensure_directory(path)
        return

    if not path.is_dir():
        return

    summary.bytes_removed += path_size(path)

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
            summary.directories_cleaned += 1
        else:
            item.unlink()
            summary.files_removed += 1

    ensure_directory(path)


def remove_file(path: Path, summary: CleanupSummary) -> None:
    """
    Remove one file.
    """

    if not path.exists() or not path.is_file():
        return

    summary.bytes_removed += path.stat().st_size
    path.unlink()
    summary.files_removed += 1


def clean_directories(
    project_root: Path,
    directories: Iterable[str],
    summary: CleanupSummary,
) -> None:
    """
    Clean configured directory contents.
    """

    for directory in directories:
        path = project_root / directory
        clean_directory_contents(path, summary)


def clean_python_cache(project_root: Path, summary: CleanupSummary) -> None:
    """
    Remove Python cache files and macOS temporary files.
    """

    for cache_dir in project_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            summary.bytes_removed += path_size(cache_dir)
            shutil.rmtree(cache_dir)
            summary.directories_cleaned += 1

    for pattern in ["*.pyc", ".DS_Store"]:
        for file_path in project_root.rglob(pattern):
            remove_file(file_path, summary)


def validate_safe_root(project_root: Path) -> None:
    """
    Prevent accidental execution outside the MedFabric project root.
    """

    required_paths = [
        project_root / "src",
        project_root / "config",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise RuntimeError(
            "This script must be run from the MedFabric project root. "
            f"Missing required paths: {missing}"
        )


def print_summary(summary: CleanupSummary, duration_seconds: float) -> None:
    """
    Print cleanup summary.
    """

    print("")
    print("=" * 80)
    print("MedFabric Cleanup Utility")
    print("=" * 80)
    print(f"Mode                 : {summary.mode}")
    print(f"Directories Cleaned  : {summary.directories_cleaned}")
    print(f"Files Removed        : {summary.files_removed}")
    print(f"Space Recovered      : {format_bytes(summary.bytes_removed)}")
    print(f"Duration             : {duration_seconds:.2f} seconds")
    print("=" * 80)
    print("Cleanup completed successfully.")
    print("=" * 80)


def run_cleanup(args: argparse.Namespace) -> CleanupSummary:
    """
    Execute cleanup based on selected mode.
    """

    project_root = get_project_root()
    validate_safe_root(project_root)

    if args.dev:
        summary = CleanupSummary(mode="Development Cleanup")
        clean_directories(project_root, GENERATED_DATA_DIRS, summary)
        clean_directories(project_root, LOG_DIRS, summary)
        clean_python_cache(project_root, summary)
        return summary

    if args.data:
        summary = CleanupSummary(mode="Generated Data Cleanup")
        clean_directories(project_root, GENERATED_DATA_DIRS, summary)
        return summary

    if args.logs:
        summary = CleanupSummary(mode="Log Cleanup")
        clean_directories(project_root, LOG_DIRS, summary)
        return summary

    if args.cache:
        summary = CleanupSummary(mode="Python Cache Cleanup")
        clean_python_cache(project_root, summary)
        return summary

    if args.all:
        summary = CleanupSummary(mode="Full Platform Reset")
        clean_directories(project_root, GENERATED_DATA_DIRS, summary)
        clean_directories(project_root, LOG_DIRS, summary)
        clean_directories(project_root, MODEL_DIRS, summary)
        clean_python_cache(project_root, summary)
        return summary

    raise RuntimeError("No cleanup mode selected.")


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Clean generated MedFabric artifacts safely."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "--dev",
        action="store_true",
        help="Clean generated data, logs, and Python cache.",
    )

    mode_group.add_argument(
        "--data",
        action="store_true",
        help="Clean generated data only.",
    )

    mode_group.add_argument(
        "--logs",
        action="store_true",
        help="Clean logs only.",
    )

    mode_group.add_argument(
        "--cache",
        action="store_true",
        help="Clean Python cache and temporary files only.",
    )

    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Full reset: generated data, logs, models, and cache.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_parser()
    args = parser.parse_args()

    start_time = time.time()
    summary = run_cleanup(args)
    duration_seconds = time.time() - start_time

    print_summary(summary, duration_seconds)


if __name__ == "__main__":
    main()