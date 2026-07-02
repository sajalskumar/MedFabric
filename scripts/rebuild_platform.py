###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/rebuild_platform.py
#
# Purpose:
#     Cleans selected generated MedFabric artifacts and then executes the
#     Enterprise Pipeline.
#
# Business Context:
#     This utility provides a single operational command for rebuilding the
#     MedFabric platform from a controlled cleanup state.
#
# Run:
#     python scripts/rebuild_platform.py --dev
#     python scripts/rebuild_platform.py --data
#     python scripts/rebuild_platform.py --all
#     python scripts/rebuild_platform.py --no-clean
#
###############################################################################

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def validate_project_root() -> None:
    """
    Ensure this script is run from the MedFabric project root.
    """

    required = [
        Path("src"),
        Path("config"),
        Path("scripts"),
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            "Run this script from the MedFabric project root. "
            f"Missing required paths: {missing}"
        )


def run_command(command: List[str], label: str) -> None:
    """
    Run one shell command and fail immediately if it fails.
    """

    print("")
    print("=" * 80)
    print(label)
    print("=" * 80)
    print(" ".join(command))
    print("=" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Clean and rebuild the MedFabric platform."
    )

    cleanup_group = parser.add_mutually_exclusive_group()

    cleanup_group.add_argument(
        "--dev",
        action="store_true",
        help="Development cleanup before rebuild.",
    )

    cleanup_group.add_argument(
        "--data",
        action="store_true",
        help="Clean generated data before rebuild.",
    )

    cleanup_group.add_argument(
        "--all",
        action="store_true",
        help="Full reset before rebuild.",
    )

    cleanup_group.add_argument(
        "--no-clean",
        action="store_true",
        help="Run pipeline without cleanup.",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-flight platform validation.",
    )

    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip post-run data integrity check.",
    )

    return parser


def determine_cleanup_mode(args: argparse.Namespace) -> str | None:
    """
    Determine cleanup mode from command-line arguments.
    """

    if args.no_clean:
        return None

    if args.data:
        return "--data"

    if args.all:
        return "--all"

    return "--dev"


def main() -> None:
    """
    Command-line entry point.
    """

    validate_project_root()

    parser = build_parser()
    args = parser.parse_args()

    start_time = time.time()

    try:
        cleanup_mode = determine_cleanup_mode(args)

        if cleanup_mode is not None:
            run_command(
                [sys.executable, "scripts/clean_medfabric.py", cleanup_mode],
                "Cleanup MedFabric Artifacts",
            )

        if not args.skip_validation:
            run_command(
                [sys.executable, "scripts/validate_platform.py"],
                "Validate MedFabric Platform",
            )

        run_command(
            [sys.executable, "-m", "src.pipeline.build_medfabric_platform"],
            "Run MedFabric Enterprise Pipeline",
        )

        if not args.skip_integrity_check:
            run_command(
                [sys.executable, "scripts/check_data_integrity.py"],
                "Check MedFabric Data Integrity",
            )

        duration = time.time() - start_time

        print("")
        print("=" * 80)
        print("MedFabric rebuild completed successfully.")
        print(f"Duration: {duration:.2f} seconds")
        print("=" * 80)

    except Exception as error:
        duration = time.time() - start_time

        print("")
        print("=" * 80)
        print("MedFabric rebuild failed.")
        print("=" * 80)
        print(f"Error   : {error}")
        print(f"Duration: {duration:.2f} seconds")
        print("=" * 80)

        sys.exit(1)


if __name__ == "__main__":
    main()