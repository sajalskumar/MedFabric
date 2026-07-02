###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/run_layer.py
#
# Purpose:
#     Executes a single MedFabric platform layer.
#
# Business Context:
#     During development and stabilization it is inefficient to execute the
#     complete enterprise pipeline every time. This utility allows developers
#     to execute one layer independently while preserving the same execution
#     entry points used by the platform orchestrator.
#
# Usage:
#
#     python scripts/run_layer.py data_generation
#     python scripts/run_layer.py ingestion
#     python scripts/run_layer.py silver
#     python scripts/run_layer.py gold
#     python scripts/run_layer.py feature_store
#     python scripts/run_layer.py modeling
#     python scripts/run_layer.py semantic_layer
#     python scripts/run_layer.py analytics_platform
#     python scripts/run_layer.py insights
#
###############################################################################

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path


###############################################################################
# Supported Layers
###############################################################################

LAYER_MODULES = {
    "data_generation": "src.data_generation.build_raw_data",
    "ingestion": "src.ingestion.build_ingestion_layer",
    "silver": "src.silver.build_silver_layer",
    "gold": "src.gold.build_gold_layer",
    "feature_store": "src.feature_store.build_feature_store",
    "modeling": "src.modeling.build_modeling_layer",
    "semantic_layer": "src.semantic_layer.build_semantic_layer",
    "analytics_platform": "src.analytics_platform.build_analytics_platform",
    "insights": "src.insights.build_insights_platform",
}


###############################################################################
# Helpers
###############################################################################

def validate_project_root() -> None:
    """
    Ensure this utility is executed from the MedFabric project root.
    """

    required = [
        Path("src"),
        Path("config"),
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            "This script must be executed from the MedFabric project root.\n"
            f"Missing: {', '.join(missing)}"
        )


def execute_layer(layer_name: str) -> None:
    """
    Import the requested module and execute its main() function.
    """

    module_name = LAYER_MODULES[layer_name]

    print("=" * 80)
    print("MedFabric Layer Runner")
    print("=" * 80)
    print(f"Layer : {layer_name}")
    print(f"Module: {module_name}")
    print("=" * 80)

    module = importlib.import_module(module_name)

    if not hasattr(module, "main"):
        raise RuntimeError(
            f"{module_name} does not expose a main() entry point."
        )

    start = time.time()

    module.main()

    duration = time.time() - start

    print("")
    print("=" * 80)
    print("Layer completed successfully.")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 80)


###############################################################################
# Main
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Run one MedFabric platform layer."
    )

    parser.add_argument(
        "layer",
        choices=LAYER_MODULES.keys(),
        help="Layer to execute.",
    )

    return parser


def main() -> None:
    """
    Script entry point.
    """

    validate_project_root()

    parser = build_parser()
    args = parser.parse_args()

    try:
        execute_layer(args.layer)

    except KeyboardInterrupt:
        print("\nExecution cancelled.")
        sys.exit(1)

    except Exception as error:
        print("")
        print("=" * 80)
        print("Layer execution failed.")
        print("=" * 80)
        print(error)
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()