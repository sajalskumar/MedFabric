###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/configuration.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Provides configuration loading and validation utilities for the
#     Enterprise Modeling Framework.
#
# Responsibilities:
#     - Load standard YAML files.
#     - Load optional YAML files.
#     - Load modular Modeling Framework configuration.
#     - Validate required configuration sections.
#
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.modeling.config_loader import load_modeling_config
from src.modeling.common.constants import (
    DEFAULT_OUTPUT_FORMAT,
    SUPPORTED_OUTPUT_FORMATS,
)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load one YAML configuration file.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    return config


def load_optional_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load optional YAML configuration.

    Used for config/pipeline.yaml. If the file is missing, Modeling continues
    with fallback runtime behavior.
    """

    if not config_path.exists():
        return {}

    return load_yaml_config(config_path)


def load_runtime_modeling_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the full modular Modeling Framework configuration.

    This delegates to src.modeling.config_loader.load_modeling_config, which
    merges config/modeling/modeling.yaml, top-level modular YAML files,
    algorithm YAML files, and model YAML files into one backward-compatible
    dictionary.
    """

    return load_modeling_config(config_path)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required Modeling Framework configuration sections.
    """

    required_sections = [
        "modeling",
        "logging",
        "paths",
        "join_keys",
        "feature_matrix",
        "modeling_defaults",
        "training",
        "models",
        "risk_tiers",
        "validation",
        "metadata",
        "audit",
    ]

    missing = [section for section in required_sections if section not in config]

    if missing:
        raise ValueError(f"Missing required Modeling config sections: {missing}")

    output_format = config.get("modeling", {}).get(
        "output_format",
        DEFAULT_OUTPUT_FORMAT,
    )

    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output_format '{output_format}'. "
            f"Supported: {sorted(SUPPORTED_OUTPUT_FORMATS)}"
        )

    paths = config.get("paths", {})

    required_path_sections = [
        "inputs",
        "outputs",
        "model_outputs",
        "metadata_outputs",
        "audit_outputs",
    ]

    missing_path_sections = [
        section for section in required_path_sections if section not in paths
    ]

    if missing_path_sections:
        raise ValueError(f"Missing required paths sections: {missing_path_sections}")


###############################################################################
# End of File
###############################################################################