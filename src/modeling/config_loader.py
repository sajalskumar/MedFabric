###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/config_loader.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Loads and merges the modular Modeling Framework YAML configuration files
#     into one backward-compatible dictionary.
#
# Run:
#     python -m src.modeling.config_loader
#
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_MODELING_CONFIG_DIR = "config/modeling"
DEFAULT_MAIN_CONFIG_FILE = "modeling.yaml"

TOP_LEVEL_CONFIG_FILES = [
    "feature_matrix.yaml",
    "training.yaml",
    "explainability.yaml",
    "drift.yaml",
    "scoring.yaml",
    "risk_tiers.yaml",
    "leakage_detection.yaml",
]


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load one YAML file as a dictionary.
    """

    if not path.exists():
        raise FileNotFoundError(f"Modeling configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML file must contain a mapping/dictionary: {path}")

    return config


def deep_merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge override into base.
    """

    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_yaml_files_from_directory(directory: Path) -> Dict[str, Any]:
    """
    Load and merge all YAML files from one directory.
    """

    merged: Dict[str, Any] = {}

    if not directory.exists():
        return merged

    for path in sorted(directory.glob("*.yaml")):
        file_config = load_yaml_file(path)
        merged = deep_merge_dicts(merged, file_config)

    return merged


def load_modeling_config(
    config_path: str | Path = DEFAULT_MODELING_CONFIG_DIR,
) -> Dict[str, Any]:
    """
    Load complete modular Modeling Framework configuration.

    The returned dictionary is intentionally shaped like the previous
    monolithic config/modeling/modeling.yaml so existing Modeling Framework
    code can keep using:

        config["training"]["algorithms"]
        config["models"]
        config["feature_matrix"]
        config["explainability"]
        config["drift"]
        config["scoring"]
        config["risk_tiers"]
        config["leakage_detection"]

    without needing major orchestration changes.
    """

    config_dir = Path(config_path)

    if config_dir.is_file():
        config_dir = config_dir.parent

    main_config_path = config_dir / DEFAULT_MAIN_CONFIG_FILE

    merged_config = load_yaml_file(main_config_path)

    for file_name in TOP_LEVEL_CONFIG_FILES:
        file_path = config_dir / file_name

        if file_path.exists():
            merged_config = deep_merge_dicts(
                merged_config,
                load_yaml_file(file_path),
            )

    algorithms_config = load_yaml_files_from_directory(config_dir / "algorithms")

    if algorithms_config:
        merged_config.setdefault("training", {})
        merged_config["training"].setdefault("algorithms", {})
        merged_config["training"]["algorithms"] = deep_merge_dicts(
            merged_config["training"]["algorithms"],
            algorithms_config,
        )

    models_config = load_yaml_files_from_directory(config_dir / "models")

    if models_config:
        merged_config.setdefault("models", {})
        merged_config["models"] = deep_merge_dicts(
            merged_config["models"],
            models_config,
        )

    return merged_config


def main() -> None:
    """
    Lightweight validation entry point.
    """

    config = load_modeling_config()

    print("Modeling modular configuration loaded successfully.")
    print(f"Top-level sections: {sorted(config.keys())}")

    algorithms = config.get("training", {}).get("algorithms", {})
    models = config.get("models", {})

    print(f"Algorithms loaded: {sorted(algorithms.keys())}")
    print(f"Models loaded: {sorted(models.keys())}")


if __name__ == "__main__":
    main()