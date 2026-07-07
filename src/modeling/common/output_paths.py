###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/output_paths.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Provides path resolution helpers for the Enterprise Modeling Framework.
#
# Responsibilities:
#     - Resolve project-relative paths.
#     - Ensure directories exist.
#     - Apply configured output file formats.
#     - Resolve named output paths from modeling configuration.
#     - Resolve model-specific artifact output configuration.
#
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.modeling.common.runtime import ModelingRuntime


def normalize_path(project_root: Path, raw_path: str | Path) -> Path:
    """
    Resolve a configured path relative to the project root.
    """

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_directory(path: Path) -> None:
    """
    Create a directory when it does not already exist.
    """

    path.mkdir(parents=True, exist_ok=True)


def output_path_with_format(path: Path, output_format: str) -> Path:
    """
    Ensure an output path has the configured file suffix.
    """

    suffix = f".{output_format}"

    if path.suffix:
        return path.with_suffix(suffix)

    return Path(str(path) + suffix)


def get_output_path(
    runtime: ModelingRuntime,
    output_group: str,
    output_name: str,
) -> Path:
    """
    Resolve a named output path from modeling configuration.
    """

    output_config = runtime.config.get("paths", {}).get(output_group, {})
    output_entry = output_config.get(output_name)

    if isinstance(output_entry, dict):
        raw_path = output_entry.get("path")
    else:
        raw_path = output_entry

    if not raw_path:
        if output_group == "outputs":
            raw_path = f"data/modeling/{output_name}"
        elif output_group == "metadata_outputs":
            raw_path = f"data/metadata/{output_name}"
        elif output_group == "audit_outputs":
            raw_path = f"data/audit/{output_name}"
        else:
            raw_path = f"data/{output_group}/{output_name}"

    return normalize_path(runtime.project_root, raw_path)


def get_model_output_config(
    runtime: ModelingRuntime,
    model_key: str,
) -> Dict[str, Any]:
    """
    Return artifact and scoring output paths for a configured model.
    """

    output_config = (
        runtime.config
        .get("paths", {})
        .get("model_outputs", {})
        .get(model_key, {})
    )

    if not output_config:
        raise ValueError(f"Missing model output config for model: {model_key}")

    required_keys = [
        "model_path",
        "metrics_path",
        "feature_importance_path",
        "scoring_path",
    ]

    missing = [key for key in required_keys if key not in output_config]

    if missing:
        raise ValueError(
            f"Model output config for {model_key} missing keys: {missing}"
        )

    return output_config


###############################################################################
# End of File
###############################################################################