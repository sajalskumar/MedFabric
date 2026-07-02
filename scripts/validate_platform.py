###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/validate_platform.py
#
# Purpose:
#     Validates the MedFabric project structure, configuration files, source
#     modules, package initialization files, and pipeline configuration before
#     running the full enterprise pipeline.
#
# Business Context:
#     This utility is used as a pre-flight validation tool. It helps confirm
#     that the MedFabric repository is structurally ready before expensive
#     pipeline execution begins.
#
# Run:
#     python scripts/validate_platform.py
#
###############################################################################

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


REQUIRED_DIRECTORIES = [
    "src",
    "config",
    "docs",
    "scripts",
    "reference",
    "data",
    "models",
    "logs",
    "tests",
    "src/common",
    "src/data_generation",
    "src/ingestion",
    "src/silver",
    "src/gold",
    "src/feature_store",
    "src/modeling",
    "src/semantic_layer",
    "src/analytics_platform",
    "src/insights",
    "src/pipeline",
    "config/data_generation",
    "config/ingestion",
    "config/silver",
    "config/gold",
    "config/feature_store",
    "config/modeling",
    "config/semantic_layer",
    "config/analytics_platform",
    "config/insights",
    "config/pipeline",
]

REQUIRED_CONFIG_FILES = [
    "config/pipeline.yaml",
    "config/logging.yaml",
    "config/paths.yaml",
    "config/data_generation/generation.yaml",
    "config/ingestion/ingestion.yaml",
    "config/silver/silver.yaml",
    "config/gold/gold.yaml",
    "config/feature_store/feature_store.yaml",
    "config/modeling/modeling.yaml",
    "config/semantic_layer/semantic_layer.yaml",
    "config/analytics_platform/analytics_platform.yaml",
    "config/insights/insights.yaml",
    "config/pipeline/medfabric_platform.yaml",
]

REQUIRED_SOURCE_FILES = [
    "src/data_generation/build_raw_data.py",
    "src/ingestion/build_ingestion_layer.py",
    "src/silver/build_silver_layer.py",
    "src/gold/build_gold_layer.py",
    "src/feature_store/build_feature_store.py",
    "src/modeling/build_modeling_layer.py",
    "src/semantic_layer/build_semantic_layer.py",
    "src/analytics_platform/build_analytics_platform.py",
    "src/insights/build_insights_platform.py",
    "src/pipeline/build_medfabric_platform.py",
]

PIPELINE_MODULE_KEYS = [
    "src.data_generation.build_raw_data",
    "src.ingestion.build_ingestion_layer",
    "src.silver.build_silver_layer",
    "src.gold.build_gold_layer",
    "src.feature_store.build_feature_store",
    "src.modeling.build_modeling_layer",
    "src.semantic_layer.build_semantic_layer",
    "src.analytics_platform.build_analytics_platform",
    "src.insights.build_insights_platform",
]


@dataclass
class ValidationIssue:
    category: str
    path: str
    message: str
    severity: str


def get_project_root() -> Path:
    return Path.cwd()


def add_issue(
    issues: List[ValidationIssue],
    category: str,
    path: str,
    message: str,
    severity: str = "ERROR",
) -> None:
    issues.append(
        ValidationIssue(
            category=category,
            path=path,
            message=message,
            severity=severity,
        )
    )


def validate_project_root(project_root: Path, issues: List[ValidationIssue]) -> None:
    if not (project_root / "src").exists():
        add_issue(issues, "project_root", "src", "Missing src directory.")

    if not (project_root / "config").exists():
        add_issue(issues, "project_root", "config", "Missing config directory.")


def validate_directories(project_root: Path, issues: List[ValidationIssue]) -> None:
    for directory in REQUIRED_DIRECTORIES:
        path = project_root / directory
        if not path.exists() or not path.is_dir():
            add_issue(
                issues,
                "directory",
                directory,
                "Required directory is missing.",
            )


def validate_config_files(project_root: Path, issues: List[ValidationIssue]) -> None:
    for config_file in REQUIRED_CONFIG_FILES:
        path = project_root / config_file
        if not path.exists() or not path.is_file():
            add_issue(
                issues,
                "config_file",
                config_file,
                "Required configuration file is missing.",
            )


def validate_source_files(project_root: Path, issues: List[ValidationIssue]) -> None:
    for source_file in REQUIRED_SOURCE_FILES:
        path = project_root / source_file
        if not path.exists() or not path.is_file():
            add_issue(
                issues,
                "source_file",
                source_file,
                "Required source file is missing.",
            )


def validate_yaml_files(project_root: Path, issues: List[ValidationIssue]) -> None:
    for yaml_file in (project_root / "config").rglob("*.yaml"):
        relative_path = str(yaml_file.relative_to(project_root))

        try:
            with yaml_file.open("r", encoding="utf-8") as file:
                yaml.safe_load(file)

        except Exception as error:
            add_issue(
                issues,
                "yaml_parse",
                relative_path,
                f"YAML parse failed: {error}",
            )


def validate_pipeline_yaml(project_root: Path, issues: List[ValidationIssue]) -> None:
    pipeline_config = project_root / "config/pipeline/medfabric_platform.yaml"

    if not pipeline_config.exists():
        add_issue(
            issues,
            "pipeline_yaml",
            str(pipeline_config),
            "Pipeline YAML is missing.",
        )
        return

    try:
        with pipeline_config.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except Exception as error:
        add_issue(
            issues,
            "pipeline_yaml",
            str(pipeline_config),
            f"Pipeline YAML parse failed: {error}",
        )
        return

    if not isinstance(config, dict):
        add_issue(
            issues,
            "pipeline_yaml",
            str(pipeline_config),
            "Pipeline YAML must be a mapping.",
        )
        return

    layers = config.get("layers")

    if not isinstance(layers, list) or not layers:
        add_issue(
            issues,
            "pipeline_yaml",
            "layers",
            "Pipeline YAML must define a non-empty layers list.",
        )
        return

    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            add_issue(
                issues,
                "pipeline_yaml",
                f"layers[{index}]",
                "Layer entry must be a mapping.",
            )
            continue

        for key in ["name", "enabled", "module", "description"]:
            if key not in layer:
                add_issue(
                    issues,
                    "pipeline_yaml",
                    f"layers[{index}]",
                    f"Layer entry missing required key: {key}",
                )


def validate_importable_modules(issues: List[ValidationIssue]) -> None:
    for module_name in PIPELINE_MODULE_KEYS:
        try:
            importlib.import_module(module_name)

        except Exception as error:
            add_issue(
                issues,
                "module_import",
                module_name,
                f"Module import failed: {error}",
            )


def validate_init_files(project_root: Path, issues: List[ValidationIssue]) -> None:
    src_root = project_root / "src"

    if not src_root.exists():
        return

    for directory in src_root.rglob("*"):
        if not directory.is_dir():
            continue

        init_file = directory / "__init__.py"

        if not init_file.exists():
            add_issue(
                issues,
                "package_init",
                str(directory.relative_to(project_root)),
                "Missing __init__.py file.",
                severity="WARNING",
            )


def validate_cache_cleanliness(project_root: Path, issues: List[ValidationIssue]) -> None:
    cache_dirs = list(project_root.rglob("__pycache__"))

    for cache_dir in cache_dirs:
        add_issue(
            issues,
            "cache",
            str(cache_dir.relative_to(project_root)),
            "Python cache directory exists.",
            severity="WARNING",
        )


def print_validation_report(issues: List[ValidationIssue]) -> None:
    error_count = sum(1 for issue in issues if issue.severity == "ERROR")
    warning_count = sum(1 for issue in issues if issue.severity == "WARNING")

    print("")
    print("=" * 80)
    print("MedFabric Platform Validation Report")
    print("=" * 80)
    print(f"Errors   : {error_count}")
    print(f"Warnings : {warning_count}")
    print("=" * 80)

    if not issues:
        print("Validation passed. No issues found.")
        print("=" * 80)
        return

    for issue in issues:
        print("")
        print(f"[{issue.severity}] {issue.category}")
        print(f"Path    : {issue.path}")
        print(f"Message : {issue.message}")

    print("")
    print("=" * 80)

    if error_count > 0:
        print("Validation failed.")
    else:
        print("Validation completed with warnings.")

    print("=" * 80)


def main() -> None:
    project_root = get_project_root()
    issues: List[ValidationIssue] = []

    validate_project_root(project_root, issues)
    validate_directories(project_root, issues)
    validate_config_files(project_root, issues)
    validate_source_files(project_root, issues)
    validate_yaml_files(project_root, issues)
    validate_pipeline_yaml(project_root, issues)
    validate_importable_modules(issues)
    validate_init_files(project_root, issues)
    validate_cache_cleanliness(project_root, issues)

    print_validation_report(issues)

    error_count = sum(1 for issue in issues if issue.severity == "ERROR")

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()