###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     scripts/doctor.py
#
# Purpose:
#     Runs a broad MedFabric system health check.
#
# Run:
#     python scripts/doctor.py
#
###############################################################################

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_DIRS = [
    "src",
    "config",
    "docs",
    "scripts",
    "reference",
    "data",
    "models",
    "logs",
    "tests",
]

REQUIRED_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    ".gitignore",
    "config/pipeline.yaml",
    "config/pipeline/medfabric_platform.yaml",
]

REQUIRED_MODULES = [
    "src.data_generation.build_raw_data",
    "src.ingestion.build_ingestion_layer",
    "src.silver.build_silver_layer",
    "src.gold.build_gold_layer",
    "src.feature_store.build_feature_store",
    "src.modeling.build_modeling_layer",
    "src.semantic_layer.build_semantic_layer",
    "src.analytics_platform.build_analytics_platform",
    "src.insights.build_insights_platform",
    "src.pipeline.build_medfabric_platform",
]


@dataclass
class CheckResult:
    name: str
    status: str
    message: str


def check(condition: bool, name: str, success: str, failure: str) -> CheckResult:
    return CheckResult(name, "PASS" if condition else "FAIL", success if condition else failure)


def run_command(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def check_python_version() -> CheckResult:
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 12
    return check(
        ok,
        "Python Version",
        f"Python {version.major}.{version.minor}.{version.micro}",
        f"Python 3.12+ recommended. Current: {version.major}.{version.minor}.{version.micro}",
    )


def check_virtual_environment() -> CheckResult:
    ok = sys.prefix != sys.base_prefix
    return check(ok, "Virtual Environment", "Virtual environment is active.", "Virtual environment is not active.")


def check_project_root(root: Path) -> CheckResult:
    ok = (root / "src").exists() and (root / "config").exists()
    return check(ok, "Project Root", f"Project root looks valid: {root}", "Run from MedFabric project root.")


def check_directories(root: Path) -> List[CheckResult]:
    return [
        check((root / d).exists(), f"Directory: {d}", "Exists.", "Missing.")
        for d in REQUIRED_DIRS
    ]


def check_files(root: Path) -> List[CheckResult]:
    return [
        check((root / f).exists(), f"File: {f}", "Exists.", "Missing.")
        for f in REQUIRED_FILES
    ]


def check_yaml_files(root: Path) -> List[CheckResult]:
    results: List[CheckResult] = []

    for path in sorted((root / "config").rglob("*.yaml")):
        relative = str(path.relative_to(root))
        try:
            with path.open("r", encoding="utf-8") as file:
                yaml.safe_load(file)
            results.append(CheckResult(f"YAML: {relative}", "PASS", "Parses successfully."))
        except Exception as error:
            results.append(CheckResult(f"YAML: {relative}", "FAIL", str(error)))

    return results


def check_imports() -> List[CheckResult]:
    results: List[CheckResult] = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            results.append(CheckResult(f"Import: {module_name}", "PASS", "Import successful."))
        except Exception as error:
            results.append(CheckResult(f"Import: {module_name}", "FAIL", str(error)))

    return results


def check_required_packages() -> CheckResult:
    packages = ["pandas", "yaml", "pyarrow"]
    missing = []

    for package in packages:
        try:
            if package == "yaml":
                import yaml  # noqa: F401
            else:
                importlib.import_module(package)
        except Exception:
            missing.append(package)

    return check(
        not missing,
        "Required Packages",
        "Required packages are available.",
        f"Missing packages: {missing}",
    )


def check_git_available() -> CheckResult:
    git_path = shutil.which("git")
    return check(git_path is not None, "Git", "Git is available.", "Git is not available.")


def check_git_status() -> CheckResult:
    code, output = run_command(["git", "status", "--short"])

    if code != 0:
        return CheckResult("Git Status", "WARNING", "Git status could not be checked.")

    if output.strip():
        return CheckResult("Git Status", "WARNING", "Working tree has uncommitted changes.")

    return CheckResult("Git Status", "PASS", "Working tree is clean.")


def check_reference_data(root: Path) -> CheckResult:
    reference = root / "reference"
    parquet_count = len(list(reference.rglob("*.parquet"))) if reference.exists() else 0
    return check(parquet_count > 0, "Reference Data", f"{parquet_count} parquet files found.", "No reference parquet files found.")


def check_generated_data(root: Path) -> CheckResult:
    data = root / "data"
    parquet_count = len(list(data.rglob("*.parquet"))) if data.exists() else 0
    status = "PASS" if parquet_count > 0 else "WARNING"
    message = f"{parquet_count} generated parquet files found."
    return CheckResult("Generated Data", status, message)


def check_logs(root: Path) -> CheckResult:
    logs = root / "logs"
    log_count = len(list(logs.rglob("*"))) if logs.exists() else 0
    return check(logs.exists(), "Logs", f"Logs folder exists. Items: {log_count}", "Logs folder missing.")


def check_cache_cleanliness(root: Path) -> CheckResult:
    cache_count = len(list(root.rglob("__pycache__")))
    status = "PASS" if cache_count == 0 else "WARNING"
    message = "No __pycache__ folders found." if cache_count == 0 else f"{cache_count} __pycache__ folders found."
    return CheckResult("Python Cache", status, message)


def print_report(results: List[CheckResult]) -> None:
    pass_count = sum(1 for r in results if r.status == "PASS")
    warning_count = sum(1 for r in results if r.status == "WARNING")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    print("")
    print("=" * 80)
    print("MedFabric Doctor")
    print("=" * 80)
    print(f"PASS    : {pass_count}")
    print(f"WARNING : {warning_count}")
    print(f"FAIL    : {fail_count}")
    print("=" * 80)

    for result in results:
        symbol = "✓" if result.status == "PASS" else "!" if result.status == "WARNING" else "✗"
        print(f"{symbol} {result.status:<8} {result.name:<45} {result.message}")

    print("=" * 80)

    if fail_count > 0:
        print("Overall Status: FAIL")
    elif warning_count > 0:
        print("Overall Status: WARNING")
    else:
        print("Overall Status: PASS")

    print("=" * 80)


def main() -> None:
    root = PROJECT_ROOT

    results: List[CheckResult] = []
    results.append(check_python_version())
    results.append(check_virtual_environment())
    results.append(check_project_root(root))
    results.append(check_required_packages())
    results.extend(check_directories(root))
    results.extend(check_files(root))
    results.extend(check_yaml_files(root))
    results.extend(check_imports())
    results.append(check_git_available())
    results.append(check_git_status())
    results.append(check_reference_data(root))
    results.append(check_generated_data(root))
    results.append(check_logs(root))
    results.append(check_cache_cleanliness(root))

    print_report(results)

    if any(result.status == "FAIL" for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()