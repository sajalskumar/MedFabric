###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/path_manager.py
#
# Purpose:
#     Provides centralized path resolution, path validation, and directory
#     management for the MedFabric platform.
#
# Business Context:
#     MedFabric is local-first and cloud-ready. This module ensures all local
#     paths are resolved consistently from YAML configuration, preventing
#     hardcoded paths throughout business modules.
#
# Inputs:
#     config/paths.yaml
#
# Outputs:
#     Resolved pathlib.Path objects
#     Created directories when required
#     Path validation results
#
# Dependencies:
#     pathlib
#     typing
#     src.common.configuration_manager.ConfigurationManager
#     src.common.exception_manager.PathError
#
# Used By:
#     src/common/logging_manager.py
#     src/common/storage_manager.py
#     src/pipeline/*
#     src/data_generation/*
#     src/ingestion/*
#     src/silver/*
#     src/gold/*
#     src/feature_store/*
#     src/modeling/*
#     src/scoring/*
#     src/governance/*
#
# Public Interface:
#     resolve_path()
#     get_path()
#     ensure_directory()
#     ensure_core_directories()
#     file_exists()
#     directory_exists()
#     validate_required_directories()
#     get_project_root()
#     get_data_path()
#     get_log_path()
#     get_model_path()
#
# Example Run Command:
#     python -m src.common.path_manager
#
# Expected Output:
#     MedFabric path manager validation completed successfully.
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.common.configuration_manager import ConfigurationManager
from src.common.exception_manager import PathError


class PathManager:
    """
    Centralized path manager for MedFabric.

    This class owns local path resolution and directory creation.

    Business modules should not directly create directories or hardcode project
    paths. They should request paths through this manager.

    Parameters
    ----------
    configuration_manager:
        Active ConfigurationManager instance used to load paths.yaml.
    """

    DIRECTORY_SECTIONS = [
        "config",
        "documentation",
        "source",
        "data",
        "logs",
        "models",
        "notebooks",
        "tests",
        "temporary",
    ]

    def __init__(self, configuration_manager: ConfigurationManager) -> None:
        """
        Initialize PathManager.

        Parameters
        ----------
        configuration_manager:
            Active MedFabric ConfigurationManager instance.

        Raises
        ------
        PathError
            Raised when paths.yaml is missing required sections.
        """
        self.configuration_manager = configuration_manager
        self.paths_config = self.configuration_manager.get_paths_config()
        self.project_root = self._load_project_root()

    def resolve_path(self, path_value: str | Path) -> Path:
        """
        Resolve a path relative to the MedFabric project root.

        Parameters
        ----------
        path_value:
            Path value from configuration or caller input.

        Returns
        -------
        Path
            Fully resolved local path.

        Raises
        ------
        PathError
            Raised when the path value is empty or invalid.
        """
        if path_value is None:
            raise PathError("Cannot resolve an empty path value.")

        raw_path = Path(path_value)

        if raw_path == Path(""):
            raise PathError("Cannot resolve an empty path value.")

        if raw_path.is_absolute():
            return raw_path.resolve()

        return (self.project_root / raw_path).resolve()

    def get_path(self, section: str, key: str) -> Path:
        """
        Return a configured path by section and key.

        Parameters
        ----------
        section:
            Top-level section in config/paths.yaml.

        key:
            Path key inside the selected section.

        Returns
        -------
        Path
            Resolved path.

        Raises
        ------
        PathError
            Raised when the section or key is missing.
        """
        if section not in self.paths_config:
            raise PathError(f"Missing path section in paths.yaml: {section}")

        section_data = self.paths_config[section]

        if not isinstance(section_data, dict):
            raise PathError(
                f"Path section '{section}' must be a YAML mapping."
            )

        if key not in section_data:
            raise PathError(
                f"Missing path key in paths.yaml: {section}.{key}"
            )

        return self.resolve_path(section_data[key])

    def ensure_directory(self, path: str | Path) -> Path:
        """
        Create a directory if it does not already exist.

        Parameters
        ----------
        path:
            Directory path to create.

        Returns
        -------
        Path
            Resolved directory path.

        Raises
        ------
        PathError
            Raised when directory creation fails.
        """
        directory_path = self.resolve_path(path)

        try:
            directory_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise PathError(
                f"Unable to create directory: {directory_path}"
            ) from error

        return directory_path

    def ensure_core_directories(self) -> List[Path]:
        """
        Create all configured MedFabric core directories.

        Returns
        -------
        list[Path]
            List of directories that were validated or created.

        Raises
        ------
        PathError
            Raised when any configured directory cannot be created.
        """
        created_or_validated_directories: List[Path] = []

        for section_name in self.DIRECTORY_SECTIONS:
            section_data = self.paths_config.get(section_name)

            if section_data is None:
                continue

            if not isinstance(section_data, dict):
                raise PathError(
                    f"Path section '{section_name}' must be a YAML mapping."
                )

            for path_key, path_value in section_data.items():
                if path_key == "root" or isinstance(path_value, str):
                    directory_path = self.ensure_directory(path_value)
                    created_or_validated_directories.append(directory_path)

        return created_or_validated_directories

    def file_exists(self, path: str | Path) -> bool:
        """
        Check whether a file exists.

        Parameters
        ----------
        path:
            File path to check.

        Returns
        -------
        bool
            True if the path exists and is a file.
        """
        resolved_path = self.resolve_path(path)
        return resolved_path.exists() and resolved_path.is_file()

    def directory_exists(self, path: str | Path) -> bool:
        """
        Check whether a directory exists.

        Parameters
        ----------
        path:
            Directory path to check.

        Returns
        -------
        bool
            True if the path exists and is a directory.
        """
        resolved_path = self.resolve_path(path)
        return resolved_path.exists() and resolved_path.is_dir()

    def validate_required_directories(self) -> None:
        """
        Validate that all configured core directories exist.

        Raises
        ------
        PathError
            Raised when one or more required directories are missing.
        """
        missing_directories: List[str] = []

        for section_name in self.DIRECTORY_SECTIONS:
            section_data = self.paths_config.get(section_name)

            if section_data is None:
                continue

            if not isinstance(section_data, dict):
                raise PathError(
                    f"Path section '{section_name}' must be a YAML mapping."
                )

            for _, path_value in section_data.items():
                if not isinstance(path_value, str):
                    continue

                resolved_path = self.resolve_path(path_value)

                if not resolved_path.exists() or not resolved_path.is_dir():
                    missing_directories.append(str(resolved_path))

        if missing_directories:
            raise PathError(
                "Required directories are missing: "
                f"{missing_directories}"
            )

    def get_project_root(self) -> Path:
        """
        Return the resolved MedFabric project root.

        Returns
        -------
        Path
            Project root path.
        """
        return self.project_root

    def get_data_path(self, key: str) -> Path:
        """
        Return a configured data path.

        Parameters
        ----------
        key:
            Data path key from paths.yaml.
            Example: raw, bronze, silver, gold, metadata.

        Returns
        -------
        Path
            Resolved data path.
        """
        return self.get_path("data", key)

    def get_log_path(self, key: str) -> Path:
        """
        Return a configured log path.

        Parameters
        ----------
        key:
            Log path key from paths.yaml.
            Example: pipeline, modules, errors, audit.

        Returns
        -------
        Path
            Resolved log path.
        """
        return self.get_path("logs", key)

    def get_model_path(self, key: str = "root") -> Path:
        """
        Return a configured model path.

        Parameters
        ----------
        key:
            Model path key from paths.yaml. Default is "root".

        Returns
        -------
        Path
            Resolved model path.
        """
        return self.get_path("models", key)

    def get_all_configured_paths(self) -> Dict[str, Path]:
        """
        Return all configured local paths as resolved Path objects.

        Returns
        -------
        dict
            Mapping of "section.key" to resolved Path.
        """
        resolved_paths: Dict[str, Path] = {}

        for section_name, section_data in self.paths_config.items():
            if not isinstance(section_data, dict):
                continue

            for key, value in section_data.items():
                if isinstance(value, str):
                    continue

                if value.strip() == "":
                    continue
                
                    resolved_paths[f"{section_name}.{key}"] = self.resolve_path(value)

        return resolved_paths

    def _load_project_root(self) -> Path:
        """
        Load and resolve the project root from paths.yaml.

        Returns
        -------
        Path
            Resolved project root.

        Raises
        ------
        PathError
            Raised when project.root is missing.
        """
        try:
            project_root_value = self.paths_config["project"]["root"]
        except KeyError as error:
            raise PathError(
                "Required path setting missing: project.root"
            ) from error

        return Path(project_root_value).resolve()


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.path_manager

    Expected Output
    ---------------
    MedFabric path manager validation completed successfully.
    Project root: <resolved project root>
    Total configured paths: <count>
    """
    configuration_manager = ConfigurationManager()
    path_manager = PathManager(configuration_manager)

    path_manager.ensure_core_directories()
    path_manager.validate_required_directories()

    all_paths = path_manager.get_all_configured_paths()

    print("MedFabric path manager validation completed successfully.")
    print(f"Project root: {path_manager.get_project_root()}")
    print(f"Total configured paths: {len(all_paths)}")


if __name__ == "__main__":
    main()