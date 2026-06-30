###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/configuration_manager.py
#
# Purpose:
#     Provides library-quality configuration management for the MedFabric
#     platform.
#
# Business Context:
#     MedFabric is a configuration-driven healthcare data platform. This module
#     centralizes YAML loading, caching, validation, and configuration access so
#     business modules do not directly parse YAML files.
#
# Inputs:
#     config/app.yaml
#     config/paths.yaml
#     config/logging.yaml
#     config/pipeline.yaml
#     Any additional YAML configuration file under config/
#
# Outputs:
#     In-memory Python dictionaries containing validated configuration content.
#
# Dependencies:
#     PyYAML
#     pathlib
#     typing
#     copy
#
# Used By:
#     src/common/path_manager.py
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
#     load_yaml()
#     get_section()
#     load_core_configs()
#     get_app_config()
#     get_paths_config()
#     get_logging_config()
#     get_pipeline_config()
#     require_sections()
#     clear_cache()
#     reload()
#     list_cached_configs()
#
# Example Run Command:
#     python -m src.common.configuration_manager
#
# Expected Output:
#     MedFabric configuration manager validation completed successfully.
###############################################################################

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.common.exception_manager import ConfigurationError


class ConfigurationManager:
    """
    Centralized configuration manager for MedFabric.

    This class owns all YAML configuration loading for the platform.

    Business modules should not directly import or call the YAML parser.
    Instead, they should request configuration through this manager.

    Parameters
    ----------
    config_root:
        Root directory where YAML configuration files are stored.
        Default is "config".

    use_cache:
        Whether loaded configuration files should be cached in memory.
        Default is True.
    """

    CORE_CONFIG_FILES = {
        "app": "app.yaml",
        "paths": "paths.yaml",
        "logging": "logging.yaml",
        "pipeline": "pipeline.yaml",
    }

    def __init__(self, config_root: str | Path = "config", use_cache: bool = True) -> None:
        """
        Initialize the ConfigurationManager.

        Parameters
        ----------
        config_root:
            Path to the configuration directory.

        use_cache:
            Enables or disables configuration caching.
        """
        self.config_root = Path(config_root)
        self.use_cache = use_cache
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_yaml(self, file_name: str, use_cache: Optional[bool] = None) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Parameters
        ----------
        file_name:
            YAML file name relative to the configuration root.
            Example: "app.yaml" or "models/readmission.yaml".

        use_cache:
            Optional override for cache behavior.
            If None, the manager-level cache setting is used.

        Returns
        -------
        dict
            Parsed YAML content.

        Raises
        ------
        ConfigurationError
            Raised when the file is missing, empty, invalid, or not a mapping.
        """
        should_use_cache = self.use_cache if use_cache is None else use_cache
        normalized_file_name = self._normalize_file_name(file_name)

        if should_use_cache and normalized_file_name in self._cache:
            return deepcopy(self._cache[normalized_file_name])

        config_path = self._resolve_config_path(normalized_file_name)

        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}"
            )

        if not config_path.is_file():
            raise ConfigurationError(
                f"Configuration path is not a file: {config_path}"
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

        except yaml.YAMLError as error:
            raise ConfigurationError(
                f"Invalid YAML syntax in configuration file: {config_path}"
            ) from error

        except OSError as error:
            raise ConfigurationError(
                f"Unable to read configuration file: {config_path}"
            ) from error

        if config_data is None:
            raise ConfigurationError(
                f"Configuration file is empty: {config_path}"
            )

        if not isinstance(config_data, dict):
            raise ConfigurationError(
                f"Configuration file must contain a YAML mapping: {config_path}"
            )

        if should_use_cache:
            self._cache[normalized_file_name] = deepcopy(config_data)

        return deepcopy(config_data)

    def get_section(
        self,
        file_name: str,
        section_name: str,
        required: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a top-level section from a YAML configuration file.

        Parameters
        ----------
        file_name:
            YAML file name relative to the configuration root.

        section_name:
            Top-level YAML section to retrieve.

        required:
            If True, raise ConfigurationError when the section is missing.

        Returns
        -------
        dict or None
            Section content when found. None when missing and required=False.

        Raises
        ------
        ConfigurationError
            Raised when a required section is missing or the section is invalid.
        """
        config_data = self.load_yaml(file_name)

        if section_name not in config_data:
            if required:
                raise ConfigurationError(
                    f"Required section '{section_name}' missing from {file_name}"
                )
            return None

        section = config_data[section_name]

        if section is not None and not isinstance(section, dict):
            raise ConfigurationError(
                f"Section '{section_name}' in {file_name} must be a YAML mapping"
            )

        return deepcopy(section)

    def load_core_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all required core MedFabric configuration files.

        Returns
        -------
        dict
            Dictionary containing app, paths, logging, and pipeline configs.
        """
        core_configs: Dict[str, Dict[str, Any]] = {}

        for config_name, file_name in self.CORE_CONFIG_FILES.items():
            core_configs[config_name] = self.load_yaml(file_name)

        return core_configs

    def get_app_config(self) -> Dict[str, Any]:
        """
        Load and return config/app.yaml.

        Returns
        -------
        dict
            Application configuration.
        """
        return self.load_yaml(self.CORE_CONFIG_FILES["app"])

    def get_paths_config(self) -> Dict[str, Any]:
        """
        Load and return config/paths.yaml.

        Returns
        -------
        dict
            Paths configuration.
        """
        return self.load_yaml(self.CORE_CONFIG_FILES["paths"])

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Load and return config/logging.yaml.

        Returns
        -------
        dict
            Logging configuration.
        """
        return self.load_yaml(self.CORE_CONFIG_FILES["logging"])

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Load and return config/pipeline.yaml.

        Returns
        -------
        dict
            Pipeline configuration.
        """
        return self.load_yaml(self.CORE_CONFIG_FILES["pipeline"])

    def require_sections(self, file_name: str, required_sections: List[str]) -> None:
        """
        Validate that a YAML file contains all required top-level sections.

        Parameters
        ----------
        file_name:
            YAML file name relative to the configuration root.

        required_sections:
            List of required top-level section names.

        Raises
        ------
        ConfigurationError
            Raised when one or more required sections are missing.
        """
        config_data = self.load_yaml(file_name)

        missing_sections = [
            section_name
            for section_name in required_sections
            if section_name not in config_data
        ]

        if missing_sections:
            raise ConfigurationError(
                f"Configuration file '{file_name}' is missing required sections: "
                f"{missing_sections}"
            )

    def clear_cache(self) -> None:
        """
        Clear all cached configuration content.

        Returns
        -------
        None
        """
        self._cache.clear()

    def reload(self, file_name: Optional[str] = None) -> Dict[str, Any] | Dict[str, Dict[str, Any]]:
        """
        Reload one configuration file or all core configuration files.

        Parameters
        ----------
        file_name:
            Optional YAML file name. If provided, only this file is reloaded.
            If omitted, all core configuration files are reloaded.

        Returns
        -------
        dict
            Reloaded configuration content.
        """
        if file_name is not None:
            normalized_file_name = self._normalize_file_name(file_name)
            self._cache.pop(normalized_file_name, None)
            return self.load_yaml(normalized_file_name, use_cache=True)

        for core_file_name in self.CORE_CONFIG_FILES.values():
            normalized_file_name = self._normalize_file_name(core_file_name)
            self._cache.pop(normalized_file_name, None)

        return self.load_core_configs()

    def list_cached_configs(self) -> List[str]:
        """
        Return a list of cached configuration file names.

        Returns
        -------
        list[str]
            Cached configuration file names.
        """
        return sorted(self._cache.keys())

    def validate_core_configs(self) -> None:
        """
        Validate that required core configuration files and sections exist.

        Raises
        ------
        ConfigurationError
            Raised when a required file or section is missing.
        """
        self.require_sections(
            "app.yaml",
            required_sections=[
                "application",
                "runtime",
                "platform",
                "data",
                "development",
            ],
        )

        self.require_sections(
            "paths.yaml",
            required_sections=[
                "project",
                "config",
                "documentation",
                "source",
                "data",
                "logs",
                "models",
                "notebooks",
                "tests",
            ],
        )

        self.require_sections(
            "logging.yaml",
            required_sections=[
                "logging",
                "destinations",
                "format",
                "rotation",
                "metrics",
                "errors",
                "audit",
            ],
        )

        self.require_sections(
            "pipeline.yaml",
            required_sections=[
                "pipeline",
                "execution",
                "layers",
                "execution_order",
                "validation",
                "performance",
                "outputs",
                "failure_handling",
                "notifications",
            ],
        )

    def _resolve_config_path(self, file_name: str) -> Path:
        """
        Resolve a YAML file name to an absolute path.

        Parameters
        ----------
        file_name:
            YAML file name relative to the configuration root.

        Returns
        -------
        Path
            Resolved configuration file path.
        """
        return (self.config_root / file_name).resolve()

    @staticmethod
    def _normalize_file_name(file_name: str) -> str:
        """
        Normalize configuration file names for consistent cache keys.

        Parameters
        ----------
        file_name:
            Raw file name.

        Returns
        -------
        str
            Normalized file name using forward slashes.
        """
        return str(Path(file_name)).replace("\\", "/")


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.configuration_manager

    Expected Output
    ---------------
    MedFabric configuration manager validation completed successfully.
    Core configs loaded: ['app', 'logging', 'paths', 'pipeline']
    """
    manager = ConfigurationManager()

    manager.validate_core_configs()
    core_configs = manager.load_core_configs()

    loaded_config_names = sorted(core_configs.keys())

    print("MedFabric configuration manager validation completed successfully.")
    print(f"Core configs loaded: {loaded_config_names}")


if __name__ == "__main__":
    main()