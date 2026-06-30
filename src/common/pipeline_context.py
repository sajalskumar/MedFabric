###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/pipeline_context.py
#
# Purpose:
#     Defines the shared execution context used across MedFabric pipeline
#     components.
#
# Business Context:
#     MedFabric pipeline modules need consistent access to run-level information,
#     configuration, logging, paths, storage, validation, and metadata services.
#     This module provides a single context object so pipeline steps do not need
#     to pass many separate Foundation objects between functions.
#
# Inputs:
#     ConfigurationManager
#     PathManager
#     LoggingManager
#     StorageManager
#     ValidationManager
#     MetadataManager
#
# Outputs:
#     PipelineContext object containing run metadata and Foundation services.
#
# Dependencies:
#     dataclasses
#     datetime
#     getpass
#     typing
#     src.common.configuration_manager.ConfigurationManager
#     src.common.path_manager.PathManager
#     src.common.logging_manager.LoggingManager
#     src.common.storage_manager.StorageManager
#     src.common.validation_manager.ValidationManager
#     src.common.metadata_manager.MetadataManager
#     src.common.exception_manager.PipelineError
#
# Used By:
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
#     PipelineContext
#     create_pipeline_context()
#     to_dict()
#
# Example Run Command:
#     python -m src.common.pipeline_context
#
# Expected Output:
#     MedFabric pipeline context validation completed successfully.
###############################################################################

from __future__ import annotations

import getpass
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.common.configuration_manager import ConfigurationManager
from src.common.exception_manager import PipelineError
from src.common.logging_manager import LoggingManager
from src.common.metadata_manager import MetadataManager
from src.common.path_manager import PathManager
from src.common.storage_manager import StorageManager
from src.common.validation_manager import ValidationManager


@dataclass
class PipelineContext:
    """
    Shared execution context for MedFabric pipeline components.

    Purpose
    -------
    Provides a single object containing run metadata and Foundation services.

    Notes
    -----
    Business modules should receive PipelineContext instead of receiving many
    independent Foundation objects.
    """

    run_id: str
    pipeline_name: str
    environment: str
    application_version: str
    start_time: datetime
    user: str

    configuration: ConfigurationManager
    paths: PathManager
    logging: LoggingManager
    storage: StorageManager
    validation: ValidationManager
    metadata: MetadataManager

    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pipeline context metadata to a dictionary.

        Returns
        -------
        dict
            Run-level context information suitable for logging, audit, or
            pipeline history output.
        """
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "environment": self.environment,
            "application_version": self.application_version,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "user": self.user,
            "extra_metadata": self.extra_metadata,
        }

    def get_logger(self, module_name: str):
        """
        Convenience method to retrieve a configured logger.

        Parameters
        ----------
        module_name:
            Name of the module requesting a logger.

        Returns
        -------
        logging.Logger
            Configured logger from LoggingManager.
        """
        return self.logging.get_logger(module_name)


def create_pipeline_context(
    pipeline_name: Optional[str] = None,
    environment: Optional[str] = None,
    run_id: Optional[str] = None,
) -> PipelineContext:
    """
    Create a fully initialized MedFabric PipelineContext.

    Parameters
    ----------
    pipeline_name:
        Optional pipeline name override. If omitted, the value is read from
        config/pipeline.yaml.

    environment:
        Optional environment override. If omitted, the value is read from
        config/app.yaml.

    run_id:
        Optional run ID override. If omitted, LoggingManager generates one.

    Returns
    -------
    PipelineContext
        Fully initialized execution context.

    Raises
    ------
    PipelineError
        Raised when context creation fails.
    """
    try:
        configuration_manager = ConfigurationManager()
        configuration_manager.validate_core_configs()

        app_config = configuration_manager.get_app_config()
        pipeline_config = configuration_manager.get_pipeline_config()

        path_manager = PathManager(configuration_manager)
        path_manager.ensure_core_directories()

        logging_manager = LoggingManager(
            configuration_manager=configuration_manager,
            path_manager=path_manager,
            run_id=run_id,
        )
        logging_manager.initialize()

        storage_manager = StorageManager(path_manager)
        validation_manager = ValidationManager()
        metadata_manager = MetadataManager(storage_manager)

        resolved_pipeline_name = (
            pipeline_name
            or pipeline_config.get("pipeline", {}).get("name")
            or "MedFabric Pipeline"
        )

        resolved_environment = (
            environment
            or app_config.get("application", {}).get("environment")
            or "local"
        )

        application_version = (
            app_config.get("application", {}).get("version")
            or "unknown"
        )

        context = PipelineContext(
            run_id=logging_manager.get_run_id(),
            pipeline_name=resolved_pipeline_name,
            environment=resolved_environment,
            application_version=application_version,
            start_time=datetime.now(),
            user=getpass.getuser(),
            configuration=configuration_manager,
            paths=path_manager,
            logging=logging_manager,
            storage=storage_manager,
            validation=validation_manager,
            metadata=metadata_manager,
        )

        logger = context.get_logger("medfabric.pipeline_context")
        logger.info("PipelineContext created successfully.")
        logger.info("Pipeline name: %s", context.pipeline_name)
        logger.info("Environment: %s", context.environment)

        return context

    except Exception as error:
        if isinstance(error, PipelineError):
            raise

        raise PipelineError("Failed to create MedFabric PipelineContext.") from error


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.pipeline_context

    Expected Output
    ---------------
    MedFabric pipeline context validation completed successfully.
    """
    context = create_pipeline_context()

    logger = context.get_logger("medfabric.pipeline_context.test")
    logger.info("Testing PipelineContext standalone validation.")

    context_dict = context.to_dict()

    required_keys = [
        "run_id",
        "pipeline_name",
        "environment",
        "application_version",
        "start_time",
        "user",
    ]

    missing_keys = [
        key for key in required_keys if key not in context_dict
    ]

    if missing_keys:
        raise PipelineError(
            f"PipelineContext dictionary is missing keys: {missing_keys}"
        )

    print("MedFabric pipeline context validation completed successfully.")
    print(f"Run ID: {context.run_id}")
    print(f"Pipeline Name: {context.pipeline_name}")
    print(f"Environment: {context.environment}")

    context.logging.close()


if __name__ == "__main__":
    main()