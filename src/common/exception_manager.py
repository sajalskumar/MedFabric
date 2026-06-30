###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/exception_manager.py
#
# Purpose:
#     Defines standardized exception classes used across the MedFabric platform.
#
# Business Context:
#     MedFabric uses a reusable Foundation layer. Standard exceptions make
#     failures easier to classify, log, debug, and explain across configuration,
#     storage, validation, metadata, pipeline, governance, and modeling modules.
#
# Inputs:
#     None
#
# Outputs:
#     Standard MedFabric exception classes.
#
# Dependencies:
#     Python standard library only.
#
# Used By:
#     src/common/configuration_manager.py
#     src/common/path_manager.py
#     src/common/logging_manager.py
#     src/common/storage_manager.py
#     src/common/validation_manager.py
#     src/common/metadata_manager.py
#     src/pipeline/*
#     src/governance/*
#     src/modeling/*
#
# Processing Steps:
#     1. Define base MedFabricError.
#     2. Define category-specific platform exceptions.
#     3. Provide a simple standalone test through main().
#
# Example Run Command:
#     python -m src.common.exception_manager
#
# Expected Output:
#     MedFabric exception classes loaded successfully.
###############################################################################


class MedFabricError(Exception):
    """
    Base exception for all MedFabric platform errors.

    Purpose
    -------
    Provides a single parent exception class for all custom MedFabric errors.

    Notes
    -----
    Catch this exception when a caller wants to handle any platform-specific
    failure without catching unrelated Python exceptions.
    """


class ConfigurationError(MedFabricError):
    """
    Raised when configuration loading, parsing, or validation fails.
    """


class PathError(MedFabricError):
    """
    Raised when path resolution or directory management fails.
    """


class LoggingError(MedFabricError):
    """
    Raised when logging initialization or logging operations fail.
    """


class StorageError(MedFabricError):
    """
    Raised when file input/output or storage operations fail.
    """


class ValidationError(MedFabricError):
    """
    Raised when data validation or quality checks fail.
    """


class MetadataError(MedFabricError):
    """
    Raised when metadata generation or metadata writing fails.
    """


class PipelineError(MedFabricError):
    """
    Raised when pipeline orchestration or pipeline execution fails.
    """


class ModelError(MedFabricError):
    """
    Raised when model training, scoring, evaluation, or registry operations fail.
    """


class GovernanceError(MedFabricError):
    """
    Raised when governance processing or governance output generation fails.
    """


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.exception_manager
    """
    print("MedFabric exception classes loaded successfully.")


if __name__ == "__main__":
    main()