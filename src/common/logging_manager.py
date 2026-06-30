###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/logging_manager.py
#
# Purpose:
#     Provides centralized logging management for the MedFabric platform.
#
# Business Context:
#     MedFabric requires consistent logging across configuration, ingestion,
#     transformation, modeling, scoring, governance, and pipeline orchestration.
#     This module standardizes logger creation, run identifiers, pipeline logs,
#     module logs, error logs, audit logs, step timing, dataset logging, and
#     validation logging.
#
# Inputs:
#     config/logging.yaml
#     config/paths.yaml
#
# Outputs:
#     Console log messages
#     Pipeline log files under logs/pipeline/
#     Module log files under logs/modules/
#     Error log files under logs/errors/
#     Audit log files under logs/audit/
#
# Dependencies:
#     logging
#     logging.handlers
#     datetime
#     time
#     uuid
#     pathlib
#     typing
#     src.common.configuration_manager.ConfigurationManager
#     src.common.path_manager.PathManager
#     src.common.exception_manager.LoggingError
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
#     initialize()
#     get_logger()
#     start_step()
#     end_step()
#     log_dataset()
#     log_validation()
#     log_exception()
#     close()
#
# Example Run Command:
#     python -m src.common.logging_manager
#
# Expected Output:
#     Console log messages confirming successful logging manager validation.
#     Log files created under logs/pipeline, logs/modules, logs/errors, logs/audit.
###############################################################################

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.configuration_manager import ConfigurationManager
from src.common.exception_manager import LoggingError
from src.common.path_manager import PathManager


class RunIdFilter(logging.Filter):
    """
    Logging filter that injects a MedFabric run ID into every log record.

    Purpose
    -------
    Python logging formatters can reference custom fields only when those fields
    exist on the log record. This filter guarantees that every log record has
    a run_id attribute.
    """

    def __init__(self, run_id: str) -> None:
        """
        Initialize the run ID filter.

        Parameters
        ----------
        run_id:
            Current MedFabric execution run identifier.
        """
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add run_id to the log record.

        Parameters
        ----------
        record:
            Python logging record.

        Returns
        -------
        bool
            Always True so the log record continues processing.
        """
        record.run_id = self.run_id
        return True


class LoggingManager:
    """
    Centralized logging manager for MedFabric.

    This class owns all logger setup and logging conventions for the platform.

    Business modules should not create independent logging configurations.
    They should request loggers and logging behavior from this manager.

    Parameters
    ----------
    configuration_manager:
        Active ConfigurationManager instance.

    path_manager:
        Active PathManager instance.

    run_id:
        Optional run identifier. If omitted, a new run ID is generated.
    """

    def __init__(
        self,
        configuration_manager: ConfigurationManager,
        path_manager: PathManager,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize LoggingManager.

        Parameters
        ----------
        configuration_manager:
            Active MedFabric ConfigurationManager instance.

        path_manager:
            Active MedFabric PathManager instance.

        run_id:
            Optional run ID for this execution.
        """
        self.configuration_manager = configuration_manager
        self.path_manager = path_manager
        self.logging_config = self.configuration_manager.get_logging_config()

        self.run_id = run_id or self._generate_run_id()
        self.initialized = False

        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: list[logging.Handler] = []
        self._step_start_times: Dict[str, float] = {}

    def initialize(self) -> None:
        """
        Initialize centralized logging.

        This method creates console, pipeline, module, error, and audit handlers
        based on config/logging.yaml.

        Raises
        ------
        LoggingError
            Raised when logging initialization fails.
        """
        if self.initialized:
            return

        try:
            self.path_manager.ensure_core_directories()

            root_logger = logging.getLogger()
            root_logger.setLevel(self._get_log_level(self._get_default_level()))

            self._clear_existing_handlers(root_logger)

            handlers = self._build_handlers()

            for handler in handlers:
                root_logger.addHandler(handler)

            self._handlers = handlers
            self.initialized = True

            logger = self.get_logger("medfabric.logging_manager")
            logger.info("MedFabric logging initialized successfully.")
            logger.info("Run ID: %s", self.run_id)

        except Exception as error:
            raise LoggingError("Failed to initialize MedFabric logging.") from error

    def get_logger(self, module_name: str) -> logging.Logger:
        """
        Return a logger for a module.

        Parameters
        ----------
        module_name:
            Name of the module requesting a logger.

        Returns
        -------
        logging.Logger
            Configured Python logger.
        """
        if not module_name:
            raise LoggingError("Logger module name cannot be empty.")

        if module_name in self._loggers:
            return self._loggers[module_name]

        logger = logging.getLogger(module_name)
        logger.setLevel(self._get_log_level(self._get_default_level()))
        logger.propagate = True

        self._loggers[module_name] = logger
        return logger

    def start_step(self, step_name: str) -> None:
        """
        Log the beginning of a pipeline or module step.

        Parameters
        ----------
        step_name:
            Name of the step being started.
        """
        self._require_initialized()

        if not step_name:
            raise LoggingError("Step name cannot be empty.")

        self._step_start_times[step_name] = time.perf_counter()

        logger = self.get_logger("medfabric.step")
        logger.info("=" * 80)
        logger.info("START: %s", step_name)

    def end_step(self, step_name: str) -> None:
        """
        Log the completion of a pipeline or module step.

        Parameters
        ----------
        step_name:
            Name of the step being completed.
        """
        self._require_initialized()

        if not step_name:
            raise LoggingError("Step name cannot be empty.")

        logger = self.get_logger("medfabric.step")

        start_time = self._step_start_times.pop(step_name, None)

        if start_time is None:
            logger.warning("END called for step without START: %s", step_name)
            return

        duration_seconds = time.perf_counter() - start_time

        logger.info(
            "COMPLETE: %s | Duration: %.2f seconds",
            step_name,
            duration_seconds,
        )
        logger.info("=" * 80)

    def log_dataset(
        self,
        dataset_name: str,
        row_count: int,
        column_count: int,
        path: Optional[str | Path] = None,
    ) -> None:
        """
        Log dataset row and column counts.

        Parameters
        ----------
        dataset_name:
            Name of the dataset.

        row_count:
            Number of rows in the dataset.

        column_count:
            Number of columns in the dataset.

        path:
            Optional dataset path.
        """
        self._require_initialized()

        if not dataset_name:
            raise LoggingError("Dataset name cannot be empty.")

        logger = self.get_logger("medfabric.dataset")

        if path is None:
            logger.info(
                "DATASET: %s | Rows: %s | Columns: %s",
                dataset_name,
                row_count,
                column_count,
            )
        else:
            logger.info(
                "DATASET: %s | Rows: %s | Columns: %s | Path: %s",
                dataset_name,
                row_count,
                column_count,
                path,
            )

    def log_validation(
        self,
        validation_name: str,
        status: str,
        message: str,
    ) -> None:
        """
        Log validation results.

        Parameters
        ----------
        validation_name:
            Name of the validation rule or validation group.

        status:
            Validation status such as PASS, WARN, or FAIL.

        message:
            Human-readable validation message.
        """
        self._require_initialized()

        if not validation_name:
            raise LoggingError("Validation name cannot be empty.")

        normalized_status = status.upper().strip()

        logger = self.get_logger("medfabric.validation")

        if normalized_status == "PASS":
            logger.info("VALIDATION PASS: %s | %s", validation_name, message)
        elif normalized_status == "WARN":
            logger.warning("VALIDATION WARN: %s | %s", validation_name, message)
        elif normalized_status == "FAIL":
            logger.error("VALIDATION FAIL: %s | %s", validation_name, message)
        else:
            logger.info(
                "VALIDATION %s: %s | %s",
                normalized_status,
                validation_name,
                message,
            )

    def log_exception(self, error: Exception, message: Optional[str] = None) -> None:
        """
        Log an exception with stack trace.

        Parameters
        ----------
        error:
            Exception instance to log.

        message:
            Optional additional context message.
        """
        self._require_initialized()

        logger = self.get_logger("medfabric.exception")

        if message:
            logger.exception("%s | Error: %s", message, error)
        else:
            logger.exception("Unhandled exception: %s", error)

    def close(self) -> None:
        """
        Close all logging handlers managed by this LoggingManager.

        This is useful at the end of pipeline execution to flush file handlers.
        """
        root_logger = logging.getLogger()

        for handler in self._handlers:
            try:
                handler.flush()
                handler.close()
                root_logger.removeHandler(handler)
            except Exception:
                pass

        self._handlers.clear()
        self._loggers.clear()
        self.initialized = False

    def get_run_id(self) -> str:
        """
        Return the current run ID.

        Returns
        -------
        str
            Current MedFabric run identifier.
        """
        return self.run_id

    def _build_handlers(self) -> list[logging.Handler]:
        """
        Build logging handlers based on logging.yaml.

        Returns
        -------
        list[logging.Handler]
            Configured logging handlers.
        """
        handlers: list[logging.Handler] = []

        destinations = self.logging_config.get("destinations", {})
        format_config = self.logging_config.get("format", {})

        standard_format = format_config.get(
            "standard",
            "[%(asctime)s] [RUN_ID=%(run_id)s] [%(levelname)s] [%(name)s] %(message)s",
        )

        console_format = format_config.get(
            "console",
            "[%(levelname)s] %(message)s",
        )

        timestamp_format = self.logging_config.get("logging", {}).get(
            "timestamp_format",
            "%Y-%m-%d %H:%M:%S",
        )

        run_id_filter = RunIdFilter(self.run_id)

        if destinations.get("console", {}).get("enabled", True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(
                self._get_log_level(destinations["console"].get("level", "INFO"))
            )
            console_handler.setFormatter(
                logging.Formatter(console_format, datefmt=timestamp_format)
            )
            console_handler.addFilter(run_id_filter)
            handlers.append(console_handler)

        file_handler_specs = [
            ("pipeline_file", "pipeline"),
            ("module_file", "modules"),
            ("error_file", "errors"),
            ("audit_file", "audit"),
        ]

        for destination_key, log_path_key in file_handler_specs:
            destination_config = destinations.get(destination_key, {})

            if not destination_config.get("enabled", False):
                continue

            handler = self._create_file_handler(
                destination_key=destination_key,
                destination_config=destination_config,
                log_path_key=log_path_key,
                log_format=standard_format,
                timestamp_format=timestamp_format,
                run_id_filter=run_id_filter,
            )

            handlers.append(handler)

        return handlers

    def _create_file_handler(
        self,
        destination_key: str,
        destination_config: Dict[str, Any],
        log_path_key: str,
        log_format: str,
        timestamp_format: str,
        run_id_filter: RunIdFilter,
    ) -> logging.Handler:
        """
        Create a file handler for a configured logging destination.

        Parameters
        ----------
        destination_key:
            Logging destination key from logging.yaml.

        destination_config:
            Logging destination configuration.

        log_path_key:
            Key under paths.yaml logs section.

        log_format:
            Log message format.

        timestamp_format:
            Timestamp format.

        run_id_filter:
            Filter that injects run ID into log records.

        Returns
        -------
        logging.Handler
            Configured file handler.
        """
        log_directory = self.path_manager.get_log_path(log_path_key)
        self.path_manager.ensure_directory(log_directory)

        filename_prefix = destination_config.get("filename_prefix", destination_key)
        log_file_path = log_directory / f"{filename_prefix}_{self.run_id}.log"

        rotation_config = self.logging_config.get("rotation", {})

        if rotation_config.get("enabled", False):
            handler: logging.Handler = RotatingFileHandler(
                filename=log_file_path,
                maxBytes=int(rotation_config.get("max_bytes", 10485760)),
                backupCount=int(rotation_config.get("backup_count", 5)),
                encoding="utf-8",
            )
        else:
            handler = logging.FileHandler(
                filename=log_file_path,
                encoding="utf-8",
            )

        handler.setLevel(
            self._get_log_level(destination_config.get("level", "INFO"))
        )
        handler.setFormatter(
            logging.Formatter(log_format, datefmt=timestamp_format)
        )
        handler.addFilter(run_id_filter)

        return handler

    def _get_default_level(self) -> str:
        """
        Return default log level from configuration.

        Returns
        -------
        str
            Default log level.
        """
        return self.logging_config.get("logging", {}).get("default_level", "INFO")

    @staticmethod
    def _get_log_level(level_name: str) -> int:
        """
        Convert log level name to logging module level.

        Parameters
        ----------
        level_name:
            Log level name.

        Returns
        -------
        int
            Python logging level.

        Raises
        ------
        LoggingError
            Raised when the log level is invalid.
        """
        normalized_level = str(level_name).upper().strip()

        if not hasattr(logging, normalized_level):
            raise LoggingError(f"Invalid logging level: {level_name}")

        return int(getattr(logging, normalized_level))

    @staticmethod
    def _generate_run_id() -> str:
        """
        Generate a MedFabric run ID.

        Returns
        -------
        str
            Timestamp-based run identifier.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"{timestamp}_{suffix}"

    @staticmethod
    def _clear_existing_handlers(logger: logging.Logger) -> None:
        """
        Remove existing handlers from a logger.

        Parameters
        ----------
        logger:
            Logger whose handlers should be removed.
        """
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _require_initialized(self) -> None:
        """
        Ensure LoggingManager has been initialized.

        Raises
        ------
        LoggingError
            Raised when logging is used before initialize().
        """
        if not self.initialized:
            raise LoggingError(
                "LoggingManager has not been initialized. "
                "Call initialize() before logging events."
            )


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.logging_manager

    Expected Output
    ---------------
    Console messages confirming logging manager validation completed.
    Log files created in logs/pipeline, logs/modules, logs/errors, and logs/audit.
    """
    configuration_manager = ConfigurationManager()
    path_manager = PathManager(configuration_manager)
    logging_manager = LoggingManager(configuration_manager, path_manager)

    logging_manager.initialize()

    logger = logging_manager.get_logger("medfabric.logging_manager.test")

    logging_manager.start_step("LoggingManager standalone validation")

    logger.info("Testing standard info logging.")
    logger.warning("Testing warning logging.")

    logging_manager.log_dataset(
        dataset_name="Logging_Test_Dataset",
        row_count=10,
        column_count=3,
        path="data/raw/logging_test.csv",
    )

    logging_manager.log_validation(
        validation_name="Logging configuration validation",
        status="PASS",
        message="Logging configuration loaded successfully.",
    )

    logging_manager.end_step("LoggingManager standalone validation")

    print("MedFabric logging manager validation completed successfully.")
    print(f"Run ID: {logging_manager.get_run_id()}")

    logging_manager.close()


if __name__ == "__main__":
    main()