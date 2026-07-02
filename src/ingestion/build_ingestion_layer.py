###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/ingestion/build_bronze_layer.py
#
# Purpose:
#     Builds the MedFabric Bronze Layer using the Copy + Audit ingestion pattern.
#
# Business Context:
#     The Bronze Layer is the first persistent medallion layer after Raw data
#     generation. It creates auditable, replayable, traceable copies of Raw
#     healthcare datasets while preserving the original source business columns
#     and source values exactly as-is.
#
# Bronze Philosophy:
#     Bronze = Copy + Audit
#
#     Bronze DOES:
#         - Read Raw datasets
#         - Preserve source columns
#         - Preserve source values
#         - Add technical audit columns
#         - Validate row counts
#         - Validate primary keys
#         - Validate audit columns
#         - Generate metadata
#         - Generate lineage
#         - Write Bronze datasets
#
#     Bronze DOES NOT:
#         - Rename business columns
#         - Standardize business values
#         - Join datasets
#         - Deduplicate records
#         - Derive business columns
#         - Aggregate data
#         - Apply business rules
#
# Inputs:
#     config/ingestion/ingestion.yaml
#     data/raw/*.parquet
#
# Outputs:
#     data/bronze/*.parquet
#     data/metadata/bronze_*_dataset_metadata.json
#     data/metadata/bronze_*_column_metadata.csv
#     data/metadata/bronze_*_statistics.json
#     data/metadata/bronze_*_lineage.json
#
# Dependencies:
#     hashlib
#     json
#     pandas
#     src.common.pipeline_context.create_pipeline_context
#     src.common.exception_manager.PipelineError
#
# Architectural Rules:
#     1. Do not read YAML directly. Use context.configuration.
#     2. Do not read/write files directly. Use context.storage.
#     3. Do not create a new folder structure.
#     4. Use config/ingestion/ingestion.yaml as the Bronze source of truth.
#     5. Preserve Raw business columns exactly.
#     6. Add only Bronze audit columns.
#     7. Do not perform Silver transformations in Bronze.
#
# Run Command:
#     python -m src.ingestion.build_bronze_layer
#
# Expected Output:
#     Bronze datasets and metadata generated successfully.
###############################################################################

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.ingestion.bronze"
STEP_NAME = "Build Bronze Layer"
INGESTION_CONFIG_PATH = "ingestion/ingestion.yaml"


###############################################################################
# CONFIGURATION HELPERS
###############################################################################


def require_config_value(
    config: Dict[str, Any],
    key: str,
    config_name: str,
) -> Any:
    """
    Read a required configuration value.

    Parameters
    ----------
    config:
        Configuration dictionary being inspected.

    key:
        Required key name.

    config_name:
        Human-readable configuration name used in error messages.

    Returns
    -------
    Any
        Required configuration value.

    Raises
    ------
    PipelineError
        Raised when the required key is missing.
    """
    if key not in config:
        raise PipelineError(
            f"Missing required configuration value '{key}' in {config_name}."
        )

    return config[key]


def require_config_section(
    config: Dict[str, Any],
    key: str,
    config_name: str,
) -> Dict[str, Any]:
    """
    Read a required configuration section.

    Parameters
    ----------
    config:
        Configuration dictionary being inspected.

    key:
        Required section name.

    config_name:
        Human-readable configuration name used in error messages.

    Returns
    -------
    dict
        Required configuration section.

    Raises
    ------
    PipelineError
        Raised when the section is missing or is not a dictionary.
    """
    value = require_config_value(
        config=config,
        key=key,
        config_name=config_name,
    )

    if not isinstance(value, dict):
        raise PipelineError(
            f"Configuration section '{key}' in {config_name} must be a mapping."
        )

    return value


def load_bronze_config(context) -> Dict[str, Any]:
    """
    Load Bronze Layer configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/ingestion/ingestion.yaml.
    """
    return context.configuration.load_yaml(INGESTION_CONFIG_PATH)


###############################################################################
# DATASET SELECTION
###############################################################################


def get_enabled_dataset_configs(bronze_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return enabled Bronze dataset configurations.

    Parameters
    ----------
    bronze_config:
        Parsed Bronze configuration.

    Returns
    -------
    dict
        Mapping of dataset key to dataset configuration.

    Raises
    ------
    PipelineError
        Raised when datasets section is missing or invalid.
    """
    datasets_config = require_config_section(
        config=bronze_config,
        key="datasets",
        config_name="ingestion.yaml",
    )

    enabled_datasets: Dict[str, Dict[str, Any]] = {}

    for dataset_key, dataset_config in datasets_config.items():
        if not isinstance(dataset_config, dict):
            raise PipelineError(
                f"Dataset configuration for '{dataset_key}' must be a mapping."
            )

        if bool(dataset_config.get("enabled", True)):
            enabled_datasets[dataset_key] = dataset_config

    return enabled_datasets


###############################################################################
# RECORD HASHING
###############################################################################


def normalize_value_for_hash(value: Any) -> Any:
    """
    Normalize a value so it can be safely serialized for hashing.

    Parameters
    ----------
    value:
        Raw cell value.

    Returns
    -------
    Any
        JSON-serializable value.

    Processing Notes
    ----------------
    The Bronze record hash must represent the original Raw business record.
    Audit columns are not included in the hash.
    """
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def build_record_hash(row: pd.Series, source_columns: List[str]) -> str:
    """
    Build a SHA256 hash for one source record.

    Parameters
    ----------
    row:
        Source DataFrame row.

    source_columns:
        Original Raw business columns to include in the hash.

    Returns
    -------
    str
        SHA256 hash value.
    """
    record_payload = {
        column_name: normalize_value_for_hash(row[column_name])
        for column_name in source_columns
    }

    serialized_payload = json.dumps(
        record_payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def add_record_hash_column(
    dataframe: pd.DataFrame,
    record_hash_column: str,
    source_columns: List[str],
) -> pd.DataFrame:
    """
    Add Bronze record hash column.

    Parameters
    ----------
    dataframe:
        Bronze DataFrame being built.

    record_hash_column:
        Configured audit column name for record hash.

    source_columns:
        Original Raw business columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with record hash column added.
    """
    dataframe[record_hash_column] = dataframe.apply(
        lambda row: build_record_hash(row=row, source_columns=source_columns),
        axis=1,
    )

    return dataframe


###############################################################################
# BRONZE AUDIT COLUMNS
###############################################################################


def add_bronze_audit_columns(
    dataframe: pd.DataFrame,
    bronze_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    context,
) -> pd.DataFrame:
    """
    Add standardized Bronze audit columns.

    Parameters
    ----------
    dataframe:
        Raw DataFrame copy.

    bronze_config:
        Parsed Bronze configuration.

    dataset_config:
        Single dataset configuration from ingestion.yaml.

    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        DataFrame with Bronze audit columns appended.

    Processing Notes
    ----------------
    This function appends only technical metadata. It does not change any source
    business column or source business value.
    """
    audit_config = require_config_section(
        config=bronze_config,
        key="audit",
        config_name="ingestion.yaml",
    )

    configuration_config = require_config_section(
        config=bronze_config,
        key="configuration",
        config_name="ingestion.yaml",
    )

    source_columns = list(dataframe.columns)

    run_id_column = require_config_value(
        config=audit_config,
        key="bronze_run_id_column",
        config_name="bronze.audit",
    )

    load_timestamp_column = require_config_value(
        config=audit_config,
        key="bronze_load_timestamp_column",
        config_name="bronze.audit",
    )

    source_dataset_column = require_config_value(
        config=audit_config,
        key="bronze_source_dataset_column",
        config_name="bronze.audit",
    )

    pipeline_version_column = require_config_value(
        config=audit_config,
        key="bronze_pipeline_version_column",
        config_name="bronze.audit",
    )

    record_hash_column = require_config_value(
        config=audit_config,
        key="bronze_record_hash_column",
        config_name="bronze.audit",
    )

    record_status_column = require_config_value(
        config=audit_config,
        key="bronze_record_status_column",
        config_name="bronze.audit",
    )

    dataframe[run_id_column] = context.run_id
    dataframe[load_timestamp_column] = pd.Timestamp.utcnow()
    dataframe[source_dataset_column] = require_config_value(
        config=dataset_config,
        key="source_dataset_name",
        config_name="bronze.datasets",
    )
    dataframe[pipeline_version_column] = require_config_value(
        config=configuration_config,
        key="version",
        config_name="bronze.configuration",
    )
    dataframe[record_status_column] = require_config_value(
        config=audit_config,
        key="default_record_status",
        config_name="bronze.audit",
    )

    dataframe = add_record_hash_column(
        dataframe=dataframe,
        record_hash_column=record_hash_column,
        source_columns=source_columns,
    )

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def validate_raw_input(
    context,
    dataframe: pd.DataFrame,
    dataset_key: str,
    dataset_config: Dict[str, Any],
    bronze_config: Dict[str, Any],
) -> None:
    """
    Validate Raw input before Bronze processing.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Raw input DataFrame.

    dataset_key:
        Dataset key from ingestion.yaml.

    dataset_config:
        Dataset-specific configuration.

    bronze_config:
        Parsed Bronze configuration.
    """
    validation_config = require_config_section(
        config=bronze_config,
        key="validation",
        config_name="ingestion.yaml",
    )

    allow_empty = bool(validation_config.get("allow_empty", False))

    validation_rules: Dict[str, Any] = {
        "allow_empty": allow_empty,
    }

    if not allow_empty:
        validation_rules["min_rows"] = 1

    primary_key = dataset_config.get("primary_key", [])

    if primary_key:
        validation_rules["required_columns"] = primary_key
        validation_rules["no_nulls"] = primary_key
        validation_rules["primary_key"] = primary_key

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=validation_rules,
        dataset_name=f"raw.{dataset_key}",
    )


def validate_bronze_output(
    context,
    raw_dataframe: pd.DataFrame,
    bronze_dataframe: pd.DataFrame,
    dataset_key: str,
    dataset_config: Dict[str, Any],
    bronze_config: Dict[str, Any],
) -> None:
    """
    Validate Bronze output after audit columns are added.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    raw_dataframe:
        Original Raw input DataFrame.

    bronze_dataframe:
        Bronze output DataFrame.

    dataset_key:
        Dataset key from ingestion.yaml.

    dataset_config:
        Dataset-specific configuration.

    bronze_config:
        Parsed Bronze configuration.

    Raises
    ------
    PipelineError
        Raised when Bronze output fails row-count or audit validation.
    """
    validation_config = require_config_section(
        config=bronze_config,
        key="validation",
        config_name="ingestion.yaml",
    )

    audit_config = require_config_section(
        config=bronze_config,
        key="audit",
        config_name="ingestion.yaml",
    )

    if bool(validation_config.get("validate_row_count_match", True)):
        if len(raw_dataframe) != len(bronze_dataframe):
            raise PipelineError(
                f"Bronze row count mismatch for {dataset_key}. "
                f"Raw rows: {len(raw_dataframe)} | "
                f"Bronze rows: {len(bronze_dataframe)}"
            )

    required_columns: List[str] = []

    primary_key = dataset_config.get("primary_key", [])

    if primary_key:
        required_columns.extend(primary_key)

    if bool(validation_config.get("validate_audit_columns", True)):
        required_columns.extend(
            [
                require_config_value(audit_config, "bronze_run_id_column", "bronze.audit"),
                require_config_value(audit_config, "bronze_load_timestamp_column", "bronze.audit"),
                require_config_value(audit_config, "bronze_source_dataset_column", "bronze.audit"),
                require_config_value(audit_config, "bronze_pipeline_version_column", "bronze.audit"),
                require_config_value(audit_config, "bronze_record_hash_column", "bronze.audit"),
                require_config_value(audit_config, "bronze_record_status_column", "bronze.audit"),
            ]
        )

    no_null_columns = list(required_columns)

    validation_rules: Dict[str, Any] = {
        "allow_empty": bool(validation_config.get("allow_empty", False)),
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if not bool(validation_config.get("allow_empty", False)):
        validation_rules["min_rows"] = 1

    if primary_key:
        validation_rules["primary_key"] = primary_key

    context.validation.validate_dataset(
        dataframe=bronze_dataframe,
        validation_rules=validation_rules,
        dataset_name=require_config_value(
            config=dataset_config,
            key="dataset_name",
            config_name=f"bronze.datasets.{dataset_key}",
        ),
    )


###############################################################################
# METADATA
###############################################################################


def build_metadata_prefix(dataset_key: str) -> str:
    """
    Build metadata file prefix for a Bronze dataset.

    Parameters
    ----------
    dataset_key:
        Dataset key from ingestion.yaml.

    Returns
    -------
    str
        Metadata output prefix.
    """
    return f"bronze_{dataset_key}"


def write_bronze_metadata(
    context,
    bronze_dataframe: pd.DataFrame,
    dataset_key: str,
    dataset_config: Dict[str, Any],
    bronze_config: Dict[str, Any],
    written_output_path: str,
) -> None:
    """
    Write Bronze dataset metadata, column metadata, statistics, and lineage.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    bronze_dataframe:
        Bronze output DataFrame.

    dataset_key:
        Dataset key from ingestion.yaml.

    dataset_config:
        Dataset-specific configuration.

    bronze_config:
        Parsed Bronze configuration.

    written_output_path:
        Written Bronze dataset path.
    """
    metadata_config = require_config_section(
        config=bronze_config,
        key="metadata",
        config_name="ingestion.yaml",
    )

    dataset_name = require_config_value(
        config=dataset_config,
        key="dataset_name",
        config_name=f"bronze.datasets.{dataset_key}",
    )

    source_dataset_name = require_config_value(
        config=dataset_config,
        key="source_dataset_name",
        config_name=f"bronze.datasets.{dataset_key}",
    )

    primary_key = dataset_config.get("primary_key", [])
    metadata_prefix = build_metadata_prefix(dataset_key)

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name=dataset_name,
            dataframe=bronze_dataframe,
            output_path=written_output_path,
            layer="bronze",
            domain=dataset_key,
            primary_key=primary_key,
            description=f"Bronze Copy + Audit dataset for {source_dataset_name}.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path=f"data/metadata/{metadata_prefix}_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name=dataset_name,
            dataframe=bronze_dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path=f"data/metadata/{metadata_prefix}_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name=dataset_name,
            dataframe=bronze_dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path=f"data/metadata/{metadata_prefix}_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name=dataset_name,
            source_datasets=[source_dataset_name],
            output_dataset=dataset_name,
            transformation_name="copy_plus_audit",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path=f"data/metadata/{metadata_prefix}_lineage.json",
        )


###############################################################################
# BRONZE DATASET PROCESSING
###############################################################################


def build_single_bronze_dataset(
    context,
    bronze_config: Dict[str, Any],
    dataset_key: str,
    dataset_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build one Bronze dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    bronze_config:
        Parsed Bronze configuration.

    dataset_key:
        Dataset key from ingestion.yaml.

    dataset_config:
        Dataset-specific configuration.

    Returns
    -------
    pandas.DataFrame
        Bronze DataFrame.

    Processing Notes
    ----------------
    This function implements Copy + Audit for one dataset:
        1. Read Raw input.
        2. Validate Raw input.
        3. Copy DataFrame.
        4. Add audit columns.
        5. Validate Bronze output.
        6. Write Bronze output.
        7. Write metadata.
    """
    logger = context.get_logger(MODULE_NAME)

    input_path = require_config_value(
        config=dataset_config,
        key="input_path",
        config_name=f"bronze.datasets.{dataset_key}",
    )

    output_path = require_config_value(
        config=dataset_config,
        key="output_path",
        config_name=f"bronze.datasets.{dataset_key}",
    )

    dataset_name = require_config_value(
        config=dataset_config,
        key="dataset_name",
        config_name=f"bronze.datasets.{dataset_key}",
    )

    logger.info(
        "START Bronze dataset: %s | Input: %s | Output: %s",
        dataset_name,
        input_path,
        output_path,
    )

    raw_dataframe = context.storage.read_parquet(input_path)

    validate_raw_input(
        context=context,
        dataframe=raw_dataframe,
        dataset_key=dataset_key,
        dataset_config=dataset_config,
        bronze_config=bronze_config,
    )

    bronze_dataframe = raw_dataframe.copy(deep=True)

    audit_config = require_config_section(
        config=bronze_config,
        key="audit",
        config_name="ingestion.yaml",
    )

    if bool(audit_config.get("add_audit_columns", True)):
        bronze_dataframe = add_bronze_audit_columns(
            dataframe=bronze_dataframe,
            bronze_config=bronze_config,
            dataset_config=dataset_config,
            context=context,
        )

    validate_bronze_output(
        context=context,
        raw_dataframe=raw_dataframe,
        bronze_dataframe=bronze_dataframe,
        dataset_key=dataset_key,
        dataset_config=dataset_config,
        bronze_config=bronze_config,
    )

    written_output_path = context.storage.write_parquet(
        dataframe=bronze_dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name=dataset_name,
        row_count=len(bronze_dataframe),
        column_count=len(bronze_dataframe.columns),
        path=written_output_path,
    )

    write_bronze_metadata(
        context=context,
        bronze_dataframe=bronze_dataframe,
        dataset_key=dataset_key,
        dataset_config=dataset_config,
        bronze_config=bronze_config,
        written_output_path=str(written_output_path),
    )

    logger.info(
        "COMPLETE Bronze dataset: %s | Rows: %s | Columns: %s",
        dataset_name,
        len(bronze_dataframe),
        len(bronze_dataframe.columns),
    )

    return bronze_dataframe


###############################################################################
# ORCHESTRATION
###############################################################################


def build_bronze_layer(context) -> Dict[str, pd.DataFrame]:
    """
    Build all enabled Bronze datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Mapping of dataset key to generated Bronze DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    bronze_config = load_bronze_config(context)

    configuration_config = require_config_section(
        config=bronze_config,
        key="configuration",
        config_name="ingestion.yaml",
    )

    if not bool(configuration_config.get("enabled", True)):
        logger.warning("Ingestion Layer is disabled in config/ingestion/ingestion.yaml.")
        return {}

    enabled_dataset_configs = get_enabled_dataset_configs(bronze_config)

    logger.info(
        "Bronze Layer enabled datasets: %s",
        ", ".join(enabled_dataset_configs.keys()),
    )

    bronze_outputs: Dict[str, pd.DataFrame] = {}

    execution_config = require_config_section(
        config=bronze_config,
        key="execution",
        config_name="ingestion.yaml",
    )

    fail_fast = bool(execution_config.get("fail_fast", True))

    for dataset_key, dataset_config in enabled_dataset_configs.items():
        try:
            bronze_outputs[dataset_key] = build_single_bronze_dataset(
                context=context,
                bronze_config=bronze_config,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
            )

        except Exception as error:
            logger.exception(
                "Bronze dataset failed: %s | Error: %s",
                dataset_key,
                error,
            )

            if fail_fast:
                raise

    return bronze_outputs


def main() -> None:
    """
    Main entry point for Bronze Layer build.

    Run Command
    -----------
    python -m src.ingestion.build_bronze_layer
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        bronze_outputs = build_bronze_layer(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Bronze Layer completed successfully. Datasets built: %s",
            len(bronze_outputs),
        )

        print("MedFabric Bronze Layer completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Bronze Layer build failed.")
        logger.exception("Bronze Layer build failed.")
        raise PipelineError("Bronze Layer build failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()