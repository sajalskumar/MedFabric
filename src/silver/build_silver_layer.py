###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/silver/build_silver_layer.py
#
# Purpose:
#     Builds the MedFabric Silver Layer from Bronze datasets.
#
# Silver Philosophy:
#     Silver = Conform + Standardize
#
# Inputs:
#     config/silver/silver.yaml
#     data/bronze/*.parquet
#     reference/terminology/*.parquet
#
# Outputs:
#     data/silver/*.parquet
#     data/metadata/silver_*_dataset_metadata.json
#     data/metadata/silver_*_column_metadata.csv
#     data/metadata/silver_*_statistics.json
#     data/metadata/silver_*_lineage.json
#
# Run Command:
#     python -m src.silver.build_silver_layer
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.silver"
STEP_NAME = "Build Silver Layer"
SILVER_CONFIG_PATH = "silver/silver.yaml"


def require_config_value(config: Dict[str, Any], key: str, config_name: str) -> Any:
    """Read a required config value."""
    if key not in config:
        raise PipelineError(f"Missing required configuration value '{key}' in {config_name}.")
    return config[key]


def require_config_section(config: Dict[str, Any], key: str, config_name: str) -> Dict[str, Any]:
    """Read a required config section."""
    value = require_config_value(config, key, config_name)
    if not isinstance(value, dict):
        raise PipelineError(f"Configuration section '{key}' in {config_name} must be a mapping.")
    return value


def load_silver_config(context) -> Dict[str, Any]:
    """Load config/silver/silver.yaml."""
    return context.configuration.load_yaml(SILVER_CONFIG_PATH)


def add_silver_audit_columns(
    context,
    dataframe: pd.DataFrame,
    silver_config: Dict[str, Any],
) -> pd.DataFrame:
    """Add Silver audit columns."""
    audit_config = require_config_section(silver_config, "audit", "silver.yaml")

    dataframe[
        require_config_value(audit_config, "silver_run_id_column", "silver.audit")
    ] = context.run_id

    dataframe[
        require_config_value(audit_config, "silver_load_timestamp_column", "silver.audit")
    ] = pd.Timestamp.utcnow()

    dataframe[
        require_config_value(audit_config, "silver_record_status_column", "silver.audit")
    ] = require_config_value(audit_config, "default_record_status", "silver.audit")

    return dataframe


def standardize_basic_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply safe Silver standardization.

    This does not create analytics. It only performs basic enterprise hygiene:
    - trims string values
    - converts empty strings to null
    """
    for column_name in dataframe.columns:
        if dataframe[column_name].dtype == "object":
            dataframe[column_name] = dataframe[column_name].astype("string").str.strip()
            dataframe[column_name] = dataframe[column_name].replace("", pd.NA)

    return dataframe


def validate_silver_dataset(
    context,
    dataframe: pd.DataFrame,
    dataset_name: str,
    primary_key: List[str],
    silver_config: Dict[str, Any],
) -> None:
    """Validate a Silver dataset."""
    validation_config = require_config_section(silver_config, "validation", "silver.yaml")

    validation_rules: Dict[str, Any] = {
        "allow_empty": bool(validation_config.get("allow_empty", False)),
        "required_columns": primary_key,
        "no_nulls": primary_key,
    }

    if not bool(validation_config.get("allow_empty", False)):
        validation_rules["min_rows"] = 1

    if primary_key:
        validation_rules["primary_key"] = primary_key

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=validation_rules,
        dataset_name=dataset_name,
    )


def write_silver_metadata(
    context,
    dataframe: pd.DataFrame,
    dataset_key: str,
    dataset_name: str,
    output_path: str,
    primary_key: List[str],
    source_datasets: List[str],
    silver_config: Dict[str, Any],
) -> None:
    """Write Silver metadata, statistics, and lineage."""
    metadata_config = require_config_section(silver_config, "metadata", "silver.yaml")
    metadata_prefix = f"silver_{dataset_key}"

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        metadata = context.metadata.build_dataset_metadata(
            dataset_name=dataset_name,
            dataframe=dataframe,
            output_path=output_path,
            layer="silver",
            domain=dataset_key,
            primary_key=primary_key,
            description=f"Silver standardized dataset for {dataset_name}.",
        )
        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        metadata = context.metadata.build_column_metadata(
            dataset_name=dataset_name,
            dataframe=dataframe,
        )
        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        metadata = context.metadata.build_statistics(
            dataset_name=dataset_name,
            dataframe=dataframe,
        )
        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name=dataset_name,
            source_datasets=source_datasets,
            output_dataset=dataset_name,
            transformation_name="silver_standardize_and_conform",
            module_name=MODULE_NAME,
        )
        context.metadata.write_metadata(
            metadata=lineage,
            output_path=f"data/metadata/{metadata_prefix}_lineage.json",
        )


def build_standard_silver_dataset(
    context,
    silver_config: Dict[str, Any],
    dataset_key: str,
    dataset_config: Dict[str, Any],
    source_dataset_name: str,
) -> pd.DataFrame:
    """Build a standard Silver dataset from one Bronze input."""
    logger = context.get_logger(MODULE_NAME)

    input_path = require_config_value(dataset_config, "input_path", f"silver.{dataset_key}")
    output_path = require_config_value(dataset_config, "output_path", f"silver.{dataset_key}")
    dataset_name = require_config_value(dataset_config, "dataset_name", f"silver.{dataset_key}")
    primary_key = dataset_config.get("primary_key", [])

    logger.info("START Silver dataset: %s", dataset_name)

    dataframe = context.storage.read_parquet(input_path)
    dataframe = dataframe.copy(deep=True)

    dataframe = standardize_basic_values(dataframe)
    dataframe = add_silver_audit_columns(context, dataframe, silver_config)

    validate_silver_dataset(
        context=context,
        dataframe=dataframe,
        dataset_name=dataset_name,
        primary_key=primary_key,
        silver_config=silver_config,
    )

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name=dataset_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    write_silver_metadata(
        context=context,
        dataframe=dataframe,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        output_path=str(written_path),
        primary_key=primary_key,
        source_datasets=[source_dataset_name],
        silver_config=silver_config,
    )

    logger.info("COMPLETE Silver dataset: %s | Rows: %s", dataset_name, len(dataframe))

    return dataframe


def build_dim_date(
    context,
    silver_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> pd.DataFrame:
    """Build enterprise Date dimension."""
    dataset_key = "dim_date"
    dataset_name = require_config_value(dataset_config, "dataset_name", "silver.dim_date")
    output_path = require_config_value(dataset_config, "output_path", "silver.dim_date")
    primary_key = dataset_config.get("primary_key", ["date_key"])

    start_date = pd.Timestamp(require_config_value(dataset_config, "start_date", "silver.dim_date"))
    end_date = pd.Timestamp(require_config_value(dataset_config, "end_date", "silver.dim_date"))

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    dataframe = pd.DataFrame({"date": date_range})
    dataframe["date_key"] = dataframe["date"].dt.strftime("%Y%m%d").astype(int)
    dataframe["year"] = dataframe["date"].dt.year
    dataframe["quarter"] = dataframe["date"].dt.quarter
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["month_name"] = dataframe["date"].dt.month_name()
    dataframe["day"] = dataframe["date"].dt.day
    dataframe["day_of_week"] = dataframe["date"].dt.dayofweek + 1
    dataframe["day_name"] = dataframe["date"].dt.day_name()
    dataframe["is_weekend"] = dataframe["day_of_week"].isin([6, 7])
    dataframe["date"] = dataframe["date"].dt.date

    dataframe = add_silver_audit_columns(context, dataframe, silver_config)

    validate_silver_dataset(
        context=context,
        dataframe=dataframe,
        dataset_name=dataset_name,
        primary_key=primary_key,
        silver_config=silver_config,
    )

    written_path = context.storage.write_parquet(dataframe=dataframe, path=output_path, index=False)

    context.logging.log_dataset(
        dataset_name=dataset_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    write_silver_metadata(
        context=context,
        dataframe=dataframe,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        output_path=str(written_path),
        primary_key=primary_key,
        source_datasets=[],
        silver_config=silver_config,
    )

    return dataframe


def build_dim_clinical_terminology(
    context,
    silver_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> pd.DataFrame:
    """Build enterprise Clinical Terminology dimension."""
    dataset_key = "dim_clinical_terminology"
    dataset_name = require_config_value(
        dataset_config,
        "dataset_name",
        "silver.dim_clinical_terminology",
    )
    output_path = require_config_value(
        dataset_config,
        "output_path",
        "silver.dim_clinical_terminology",
    )
    primary_key = dataset_config.get("primary_key", ["terminology_key"])
    reference_inputs = require_config_section(
        dataset_config,
        "reference_inputs",
        "silver.dim_clinical_terminology",
    )

    records: List[Dict[str, Any]] = []

    for code_system, path in reference_inputs.items():
        reference_dataframe = context.storage.read_parquet(path)

        for row in reference_dataframe.to_dict("records"):
            code_value = row.get("code") or row.get("rxnorm_code") or row.get("loinc_code")
            description = row.get("description") or row.get("drug_name") or row.get("test_name")

            records.append(
                {
                    "terminology_key": f"{str(code_system).upper()}::{code_value}",
                    "code_system": str(code_system).upper(),
                    "code": code_value,
                    "description": description,
                    "condition_group": row.get("condition_group"),
                    "source_reference": path,
                }
            )

    dataframe = pd.DataFrame(records)
    dataframe = standardize_basic_values(dataframe)
    dataframe = add_silver_audit_columns(context, dataframe, silver_config)

    validate_silver_dataset(
        context=context,
        dataframe=dataframe,
        dataset_name=dataset_name,
        primary_key=primary_key,
        silver_config=silver_config,
    )

    written_path = context.storage.write_parquet(dataframe=dataframe, path=output_path, index=False)

    context.logging.log_dataset(
        dataset_name=dataset_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    write_silver_metadata(
        context=context,
        dataframe=dataframe,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        output_path=str(written_path),
        primary_key=primary_key,
        source_datasets=list(reference_inputs.values()),
        silver_config=silver_config,
    )

    return dataframe


def build_silver_dimensions(context, silver_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Build all enabled Silver dimensions."""
    dimensions_config = require_config_section(silver_config, "dimensions", "silver.yaml")
    outputs: Dict[str, pd.DataFrame] = {}

    for dataset_key, dataset_config in dimensions_config.items():
        if not bool(dataset_config.get("enabled", True)):
            continue

        if dataset_key == "dim_date":
            outputs[dataset_key] = build_dim_date(context, silver_config, dataset_config)

        elif dataset_key == "dim_clinical_terminology":
            outputs[dataset_key] = build_dim_clinical_terminology(
                context,
                silver_config,
                dataset_config,
            )

        else:
            outputs[dataset_key] = build_standard_silver_dataset(
                context=context,
                silver_config=silver_config,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                source_dataset_name=f"bronze.{dataset_key.replace('dim_', '')}",
            )

    return outputs


def build_silver_facts(context, silver_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Build all enabled Silver facts."""
    facts_config = require_config_section(silver_config, "facts", "silver.yaml")
    outputs: Dict[str, pd.DataFrame] = {}

    for dataset_key, dataset_config in facts_config.items():
        if not bool(dataset_config.get("enabled", True)):
            continue

        outputs[dataset_key] = build_standard_silver_dataset(
            context=context,
            silver_config=silver_config,
            dataset_key=dataset_key,
            dataset_config=dataset_config,
            source_dataset_name=f"bronze.{dataset_key.replace('fact_', '')}",
        )

    return outputs


def build_silver_layer(context) -> Dict[str, pd.DataFrame]:
    """Build the complete Silver Layer."""
    logger = context.get_logger(MODULE_NAME)

    silver_config = load_silver_config(context)
    configuration_config = require_config_section(silver_config, "configuration", "silver.yaml")

    if not bool(configuration_config.get("enabled", True)):
        logger.warning("Silver Layer is disabled in config/silver/silver.yaml.")
        return {}

    outputs: Dict[str, pd.DataFrame] = {}

    logger.info("START Silver dimensions.")
    outputs.update(build_silver_dimensions(context, silver_config))

    logger.info("START Silver facts.")
    outputs.update(build_silver_facts(context, silver_config))

    return outputs


def main() -> None:
    """Main entry point for Silver Layer build."""
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        outputs = build_silver_layer(context)

        context.logging.end_step(STEP_NAME)

        logger.info("MedFabric Silver Layer completed successfully. Datasets built: %s", len(outputs))

        print("MedFabric Silver Layer completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Silver Layer build failed.")
        logger.exception("Silver Layer build failed.")
        raise PipelineError("Silver Layer build failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()