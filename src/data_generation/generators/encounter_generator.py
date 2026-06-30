###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/encounter_generator.py
#
# Purpose:
#     Generates the canonical Encounter dataset used by the MedFabric Synthetic
#     Data Engine.
#
# Business Context:
#     Encounters represent healthcare service events between members and the
#     healthcare delivery system. They connect members, providers, facilities,
#     enrollment, encounter types, place of service, and service dates.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/encounters.yaml
#     data/raw/members.parquet
#     data/raw/providers.parquet
#     data/raw/facilities.parquet
#     data/raw/enrollment.parquet
#     reference/terminology/encounter_type_reference.parquet
#     reference/terminology/place_of_service_reference.parquet
#
# Outputs:
#     data/raw/encounters.parquet
#     data/metadata/encounters_dataset_metadata.json
#     data/metadata/encounters_column_metadata.csv
#     data/metadata/encounters_statistics.json
#     data/metadata/encounters_lineage.json
#
# Dependencies:
#     numpy
#     pandas
#     src.common.pipeline_context.create_pipeline_context
#     src.common.exception_manager.PipelineError
#
# Architectural Rules:
#     1. Do not read YAML directly. Use context.configuration.
#     2. Do not read/write files directly. Use context.storage.
#     3. Do not duplicate validation framework logic. Use context.validation.
#     4. Do not duplicate metadata framework logic. Use context.metadata.
#     5. Do not place configurable business values in Python.
#     6. Dataset-specific settings come from config/data_generation/encounters.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.encounter_generator
#
# Expected Output:
#     Canonical Encounter dataset and metadata written to configured output paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.encounter"
STEP_NAME = "Generate Encounter Dataset"


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
    value = require_config_value(config, key, config_name)

    if not isinstance(value, dict):
        raise PipelineError(
            f"Configuration section '{key}' in {config_name} must be a mapping."
        )

    return value


def load_global_generation_config(context) -> Dict[str, Any]:
    """
    Load platform-wide synthetic data generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/generation.yaml.
    """
    return context.configuration.load_yaml("data_generation/generation.yaml")


def load_encounter_generation_config(context) -> Dict[str, Any]:
    """
    Load Encounter-specific synthetic data generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/encounters.yaml.
    """
    return context.configuration.load_yaml("data_generation/encounters.yaml")


###############################################################################
# SAMPLING HELPERS
###############################################################################


def normalize_weights(weights: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Normalize numeric weights into probabilities.

    Parameters
    ----------
    weights:
        Numeric weight array.

    dataset_name:
        Dataset or rule name used in error messages.

    Returns
    -------
    numpy.ndarray
        Probability array that sums to 1.

    Raises
    ------
    PipelineError
        Raised when weights do not sum to a positive value.
    """
    total_weight = float(weights.sum())

    if total_weight <= 0:
        raise PipelineError(f"Weights must sum to a positive value for {dataset_name}.")

    return weights / total_weight


def sample_dataframe_row(
    rng: np.random.Generator,
    dataframe: pd.DataFrame,
    dataset_name: str,
    weight_column: str = "selection_weight",
) -> Dict[str, Any]:
    """
    Sample one row from a DataFrame.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    dataframe:
        Source DataFrame.

    dataset_name:
        Logical dataset name used in errors.

    weight_column:
        Optional weight column. If present, weighted sampling is used.

    Returns
    -------
    dict
        Sampled row as a dictionary.

    Raises
    ------
    PipelineError
        Raised when the input DataFrame is empty.
    """
    if dataframe.empty:
        raise PipelineError(f"Dataset is empty: {dataset_name}")

    if weight_column in dataframe.columns:
        weights = dataframe[weight_column].astype(float).to_numpy()
        probabilities = normalize_weights(weights, dataset_name)
        selected_index = rng.choice(dataframe.index.to_numpy(), p=probabilities)

        return dataframe.loc[selected_index].to_dict()

    selected_position = int(rng.integers(0, len(dataframe)))

    return dataframe.iloc[selected_position].to_dict()


###############################################################################
# INPUT AND REFERENCE LOADING
###############################################################################


def load_input_and_reference_data(
    context,
    encounter_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load raw inputs and encounter reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    dict
        Dictionary of loaded input and reference DataFrames.

    Processing Notes
    ----------------
    Encounters are generated from core master datasets:
        - Members
        - Providers
        - Facilities
        - Enrollment

    Encounter type and place of service come from terminology reference data.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(
        encounter_config,
        "inputs",
        "encounters.yaml",
    )

    encounter_type_config = require_config_section(
        encounter_config,
        "encounter_types",
        "encounters.yaml",
    )

    place_of_service_config = require_config_section(
        encounter_config,
        "place_of_service",
        "encounters.yaml",
    )

    paths = {
        "members": require_config_value(
            inputs_config,
            "members_dataset",
            "encounters.inputs",
        ),
        "providers": require_config_value(
            inputs_config,
            "providers_dataset",
            "encounters.inputs",
        ),
        "facilities": require_config_value(
            inputs_config,
            "facilities_dataset",
            "encounters.inputs",
        ),
        "enrollment": require_config_value(
            inputs_config,
            "enrollment_dataset",
            "encounters.inputs",
        ),
        "encounter_types": require_config_value(
            encounter_type_config,
            "source",
            "encounters.encounter_types",
        ),
        "place_of_service": require_config_value(
            place_of_service_config,
            "source",
            "encounters.place_of_service",
        ),
    }

    datasets: Dict[str, pd.DataFrame] = {}

    for dataset_name, path in paths.items():
        dataframe = context.storage.read_parquet(path)

        context.validation.validate_dataset(
            dataframe=dataframe,
            validation_rules={
                "allow_empty": False,
                "min_rows": 1,
            },
            dataset_name=dataset_name,
        )

        datasets[dataset_name] = dataframe

        logger.info(
            "Loaded Encounter input/reference: %s | Rows: %s | Path: %s",
            dataset_name,
            len(dataframe),
            path,
        )

    return datasets


###############################################################################
# IDENTIFIERS, DATES, AND VOLUME
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    encounter_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for reproducible Encounter generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        encounter_config,
        "reproducibility",
        "encounters.yaml",
    )

    if bool(reproducibility_config.get("use_global_random_seed", True)):
        execution_config = require_config_section(
            global_config,
            "execution",
            "generation.yaml",
        )

        return int(
            require_config_value(
                execution_config,
                "random_seed",
                "generation.execution",
            )
        )

    return int(
        require_config_value(
            reproducibility_config,
            "random_seed",
            "encounters.reproducibility",
        )
    )


def build_encounter_id(
    encounter_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build configured Encounter identifier.

    Parameters
    ----------
    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    sequence_number:
        Zero-based encounter sequence number.

    Returns
    -------
    str
        Generated encounter identifier.
    """
    identifier_config = require_config_section(
        encounter_config,
        "encounter_identifier",
        "encounters.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "encounters.encounter_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "encounters.encounter_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "encounters.encounter_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def sample_encounter_count(
    rng: np.random.Generator,
    encounter_config: Dict[str, Any],
) -> int:
    """
    Sample number of encounters for one member.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    int
        Encounter count for one member.
    """
    volume_config = require_config_section(
        encounter_config,
        "encounter_volume",
        "encounters.yaml",
    )

    average_encounters = float(
        require_config_value(
            volume_config,
            "average_encounters_per_member_per_year",
            "encounters.encounter_volume",
        )
    )

    minimum_encounters = int(
        require_config_value(
            volume_config,
            "minimum_encounters_per_member",
            "encounters.encounter_volume",
        )
    )

    maximum_encounters = int(
        require_config_value(
            volume_config,
            "maximum_encounters_per_member",
            "encounters.encounter_volume",
        )
    )

    sampled_count = int(rng.poisson(average_encounters))

    return max(minimum_encounters, min(maximum_encounters, sampled_count))


def sample_encounter_date(
    rng: np.random.Generator,
    encounter_config: Dict[str, Any],
) -> pd.Timestamp:
    """
    Generate synthetic encounter date.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    pandas.Timestamp
        Generated encounter date.
    """
    dates_config = require_config_section(encounter_config, "dates", "encounters.yaml")

    minimum_encounter_year = int(
        require_config_value(
            dates_config,
            "minimum_encounter_year",
            "encounters.dates",
        )
    )

    maximum_encounter_year = int(
        require_config_value(
            dates_config,
            "maximum_encounter_year",
            "encounters.dates",
        )
    )

    if minimum_encounter_year > maximum_encounter_year:
        raise PipelineError(
            "minimum_encounter_year cannot be greater than maximum_encounter_year."
        )

    encounter_year = int(rng.integers(minimum_encounter_year, maximum_encounter_year + 1))
    encounter_month = int(rng.integers(1, 13))
    encounter_day = int(rng.integers(1, 29))

    return pd.Timestamp(
        year=encounter_year,
        month=encounter_month,
        day=encounter_day,
    )


###############################################################################
# ENCOUNTER RECORD GENERATION
###############################################################################


def build_encounter_record(
    rng: np.random.Generator,
    encounter_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    member_row: Dict[str, Any],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical Encounter record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    datasets:
        Loaded input and reference datasets.

    member_row:
        Member row dictionary.

    sequence_number:
        Zero-based encounter sequence number.

    Returns
    -------
    dict
        Generated Encounter record.
    """
    audit_config = require_config_section(
        encounter_config,
        "audit",
        "encounters.yaml",
    )

    provider_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["providers"],
        dataset_name="raw.providers",
    )

    facility_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["facilities"],
        dataset_name="raw.facilities",
    )

    enrollment_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["enrollment"],
        dataset_name="raw.enrollment",
    )

    encounter_type_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["encounter_types"],
        dataset_name="reference.encounter_types",
    )

    place_of_service_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["place_of_service"],
        dataset_name="reference.place_of_service",
    )

    encounter_date = sample_encounter_date(
        rng=rng,
        encounter_config=encounter_config,
    )

    return {
        "encounter_id": build_encounter_id(
            encounter_config=encounter_config,
            sequence_number=sequence_number,
        ),
        "member_id": member_row.get("member_id"),
        "enrollment_id": enrollment_row.get("enrollment_id"),
        "provider_id": provider_row.get("provider_id"),
        "facility_id": facility_row.get("facility_id"),
        "encounter_type": encounter_type_row.get("encounter_type"),
        "place_of_service_code": place_of_service_row.get("code"),
        "place_of_service_description": place_of_service_row.get("description"),
        "encounter_date": encounter_date.date(),
        "encounter_year": int(encounter_date.year),
        "encounter_month": int(encounter_date.month),
        "source_system": require_config_value(
            audit_config,
            "source_system",
            "encounters.audit",
        ),
        "record_status": require_config_value(
            audit_config,
            "record_status",
            "encounters.audit",
        ),
        "created_at": pd.Timestamp(
            require_config_value(
                audit_config,
                "created_at",
                "encounters.audit",
            )
        ),
        "updated_at": pd.Timestamp(
            require_config_value(
                audit_config,
                "updated_at",
                "encounters.audit",
            )
        ),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_encounter_dataset(
    context,
    global_config: Dict[str, Any],
    encounter_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate canonical Encounter dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    datasets:
        Loaded input and reference datasets.

    Returns
    -------
    pandas.DataFrame
        Generated Encounter DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    random_seed = resolve_random_seed(
        global_config=global_config,
        encounter_config=encounter_config,
    )

    rng = np.random.default_rng(random_seed)
    members_dataframe = datasets["members"]

    logger.info(
        "Generating Encounters from Member dataset. Member rows: %s",
        len(members_dataframe),
    )

    records: List[Dict[str, Any]] = []
    sequence_number = 0

    for member_row in members_dataframe.to_dict("records"):
        encounter_count = sample_encounter_count(
            rng=rng,
            encounter_config=encounter_config,
        )

        for _ in range(encounter_count):
            records.append(
                build_encounter_record(
                    rng=rng,
                    encounter_config=encounter_config,
                    datasets=datasets,
                    member_row=member_row,
                    sequence_number=sequence_number,
                )
            )

            sequence_number += 1

    dataframe = pd.DataFrame(records)

    logger.info(
        "Generated Encounter dataset. Rows: %s",
        len(dataframe),
    )

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(encounter_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible rules for Encounters.

    Parameters
    ----------
    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        encounter_config,
        "validation",
        "encounters.yaml",
    )

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_unique_encounter_ids", False)):
        required_columns.append("encounter_id")
        no_null_columns.append("encounter_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_provider_id", False)):
        required_columns.append("provider_id")
        no_null_columns.append("provider_id")

    if bool(validation_config.get("require_facility_id", False)):
        required_columns.append("facility_id")
        no_null_columns.append("facility_id")

    if bool(validation_config.get("require_encounter_type", False)):
        required_columns.append("encounter_type")
        no_null_columns.append("encounter_type")

    if bool(validation_config.get("require_encounter_date", False)):
        required_columns.append("encounter_date")
        no_null_columns.append("encounter_date")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_encounter_ids", False)):
        validation_rules["primary_key"] = ["encounter_id"]

    return validation_rules


def validate_encounter_dataset(
    context,
    dataframe: pd.DataFrame,
    encounter_config: Dict[str, Any],
) -> None:
    """
    Validate generated Encounter dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Encounter DataFrame.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.
    """
    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=build_validation_rules(encounter_config),
        dataset_name="raw.encounters",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_output_path(encounter_config: Dict[str, Any]) -> str:
    """
    Resolve Encounter output path.

    Parameters
    ----------
    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(
        encounter_config,
        "output",
        "encounters.yaml",
    )

    file_name = require_config_value(
        output_config,
        "file_name",
        "encounters.output",
    )

    return f"data/raw/{file_name}"


def write_encounter_dataset(
    context,
    dataframe: pd.DataFrame,
    encounter_config: Dict[str, Any],
) -> str:
    """
    Write Encounter dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Encounter DataFrame.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    Returns
    -------
    str
        Written output path.
    """
    output_path = resolve_output_path(encounter_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.encounters",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_encounter_metadata(
    context,
    dataframe: pd.DataFrame,
    encounter_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Encounter metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Encounter DataFrame.

    encounter_config:
        Parsed config/data_generation/encounters.yaml.

    output_path:
        Written raw output path.
    """
    metadata_config = require_config_section(
        encounter_config,
        "metadata",
        "encounters.yaml",
    )

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.encounters",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="encounter",
            primary_key=["encounter_id"],
            description="Canonical MedFabric synthetic Encounter dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/encounters_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.encounters",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/encounters_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.encounters",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/encounters_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.encounters",
            source_datasets=[
                "raw.members",
                "raw.providers",
                "raw.facilities",
                "raw.enrollment",
                "reference.encounter_types",
                "reference.place_of_service",
            ],
            output_dataset="raw.encounters",
            transformation_name="generate_encounter_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/encounters_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_encounter_generation(context) -> pd.DataFrame:
    """
    Execute the complete Encounter generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Encounter dataset.
    """
    global_config = load_global_generation_config(context)
    encounter_config = load_encounter_generation_config(context)

    datasets = load_input_and_reference_data(
        context=context,
        encounter_config=encounter_config,
    )

    encounter_dataframe = generate_encounter_dataset(
        context=context,
        global_config=global_config,
        encounter_config=encounter_config,
        datasets=datasets,
    )

    validate_encounter_dataset(
        context=context,
        dataframe=encounter_dataframe,
        encounter_config=encounter_config,
    )

    output_path = write_encounter_dataset(
        context=context,
        dataframe=encounter_dataframe,
        encounter_config=encounter_config,
    )

    write_encounter_metadata(
        context=context,
        dataframe=encounter_dataframe,
        encounter_config=encounter_config,
        output_path=output_path,
    )

    return encounter_dataframe


def main() -> None:
    """
    Main entry point for Encounter generation.

    Run Command
    -----------
    python -m src.data_generation.generators.encounter_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        encounter_dataframe = run_encounter_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Encounter generation completed successfully. Rows: %s",
            len(encounter_dataframe),
        )

        print("MedFabric encounter generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Encounter generation failed.")
        logger.exception("Encounter generation failed.")
        raise PipelineError("Encounter generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()