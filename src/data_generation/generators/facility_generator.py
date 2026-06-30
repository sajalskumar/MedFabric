###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/facility_generator.py
#
# Purpose:
#     Generates the canonical Facility dataset used by the MedFabric Synthetic
#     Data Engine.
#
# Business Context:
#     Facilities represent healthcare service locations such as hospitals,
#     clinics, urgent care centers, laboratories, skilled nursing facilities,
#     ambulatory surgery centers, and other care delivery locations.
#
#     Facilities are foundational to:
#         - Encounters
#         - Claims
#         - Laboratory events
#         - Pharmacy events
#         - Provider affiliations
#         - Network analytics
#         - Utilization analytics
#         - Facility performance reporting
#         - Member 360 care location history
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/facilities.yaml
#     reference/facilities/facility_reference.parquet
#     reference/geography/us_geography_reference.parquet
#
# Outputs:
#     data/raw/facilities.parquet
#     data/metadata/facilities_dataset_metadata.json
#     data/metadata/facilities_column_metadata.csv
#     data/metadata/facilities_statistics.json
#     data/metadata/facilities_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/facilities.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.facility_generator
#
# Expected Output:
#     Canonical Facility dataset and metadata written to configured output paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.facility"
STEP_NAME = "Generate Facility Dataset"


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
        Raised when the required configuration key is missing.
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


def load_facility_generation_config(context) -> Dict[str, Any]:
    """
    Load Facility-specific synthetic data generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/facilities.yaml.
    """
    return context.configuration.load_yaml("data_generation/facilities.yaml")


###############################################################################
# REFERENCE DATA LOADING
###############################################################################


def load_reference_data(
    context,
    facility_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load Facility reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    facility_config:
        Parsed config/data_generation/facilities.yaml.

    Returns
    -------
    dict
        Dictionary containing loaded facility and geography reference datasets.

    Raises
    ------
    PipelineError
        Raised when required configuration sections or source paths are missing.

    Processing Notes
    ----------------
    This generator does not define facility types or geography values in Python.
    Those values come from the Reference Data layer.
    """
    logger = context.get_logger(MODULE_NAME)

    facility_reference_config = require_config_section(
        facility_config,
        "facility",
        "facilities.yaml",
    )

    geography_config = require_config_section(
        facility_config,
        "geography",
        "facilities.yaml",
    )

    reference_paths = {
        "facility": require_config_value(
            facility_reference_config,
            "source",
            "facilities.facility",
        ),
        "geography": require_config_value(
            geography_config,
            "source",
            "facilities.geography",
        ),
    }

    reference_data: Dict[str, pd.DataFrame] = {}

    for reference_name, path in reference_paths.items():
        dataframe = context.storage.read_parquet(path)

        context.validation.validate_dataset(
            dataframe=dataframe,
            validation_rules={
                "allow_empty": False,
                "min_rows": 1,
            },
            dataset_name=f"reference.{reference_name}",
        )

        reference_data[reference_name] = dataframe

        logger.info(
            "Loaded Facility reference data: %s | Rows: %s | Path: %s",
            reference_name,
            len(dataframe),
            path,
        )

    return reference_data


###############################################################################
# SAMPLING HELPERS
###############################################################################


def normalize_weights(weights: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Normalize numeric reference weights.

    Parameters
    ----------
    weights:
        Numeric weight array.

    dataset_name:
        Dataset name used in error messages.

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
    Sample one row from a reference DataFrame.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    dataframe:
        Source reference DataFrame.

    dataset_name:
        Logical dataset name used in validation errors.

    weight_column:
        Optional weight column. If present, weighted sampling is used. If not
        present, uniform sampling is used.

    Returns
    -------
    dict
        Sampled reference row.

    Raises
    ------
    PipelineError
        Raised when the reference DataFrame is empty or weights are invalid.
    """
    if dataframe.empty:
        raise PipelineError(f"Reference dataset is empty: {dataset_name}")

    if weight_column in dataframe.columns:
        weights = dataframe[weight_column].astype(float).to_numpy()
        probabilities = normalize_weights(weights, dataset_name)
        selected_index = rng.choice(dataframe.index.to_numpy(), p=probabilities)

        return dataframe.loc[selected_index].to_dict()

    selected_position = int(rng.integers(0, len(dataframe)))

    return dataframe.iloc[selected_position].to_dict()


###############################################################################
# FACILITY GENERATION HELPERS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    facility_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for reproducible Facility generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    facility_config:
        Parsed config/data_generation/facilities.yaml.

    Returns
    -------
    int
        Random seed.

    Raises
    ------
    PipelineError
        Raised when required seed configuration is missing.
    """
    reproducibility_config = require_config_section(
        facility_config,
        "reproducibility",
        "facilities.yaml",
    )

    use_global_random_seed = bool(
        require_config_value(
            reproducibility_config,
            "use_global_random_seed",
            "facilities.reproducibility",
        )
    )

    if use_global_random_seed:
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
            "facilities.reproducibility",
        )
    )


def build_facility_id(
    facility_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build a configured Facility identifier.

    Parameters
    ----------
    facility_config:
        Parsed config/data_generation/facilities.yaml.

    sequence_number:
        Zero-based sequence number.

    Returns
    -------
    str
        Generated Facility identifier.
    """
    identifier_config = require_config_section(
        facility_config,
        "facility_identifier",
        "facilities.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "facilities.facility_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "facilities.facility_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "facilities.facility_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def build_facility_name(
    facility_config: Dict[str, Any],
    facility_id: str,
    facility_reference_row: Dict[str, Any],
) -> str:
    """
    Build a Facility display name.

    Parameters
    ----------
    facility_config:
        Parsed config/data_generation/facilities.yaml.

    facility_id:
        Generated Facility identifier.

    facility_reference_row:
        Sampled facility reference row.

    Returns
    -------
    str
        Generated Facility display name.

    Processing Notes
    ----------------
    Facility name formatting is configurable. Add this to facilities.yaml:

        naming:
          facility_name_template: "{organization_prefix} {facility_type} {facility_id}"
          organization_prefix: "MedFabric"
    """
    naming_config = require_config_section(
        facility_config,
        "naming",
        "facilities.yaml",
    )

    template = require_config_value(
        naming_config,
        "facility_name_template",
        "facilities.naming",
    )

    organization_prefix = require_config_value(
        naming_config,
        "organization_prefix",
        "facilities.naming",
    )

    facility_type = (
        facility_reference_row.get("facility_type_description")
        or facility_reference_row.get("facility_type_code")
    )

    if facility_type is None:
        raise PipelineError(
            "Facility reference row must contain facility_type_description "
            "or facility_type_code."
        )

    return template.format(
        organization_prefix=organization_prefix,
        facility_type=facility_type,
        facility_id=facility_id,
    )


def build_phone_number(
    rng: np.random.Generator,
    enabled: bool,
) -> str | None:
    """
    Generate a synthetic phone number.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    enabled:
        Whether phone generation is enabled.

    Returns
    -------
    str or None
        Phone number when enabled, otherwise None.

    Notes
    -----
    The number is synthetic and generated only when enabled by configuration.
    """
    if not enabled:
        return None

    area_code = int(rng.integers(200, 1000))
    prefix = int(rng.integers(200, 1000))
    line_number = int(rng.integers(0, 10000))

    return f"{area_code}-{prefix}-{line_number:04d}"


def build_email_address(
    facility_config: Dict[str, Any],
    enabled: bool,
    facility_id: str,
) -> str | None:
    """
    Generate a synthetic Facility email address.

    Parameters
    ----------
    facility_config:
        Parsed config/data_generation/facilities.yaml.

    enabled:
        Whether email generation is enabled.

    facility_id:
        Generated Facility identifier.

    Returns
    -------
    str or None
        Email address when enabled, otherwise None.

    Raises
    ------
    PipelineError
        Raised when email generation is enabled but email_domain is missing.
    """
    if not enabled:
        return None

    contact_config = require_config_section(
        facility_config,
        "contact_information",
        "facilities.yaml",
    )

    email_domain = require_config_value(
        contact_config,
        "email_domain",
        "facilities.contact_information",
    )

    return f"facility.{str(facility_id).lower()}@{email_domain}"


def build_facility_record(
    rng: np.random.Generator,
    facility_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical Facility record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    facility_config:
        Parsed config/data_generation/facilities.yaml.

    reference_data:
        Loaded reference DataFrames.

    sequence_number:
        Zero-based Facility sequence number.

    Returns
    -------
    dict
        Generated Facility record.
    """
    contact_config = require_config_section(
        facility_config,
        "contact_information",
        "facilities.yaml",
    )

    audit_config = require_config_section(
        facility_config,
        "audit",
        "facilities.yaml",
    )

    facility_id = build_facility_id(
        facility_config=facility_config,
        sequence_number=sequence_number,
    )

    facility_reference_row = sample_dataframe_row(
        rng=rng,
        dataframe=reference_data["facility"],
        dataset_name="reference.facility",
    )

    geography_row = sample_dataframe_row(
        rng=rng,
        dataframe=reference_data["geography"],
        dataset_name="reference.geography",
        weight_column="population_weight",
    )

    generate_phone_numbers = bool(
        require_config_value(
            contact_config,
            "generate_phone_numbers",
            "facilities.contact_information",
        )
    )

    generate_email_addresses = bool(
        require_config_value(
            contact_config,
            "generate_email_addresses",
            "facilities.contact_information",
        )
    )

    return {
        "facility_id": facility_id,
        "facility_name": build_facility_name(
            facility_config=facility_config,
            facility_id=facility_id,
            facility_reference_row=facility_reference_row,
        ),
        "facility_type_code": facility_reference_row.get("facility_type_code"),
        "facility_type_description": facility_reference_row.get("facility_type_description"),
        "zip_code": geography_row.get("zip_code"),
        "city": geography_row.get("city"),
        "county": geography_row.get("county_name") or geography_row.get("county"),
        "state_code": geography_row.get("state_code") or geography_row.get("state"),
        "state_name": geography_row.get("state_name"),
        "region": geography_row.get("region"),
        "phone_number": build_phone_number(
            rng=rng,
            enabled=generate_phone_numbers,
        ),
        "email_address": build_email_address(
            facility_config=facility_config,
            enabled=generate_email_addresses,
            facility_id=facility_id,
        ),
        "source_system": require_config_value(
            audit_config,
            "source_system",
            "facilities.audit",
        ),
        "record_status": require_config_value(
            audit_config,
            "record_status",
            "facilities.audit",
        ),
        "created_at": pd.Timestamp(
            require_config_value(
                audit_config,
                "created_at",
                "facilities.audit",
            )
        ),
        "updated_at": pd.Timestamp(
            require_config_value(
                audit_config,
                "updated_at",
                "facilities.audit",
            )
        ),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_facility_dataset(
    context,
    global_config: Dict[str, Any],
    facility_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate the complete canonical Facility dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    facility_config:
        Parsed config/data_generation/facilities.yaml.

    reference_data:
        Loaded reference DataFrames.

    Returns
    -------
    pandas.DataFrame
        Generated Facility dataset.
    """
    logger = context.get_logger(MODULE_NAME)

    population_config = require_config_section(
        facility_config,
        "facility_population",
        "facilities.yaml",
    )

    total_facilities = int(
        require_config_value(
            population_config,
            "total_facilities",
            "facilities.facility_population",
        )
    )

    random_seed = resolve_random_seed(
        global_config=global_config,
        facility_config=facility_config,
    )

    rng = np.random.default_rng(random_seed)

    logger.info("Generating Facility dataset. Expected rows: %s", total_facilities)

    records: List[Dict[str, Any]] = []

    for sequence_number in range(total_facilities):
        records.append(
            build_facility_record(
                rng=rng,
                facility_config=facility_config,
                reference_data=reference_data,
                sequence_number=sequence_number,
            )
        )

    dataframe = pd.DataFrame(records)

    logger.info("Generated Facility dataset. Actual rows: %s", len(dataframe))

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(facility_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible validation rules.

    Parameters
    ----------
    facility_config:
        Parsed config/data_generation/facilities.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        facility_config,
        "validation",
        "facilities.yaml",
    )

    required_columns = ["facility_id"]
    no_null_columns = ["facility_id"]

    if bool(validation_config.get("require_facility_type", False)):
        required_columns.append("facility_type_code")
        no_null_columns.append("facility_type_code")

    if bool(validation_config.get("require_zip_code", False)):
        required_columns.append("zip_code")
        no_null_columns.append("zip_code")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_facility_ids", False)):
        validation_rules["primary_key"] = ["facility_id"]

    return validation_rules


def validate_facility_dataset(
    context,
    dataframe: pd.DataFrame,
    facility_config: Dict[str, Any],
) -> None:
    """
    Validate generated Facility dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Facility DataFrame.

    facility_config:
        Parsed config/data_generation/facilities.yaml.
    """
    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=build_validation_rules(facility_config),
        dataset_name="raw.facilities",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_facility_output_path(facility_config: Dict[str, Any]) -> str:
    """
    Resolve Facility output path.

    Parameters
    ----------
    facility_config:
        Parsed config/data_generation/facilities.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(
        facility_config,
        "output",
        "facilities.yaml",
    )

    file_name = require_config_value(
        output_config,
        "file_name",
        "facilities.output",
    )

    return f"data/raw/{file_name}"


def write_facility_dataset(
    context,
    dataframe: pd.DataFrame,
    facility_config: Dict[str, Any],
) -> str:
    """
    Write Facility dataset to storage.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Facility DataFrame.

    facility_config:
        Parsed config/data_generation/facilities.yaml.

    Returns
    -------
    str
        Written dataset path.
    """
    output_path = resolve_facility_output_path(facility_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.facilities",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_facility_metadata(
    context,
    dataframe: pd.DataFrame,
    facility_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Facility metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Facility DataFrame.

    facility_config:
        Parsed config/data_generation/facilities.yaml.

    output_path:
        Written Facility dataset path.
    """
    metadata_config = require_config_section(
        facility_config,
        "metadata",
        "facilities.yaml",
    )

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.facilities",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="facility",
            primary_key=["facility_id"],
            description="Canonical MedFabric synthetic Facility dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/facilities_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.facilities",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/facilities_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.facilities",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/facilities_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.facilities",
            source_datasets=[
                "reference.facility",
                "reference.geography",
            ],
            output_dataset="raw.facilities",
            transformation_name="generate_facility_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/facilities_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_facility_generation(context) -> pd.DataFrame:
    """
    Execute the complete Facility generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Facility dataset.
    """
    global_config = load_global_generation_config(context)
    facility_config = load_facility_generation_config(context)

    reference_data = load_reference_data(
        context=context,
        facility_config=facility_config,
    )

    facility_dataframe = generate_facility_dataset(
        context=context,
        global_config=global_config,
        facility_config=facility_config,
        reference_data=reference_data,
    )

    validate_facility_dataset(
        context=context,
        dataframe=facility_dataframe,
        facility_config=facility_config,
    )

    output_path = write_facility_dataset(
        context=context,
        dataframe=facility_dataframe,
        facility_config=facility_config,
    )

    write_facility_metadata(
        context=context,
        dataframe=facility_dataframe,
        facility_config=facility_config,
        output_path=output_path,
    )

    return facility_dataframe


def main() -> None:
    """
    Main entry point for Facility generation.

    Run Command
    -----------
    python -m src.data_generation.generators.facility_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        facility_dataframe = run_facility_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Facility generation completed successfully. Rows: %s",
            len(facility_dataframe),
        )

        print("MedFabric facility generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Facility generation failed.")
        logger.exception("Facility generation failed.")
        raise PipelineError("Facility generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()