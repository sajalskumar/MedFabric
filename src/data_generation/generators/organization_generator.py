###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/organization_generator.py
#
# Purpose:
#     Generates the canonical Organization dataset used by the MedFabric
#     Synthetic Data Engine.
#
# Business Context:
#     Organizations represent enterprise entities such as provider groups,
#     health systems, IPAs, specialty groups, payers, employer groups, and other
#     network or contracting entities.
#
#     Organizations are important for:
#         - Provider group hierarchy
#         - Facility ownership
#         - Network analytics
#         - Contracting analytics
#         - Attribution reporting
#         - Provider performance rollups
#         - Enterprise relationship modeling
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/organizations.yaml
#     reference/providers/provider_organization_reference.parquet
#     reference/geography/us_geography_reference.parquet
#
# Outputs:
#     data/raw/organizations.parquet
#     data/metadata/organizations_dataset_metadata.json
#     data/metadata/organizations_column_metadata.csv
#     data/metadata/organizations_statistics.json
#     data/metadata/organizations_lineage.json
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
#     6. Dataset-specific settings come from organizations.yaml.
#     7. Global execution settings come from generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.organization_generator
#
# Expected Output:
#     Canonical Organization dataset and metadata written to configured output
#     paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.organization"
STEP_NAME = "Generate Organization Dataset"


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
        Raised when the required section is missing or is not a dictionary.
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


def load_organization_generation_config(context) -> Dict[str, Any]:
    """
    Load Organization-specific generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/organizations.yaml.
    """
    return context.configuration.load_yaml("data_generation/organizations.yaml")


###############################################################################
# REFERENCE DATA LOADING
###############################################################################


def load_reference_data(
    context,
    organization_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load Organization reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    organization_config:
        Parsed config/data_generation/organizations.yaml.

    Returns
    -------
    dict
        Loaded organization and geography reference DataFrames.

    Raises
    ------
    PipelineError
        Raised when required configuration sections or source paths are missing.
    """
    logger = context.get_logger(MODULE_NAME)

    organization_reference_config = require_config_section(
        organization_config,
        "organization",
        "organizations.yaml",
    )

    geography_config = require_config_section(
        organization_config,
        "geography",
        "organizations.yaml",
    )

    reference_paths = {
        "organization": require_config_value(
            organization_reference_config,
            "source",
            "organizations.organization",
        ),
        "geography": require_config_value(
            geography_config,
            "source",
            "organizations.geography",
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
            "Loaded Organization reference data: %s | Rows: %s | Path: %s",
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
# ORGANIZATION GENERATION HELPERS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    organization_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for reproducible Organization generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    organization_config:
        Parsed config/data_generation/organizations.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        organization_config,
        "reproducibility",
        "organizations.yaml",
    )

    use_global_random_seed = bool(
        require_config_value(
            reproducibility_config,
            "use_global_random_seed",
            "organizations.reproducibility",
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
            "organizations.reproducibility",
        )
    )


def build_organization_id(
    organization_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build a configured Organization identifier.

    Parameters
    ----------
    organization_config:
        Parsed config/data_generation/organizations.yaml.

    sequence_number:
        Zero-based sequence number.

    Returns
    -------
    str
        Generated Organization identifier.
    """
    identifier_config = require_config_section(
        organization_config,
        "organization_identifier",
        "organizations.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "organizations.organization_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "organizations.organization_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "organizations.organization_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def build_organization_name(
    organization_config: Dict[str, Any],
    organization_id: str,
    organization_reference_row: Dict[str, Any],
) -> str:
    """
    Build Organization display name using configured naming template.

    Parameters
    ----------
    organization_config:
        Parsed config/data_generation/organizations.yaml.

    organization_id:
        Generated Organization identifier.

    organization_reference_row:
        Sampled organization reference row.

    Returns
    -------
    str
        Generated Organization name.

    Raises
    ------
    PipelineError
        Raised when required naming configuration or base name is missing.
    """
    naming_config = require_config_section(
        organization_config,
        "naming",
        "organizations.yaml",
    )

    template = require_config_value(
        naming_config,
        "organization_name_template",
        "organizations.naming",
    )

    base_organization_name = organization_reference_row.get("organization_name")

    if base_organization_name is None:
        raise PipelineError(
            "Organization reference row must contain organization_name."
        )

    return template.format(
        base_organization_name=base_organization_name,
        organization_id=organization_id,
    )


def build_phone_number(
    rng: np.random.Generator,
    enabled: bool,
) -> str | None:
    """
    Generate a synthetic Organization phone number.

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
    """
    if not enabled:
        return None

    area_code = int(rng.integers(200, 1000))
    prefix = int(rng.integers(200, 1000))
    line_number = int(rng.integers(0, 10000))

    return f"{area_code}-{prefix}-{line_number:04d}"


def build_email_address(
    organization_config: Dict[str, Any],
    enabled: bool,
    organization_id: str,
) -> str | None:
    """
    Generate a synthetic Organization email address.

    Parameters
    ----------
    organization_config:
        Parsed config/data_generation/organizations.yaml.

    enabled:
        Whether email generation is enabled.

    organization_id:
        Generated Organization identifier.

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
        organization_config,
        "contact_information",
        "organizations.yaml",
    )

    email_domain = require_config_value(
        contact_config,
        "email_domain",
        "organizations.contact_information",
    )

    return f"organization.{str(organization_id).lower()}@{email_domain}"


def build_organization_record(
    rng: np.random.Generator,
    organization_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical Organization record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    organization_config:
        Parsed config/data_generation/organizations.yaml.

    reference_data:
        Loaded reference DataFrames.

    sequence_number:
        Zero-based Organization sequence number.

    Returns
    -------
    dict
        Generated Organization record.
    """
    contact_config = require_config_section(
        organization_config,
        "contact_information",
        "organizations.yaml",
    )

    audit_config = require_config_section(
        organization_config,
        "audit",
        "organizations.yaml",
    )

    organization_id = build_organization_id(
        organization_config=organization_config,
        sequence_number=sequence_number,
    )

    organization_reference_row = sample_dataframe_row(
        rng=rng,
        dataframe=reference_data["organization"],
        dataset_name="reference.organization",
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
            "organizations.contact_information",
        )
    )

    generate_email_addresses = bool(
        require_config_value(
            contact_config,
            "generate_email_addresses",
            "organizations.contact_information",
        )
    )

    return {
        "organization_id": organization_id,
        "organization_name": build_organization_name(
            organization_config=organization_config,
            organization_id=organization_id,
            organization_reference_row=organization_reference_row,
        ),
        "base_organization_name": organization_reference_row.get("organization_name"),
        "organization_type": organization_reference_row.get("organization_type"),
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
            organization_config=organization_config,
            enabled=generate_email_addresses,
            organization_id=organization_id,
        ),
        "source_system": require_config_value(
            audit_config,
            "source_system",
            "organizations.audit",
        ),
        "record_status": require_config_value(
            audit_config,
            "record_status",
            "organizations.audit",
        ),
        "created_at": pd.Timestamp(
            require_config_value(
                audit_config,
                "created_at",
                "organizations.audit",
            )
        ),
        "updated_at": pd.Timestamp(
            require_config_value(
                audit_config,
                "updated_at",
                "organizations.audit",
            )
        ),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_organization_dataset(
    context,
    global_config: Dict[str, Any],
    organization_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate the complete canonical Organization dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    organization_config:
        Parsed config/data_generation/organizations.yaml.

    reference_data:
        Loaded reference DataFrames.

    Returns
    -------
    pandas.DataFrame
        Generated Organization dataset.
    """
    logger = context.get_logger(MODULE_NAME)

    population_config = require_config_section(
        organization_config,
        "organization_population",
        "organizations.yaml",
    )

    total_organizations = int(
        require_config_value(
            population_config,
            "total_organizations",
            "organizations.organization_population",
        )
    )

    random_seed = resolve_random_seed(
        global_config=global_config,
        organization_config=organization_config,
    )

    rng = np.random.default_rng(random_seed)

    logger.info(
        "Generating Organization dataset. Expected rows: %s",
        total_organizations,
    )

    records: List[Dict[str, Any]] = []

    for sequence_number in range(total_organizations):
        records.append(
            build_organization_record(
                rng=rng,
                organization_config=organization_config,
                reference_data=reference_data,
                sequence_number=sequence_number,
            )
        )

    dataframe = pd.DataFrame(records)

    logger.info(
        "Generated Organization dataset. Actual rows: %s",
        len(dataframe),
    )

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(
    organization_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible rules for Organization dataset.

    Parameters
    ----------
    organization_config:
        Parsed config/data_generation/organizations.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        organization_config,
        "validation",
        "organizations.yaml",
    )

    required_columns = ["organization_id"]
    no_null_columns = ["organization_id"]

    if bool(validation_config.get("require_organization_name", False)):
        required_columns.append("organization_name")
        no_null_columns.append("organization_name")

    if bool(validation_config.get("require_organization_type", False)):
        required_columns.append("organization_type")
        no_null_columns.append("organization_type")

    if bool(validation_config.get("require_zip_code", False)):
        required_columns.append("zip_code")
        no_null_columns.append("zip_code")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_organization_ids", False)):
        validation_rules["primary_key"] = ["organization_id"]

    return validation_rules


def validate_organization_dataset(
    context,
    dataframe: pd.DataFrame,
    organization_config: Dict[str, Any],
) -> None:
    """
    Validate generated Organization dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Organization DataFrame.

    organization_config:
        Parsed config/data_generation/organizations.yaml.
    """
    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=build_validation_rules(organization_config),
        dataset_name="raw.organizations",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_organization_output_path(
    organization_config: Dict[str, Any],
) -> str:
    """
    Resolve Organization output path.

    Parameters
    ----------
    organization_config:
        Parsed config/data_generation/organizations.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(
        organization_config,
        "output",
        "organizations.yaml",
    )

    file_name = require_config_value(
        output_config,
        "file_name",
        "organizations.output",
    )

    return f"data/raw/{file_name}"


def write_organization_dataset(
    context,
    dataframe: pd.DataFrame,
    organization_config: Dict[str, Any],
) -> str:
    """
    Write Organization dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Organization DataFrame.

    organization_config:
        Parsed config/data_generation/organizations.yaml.

    Returns
    -------
    str
        Written dataset path.
    """
    output_path = resolve_organization_output_path(organization_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.organizations",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_organization_metadata(
    context,
    dataframe: pd.DataFrame,
    organization_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Organization metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Organization DataFrame.

    organization_config:
        Parsed config/data_generation/organizations.yaml.

    output_path:
        Written Organization dataset path.
    """
    metadata_config = require_config_section(
        organization_config,
        "metadata",
        "organizations.yaml",
    )

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.organizations",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="organization",
            primary_key=["organization_id"],
            description="Canonical MedFabric synthetic Organization dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/organizations_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.organizations",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/organizations_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.organizations",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/organizations_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.organizations",
            source_datasets=[
                "reference.provider_organization",
                "reference.geography",
            ],
            output_dataset="raw.organizations",
            transformation_name="generate_organization_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/organizations_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_organization_generation(context) -> pd.DataFrame:
    """
    Execute the complete Organization generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Organization dataset.
    """
    global_config = load_global_generation_config(context)
    organization_config = load_organization_generation_config(context)

    reference_data = load_reference_data(
        context=context,
        organization_config=organization_config,
    )

    organization_dataframe = generate_organization_dataset(
        context=context,
        global_config=global_config,
        organization_config=organization_config,
        reference_data=reference_data,
    )

    validate_organization_dataset(
        context=context,
        dataframe=organization_dataframe,
        organization_config=organization_config,
    )

    output_path = write_organization_dataset(
        context=context,
        dataframe=organization_dataframe,
        organization_config=organization_config,
    )

    write_organization_metadata(
        context=context,
        dataframe=organization_dataframe,
        organization_config=organization_config,
        output_path=output_path,
    )

    return organization_dataframe


def main() -> None:
    """
    Main entry point for Organization generation.

    Run Command
    -----------
    python -m src.data_generation.generators.organization_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        organization_dataframe = run_organization_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Organization generation completed successfully. Rows: %s",
            len(organization_dataframe),
        )

        print("MedFabric organization generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Organization generation failed.")
        logger.exception("Organization generation failed.")
        raise PipelineError("Organization generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()