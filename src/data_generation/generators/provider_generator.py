###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/provider_generator.py
#
# Purpose:
#     Generates the canonical Provider dataset used by the MedFabric Synthetic
#     Data Engine.
#
# Business Context:
#     Providers are foundational entities in MedFabric. Claims, encounters,
#     facility affiliations, attribution, network analytics, provider
#     performance, quality measurement, care management, and Member 360
#     reporting all depend on a consistent, validated, and reproducible Provider
#     dataset.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/providers.yaml
#     reference/providers/provider_specialty_reference.parquet
#     reference/providers/provider_taxonomy_reference.parquet
#     reference/providers/provider_organization_reference.parquet
#     reference/providers/primary_care_specialty_reference.parquet
#     reference/facilities/facility_reference.parquet
#     reference/geography/us_geography_reference.parquet
#
# Outputs:
#     data/raw/providers.parquet
#     data/metadata/providers_dataset_metadata.json
#     data/metadata/providers_column_metadata.csv
#     data/metadata/providers_statistics.json
#     data/metadata/providers_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/providers.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.provider_generator
#
# Expected Output:
#     Canonical Provider dataset and metadata written to configured output paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.provider"
STEP_NAME = "Generate Provider Dataset"


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
        Configuration dictionary.

    key:
        Required key.

    config_name:
        Human-readable configuration name used in error messages.

    Returns
    -------
    Any
        Required configuration value.

    Raises
    ------
    PipelineError
        If the required key is missing.
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
        Configuration dictionary.

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
        If the section is missing or is not a dictionary.
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


def load_provider_generation_config(context) -> Dict[str, Any]:
    """
    Load Provider-specific generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/providers.yaml.
    """
    return context.configuration.load_yaml("data_generation/providers.yaml")


###############################################################################
# REFERENCE DATA LOADING
###############################################################################


def load_optional_reference(
    context,
    reference_name: str,
    source_path: str,
) -> pd.DataFrame:
    """
    Load and validate one configured reference dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    reference_name:
        Logical reference name.

    source_path:
        Configured Parquet path.

    Returns
    -------
    pandas.DataFrame
        Loaded reference DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    dataframe = context.storage.read_parquet(source_path)

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules={
            "allow_empty": False,
            "min_rows": 1,
        },
        dataset_name=f"reference.{reference_name}",
    )

    logger.info(
        "Loaded Provider reference data: %s | Rows: %s | Path: %s",
        reference_name,
        len(dataframe),
        source_path,
    )

    return dataframe


def load_provider_reference_data(
    context,
    provider_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load reference datasets required for Provider generation.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    provider_config:
        Parsed config/data_generation/providers.yaml.

    Returns
    -------
    dict
        Dictionary of loaded reference DataFrames.
    """
    specialty_config = require_config_section(
        provider_config,
        "specialty",
        "providers.yaml",
    )

    taxonomy_config = require_config_section(
        provider_config,
        "taxonomy",
        "providers.yaml",
    )

    organization_config = require_config_section(
        provider_config,
        "organization",
        "providers.yaml",
    )

    facility_relationship_config = require_config_section(
        provider_config,
        "facility_relationship",
        "providers.yaml",
    )

    geography_config = require_config_section(
        provider_config,
        "geography",
        "providers.yaml",
    )

    primary_care_config = require_config_section(
        provider_config,
        "primary_care",
        "providers.yaml",
    )

    reference_data: Dict[str, pd.DataFrame] = {
        "specialty": load_optional_reference(
            context=context,
            reference_name="provider_specialty",
            source_path=require_config_value(
                specialty_config,
                "source",
                "providers.specialty",
            ),
        ),
        "taxonomy": load_optional_reference(
            context=context,
            reference_name="provider_taxonomy",
            source_path=require_config_value(
                taxonomy_config,
                "source",
                "providers.taxonomy",
            ),
        ),
        "organization": load_optional_reference(
            context=context,
            reference_name="provider_organization",
            source_path=require_config_value(
                organization_config,
                "organization_source",
                "providers.organization",
            ),
        ),
        "facility": load_optional_reference(
            context=context,
            reference_name="facility_reference",
            source_path=require_config_value(
                facility_relationship_config,
                "facility_source",
                "providers.facility_relationship",
            ),
        ),
        "geography": load_optional_reference(
            context=context,
            reference_name="us_geography",
            source_path=require_config_value(
                geography_config,
                "source",
                "providers.geography",
            ),
        ),
        "primary_care_specialty": load_optional_reference(
            context=context,
            reference_name="primary_care_specialty",
            source_path=require_config_value(
                primary_care_config,
                "primary_care_specialty_source",
                "providers.primary_care",
            ),
        ),
    }

    return reference_data


###############################################################################
# SAMPLING HELPERS
###############################################################################


def normalize_weights(weights: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Normalize numeric selection weights.

    Parameters
    ----------
    weights:
        Numeric weight array.

    dataset_name:
        Dataset name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Normalized probability array.
    """
    total_weight = float(weights.sum())

    if total_weight <= 0:
        raise PipelineError(f"Weights must sum to a positive value for {dataset_name}.")

    return weights / total_weight


def sample_weighted_label(
    rng: np.random.Generator,
    weights_by_label: Dict[str, float],
    rule_name: str,
) -> str:
    """
    Sample one configured label from configured label weights.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    weights_by_label:
        Dictionary of labels and weights.

    rule_name:
        Human-readable rule name.

    Returns
    -------
    str
        Sampled label.
    """
    labels = list(weights_by_label.keys())
    weights = np.array([float(weights_by_label[label]) for label in labels])
    probabilities = normalize_weights(weights, rule_name)

    return str(rng.choice(labels, p=probabilities))


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
        Reference DataFrame.

    dataset_name:
        Dataset name used in validation errors.

    weight_column:
        Optional weight column. Used only when present.

    Returns
    -------
    dict
        Sampled row.
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


def sample_dataframe_value(
    rng: np.random.Generator,
    dataframe: pd.DataFrame,
    value_column: str,
    dataset_name: str,
    weight_column: str = "selection_weight",
) -> Any:
    """
    Sample one value from a DataFrame.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    dataframe:
        Reference DataFrame.

    value_column:
        Column to sample.

    dataset_name:
        Dataset name used in validation errors.

    weight_column:
        Optional weight column. Used only when present.

    Returns
    -------
    Any
        Sampled value.
    """
    if value_column not in dataframe.columns:
        raise PipelineError(
            f"Column '{value_column}' not found in reference dataset {dataset_name}."
        )

    if weight_column in dataframe.columns:
        sample_frame = dataframe[[value_column, weight_column]].dropna()

        if sample_frame.empty:
            raise PipelineError(f"No valid rows found in {dataset_name}.")

        values = sample_frame[value_column].to_numpy()
        weights = sample_frame[weight_column].astype(float).to_numpy()
        probabilities = normalize_weights(weights, dataset_name)

        return rng.choice(values, p=probabilities)

    values = dataframe[value_column].dropna().to_numpy()

    if len(values) == 0:
        raise PipelineError(f"No valid values found in {dataset_name}.{value_column}.")

    return rng.choice(values)


###############################################################################
# PROVIDER GENERATION HELPERS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    provider_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for Provider generation.

    Parameters
    ----------
    global_config:
        Parsed generation.yaml.

    provider_config:
        Parsed providers.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        provider_config,
        "reproducibility",
        "providers.yaml",
    )

    use_global_random_seed = bool(
        require_config_value(
            reproducibility_config,
            "use_global_random_seed",
            "providers.reproducibility",
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
            "providers.reproducibility",
        )
    )


def build_provider_id(provider_config: Dict[str, Any], sequence_number: int) -> str:
    """
    Build configured Provider identifier.

    Parameters
    ----------
    provider_config:
        Parsed providers.yaml.

    sequence_number:
        Zero-based sequence number.

    Returns
    -------
    str
        Generated provider identifier.
    """
    identifier_config = require_config_section(
        provider_config,
        "provider_identifier",
        "providers.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "providers.provider_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "providers.provider_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "providers.provider_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def build_npi(
    provider_config: Dict[str, Any],
    sequence_number: int,
) -> str | None:
    """
    Build deterministic synthetic NPI.

    Parameters
    ----------
    provider_config:
        Parsed providers.yaml.

    sequence_number:
        Zero-based sequence number.

    Returns
    -------
    str or None
        Generated NPI if enabled.
    """
    npi_config = require_config_section(provider_config, "npi", "providers.yaml")

    generate_npi = bool(
        require_config_value(
            npi_config,
            "generate_npi",
            "providers.npi",
        )
    )

    if not generate_npi:
        return None

    npi_length = int(
        require_config_value(
            npi_config,
            "npi_length",
            "providers.npi",
        )
    )

    npi_numeric_value = sequence_number + 1

    return f"{npi_numeric_value:0{npi_length}d}"[-npi_length:]


def sample_provider_type(
    rng: np.random.Generator,
    provider_config: Dict[str, Any],
) -> str:
    """
    Sample provider type from configured probabilities.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    provider_config:
        Parsed providers.yaml.

    Returns
    -------
    str
        Generated provider type.
    """
    provider_type_config = require_config_section(
        provider_config,
        "provider_type",
        "providers.yaml",
    )

    weights_by_label = {
        "Individual": float(
            require_config_value(
                provider_type_config,
                "individual_provider_probability",
                "providers.provider_type",
            )
        ),
        "Organization": float(
            require_config_value(
                provider_type_config,
                "organizational_provider_probability",
                "providers.provider_type",
            )
        ),
    }

    return sample_weighted_label(
        rng=rng,
        weights_by_label=weights_by_label,
        rule_name="providers.provider_type",
    )


def sample_network_status(
    rng: np.random.Generator,
    provider_config: Dict[str, Any],
) -> str | None:
    """
    Sample network status.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    provider_config:
        Parsed providers.yaml.

    Returns
    -------
    str or None
        Network status if enabled.
    """
    network_config = require_config_section(provider_config, "network", "providers.yaml")

    generate_network_status = bool(
        require_config_value(
            network_config,
            "generate_network_status",
            "providers.network",
        )
    )

    if not generate_network_status:
        return None

    weights_by_label = {
        "In Network": float(
            require_config_value(
                network_config,
                "in_network_probability",
                "providers.network",
            )
        ),
        "Out of Network": float(
            require_config_value(
                network_config,
                "out_of_network_probability",
                "providers.network",
            )
        ),
    }

    return sample_weighted_label(
        rng=rng,
        weights_by_label=weights_by_label,
        rule_name="providers.network",
    )


def build_organization_assignments(
    rng: np.random.Generator,
    provider_config: Dict[str, Any],
    total_providers: int,
    organization_reference: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Assign providers to organizations.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    provider_config:
        Parsed providers.yaml.

    total_providers:
        Number of providers being generated.

    organization_reference:
        Provider organization reference DataFrame.

    Returns
    -------
    list[dict]
        Organization row assignment for each provider.
    """
    organization_population_config = require_config_section(
        provider_config,
        "organization_population",
        "providers.yaml",
    )

    minimum_providers = int(
        require_config_value(
            organization_population_config,
            "minimum_providers_per_organization",
            "providers.organization_population",
        )
    )

    maximum_providers = int(
        require_config_value(
            organization_population_config,
            "maximum_providers_per_organization",
            "providers.organization_population",
        )
    )

    if minimum_providers <= 0:
        raise PipelineError("minimum_providers_per_organization must be positive.")

    if maximum_providers < minimum_providers:
        raise PipelineError(
            "maximum_providers_per_organization cannot be less than minimum_providers_per_organization."
        )

    assignments: List[Dict[str, Any]] = []

    while len(assignments) < total_providers:
        organization_row = sample_dataframe_row(
            rng=rng,
            dataframe=organization_reference,
            dataset_name="reference.provider_organization",
        )

        organization_size = int(
            rng.integers(minimum_providers, maximum_providers + 1)
        )

        for _ in range(organization_size):
            if len(assignments) < total_providers:
                assignments.append(organization_row)

    return assignments


def select_taxonomy_for_specialty(
    rng: np.random.Generator,
    taxonomy_reference: pd.DataFrame,
    specialty_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Select taxonomy row related to selected specialty when possible.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    taxonomy_reference:
        Provider taxonomy reference DataFrame.

    specialty_row:
        Selected specialty row.

    Returns
    -------
    dict
        Selected taxonomy row.
    """
    if "specialty_code" in specialty_row and "specialty_code" in taxonomy_reference.columns:
        matched_taxonomy = taxonomy_reference[
            taxonomy_reference["specialty_code"] == specialty_row["specialty_code"]
        ]

        if not matched_taxonomy.empty:
            return sample_dataframe_row(
                rng=rng,
                dataframe=matched_taxonomy,
                dataset_name="reference.provider_taxonomy",
            )

    return sample_dataframe_row(
        rng=rng,
        dataframe=taxonomy_reference,
        dataset_name="reference.provider_taxonomy",
    )


def derive_primary_care_flag(
    specialty_row: Dict[str, Any],
    primary_care_reference: pd.DataFrame,
) -> bool:
    """
    Derive primary care flag from specialty data.

    Parameters
    ----------
    specialty_row:
        Selected specialty row.

    primary_care_reference:
        Primary care specialty reference DataFrame.

    Returns
    -------
    bool
        Whether selected provider is primary care.
    """
    if "is_primary_care" in specialty_row:
        return bool(specialty_row["is_primary_care"])

    if "specialty_code" not in specialty_row:
        return False

    if "specialty_code" not in primary_care_reference.columns:
        return False

    specialty_code = specialty_row["specialty_code"]

    return bool(
        primary_care_reference["specialty_code"]
        .astype(str)
        .eq(str(specialty_code))
        .any()
    )


def derive_attribution_eligible_flag(
    provider_config: Dict[str, Any],
    is_primary_care: bool,
) -> bool:
    """
    Derive attribution eligibility flag.

    Parameters
    ----------
    provider_config:
        Parsed providers.yaml.

    is_primary_care:
        Provider primary care flag.

    Returns
    -------
    bool
        Attribution eligibility flag.
    """
    attribution_config = require_config_section(
        provider_config,
        "attribution_eligibility",
        "providers.yaml",
    )

    generate_attribution_eligible_flag = bool(
        require_config_value(
            attribution_config,
            "generate_attribution_eligible_flag",
            "providers.attribution_eligibility",
        )
    )

    if not generate_attribution_eligible_flag:
        return False

    require_primary_care = bool(
        require_config_value(
            attribution_config,
            "require_primary_care_for_pcp_attribution",
            "providers.attribution_eligibility",
        )
    )

    if require_primary_care:
        return bool(is_primary_care)

    return True


def build_phone_number(
    rng: np.random.Generator,
    enabled: bool,
) -> str | None:
    """
    Generate synthetic provider phone number when enabled.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    enabled:
        Whether phone number generation is enabled.

    Returns
    -------
    str or None
        Phone number when enabled.
    """
    if not enabled:
        return None

    area_code = int(rng.integers(200, 1000))
    prefix = int(rng.integers(200, 1000))
    line_number = int(rng.integers(0, 10000))

    return f"{area_code}-{prefix}-{line_number:04d}"


def build_email_address(
    enabled: bool,
    provider_id: str,
    provider_type: str,
) -> str | None:
    """
    Generate synthetic provider email address when enabled.

    Parameters
    ----------
    enabled:
        Whether email generation is enabled.

    provider_id:
        Generated provider ID.

    provider_type:
        Generated provider type.

    Returns
    -------
    str or None
        Email address when enabled.
    """
    if not enabled:
        return None

    normalized_provider_type = str(provider_type).lower().replace(" ", "_")
    normalized_provider_id = str(provider_id).lower()

    return f"{normalized_provider_type}.{normalized_provider_id}@medfabric.example"


def build_provider_name(
    provider_id: str,
    provider_type: str,
    specialty_row: Dict[str, Any],
    organization_row: Dict[str, Any],
) -> str:
    """
    Build provider display name.

    Parameters
    ----------
    provider_id:
        Generated provider identifier.

    provider_type:
        Generated provider type.

    specialty_row:
        Selected specialty row.

    organization_row:
        Selected organization row.

    Returns
    -------
    str
        Provider display name.
    """
    if provider_type == "Organization":
        if "organization_name" in organization_row:
            return str(organization_row["organization_name"])

        return f"Provider Organization {provider_id}"

    specialty_description = (
        specialty_row.get("specialty_description")
        or specialty_row.get("specialty_name")
        or specialty_row.get("specialty_code")
        or "Provider"
    )

    return f"{specialty_description} {provider_id}"


def build_provider_record(
    rng: np.random.Generator,
    provider_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
    sequence_number: int,
    organization_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one canonical Provider record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    provider_config:
        Parsed providers.yaml.

    reference_data:
        Loaded reference DataFrames.

    sequence_number:
        Zero-based provider sequence number.

    organization_row:
        Assigned provider organization row.

    Returns
    -------
    dict
        Generated Provider record.
    """
    contact_config = require_config_section(
        provider_config,
        "contact_information",
        "providers.yaml",
    )

    provider_id = build_provider_id(provider_config, sequence_number)
    provider_type = sample_provider_type(rng, provider_config)
    npi = build_npi(provider_config, sequence_number)

    specialty_row = sample_dataframe_row(
        rng=rng,
        dataframe=reference_data["specialty"],
        dataset_name="reference.provider_specialty",
    )

    taxonomy_row = select_taxonomy_for_specialty(
        rng=rng,
        taxonomy_reference=reference_data["taxonomy"],
        specialty_row=specialty_row,
    )

    geography_row = sample_dataframe_row(
        rng=rng,
        dataframe=reference_data["geography"],
        dataset_name="reference.us_geography",
        weight_column="population_weight",
    )

    facility_row = sample_dataframe_row(
        rng=rng,
        dataframe=reference_data["facility"],
        dataset_name="reference.facility",
    )

    is_primary_care = derive_primary_care_flag(
        specialty_row=specialty_row,
        primary_care_reference=reference_data["primary_care_specialty"],
    )

    attribution_eligible = derive_attribution_eligible_flag(
        provider_config=provider_config,
        is_primary_care=is_primary_care,
    )

    generate_phone_numbers = bool(
        require_config_value(
            contact_config,
            "generate_phone_numbers",
            "providers.contact_information",
        )
    )

    generate_email_addresses = bool(
        require_config_value(
            contact_config,
            "generate_email_addresses",
            "providers.contact_information",
        )
    )

    provider_name = build_provider_name(
        provider_id=provider_id,
        provider_type=provider_type,
        specialty_row=specialty_row,
        organization_row=organization_row,
    )

    return {
        "provider_id": provider_id,
        "npi": npi,
        "provider_name": provider_name,
        "provider_type": provider_type,
        "specialty_code": specialty_row.get("specialty_code"),
        "specialty_description": (
            specialty_row.get("specialty_description")
            or specialty_row.get("specialty_name")
        ),
        "taxonomy_code": taxonomy_row.get("taxonomy_code"),
        "taxonomy_description": taxonomy_row.get("taxonomy_description"),
        "organization_name": organization_row.get("organization_name"),
        "organization_type": organization_row.get("organization_type"),
        "facility_type_code": facility_row.get("facility_type_code"),
        "facility_type_description": facility_row.get("facility_type_description"),
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
            enabled=generate_email_addresses,
            provider_id=provider_id,
            provider_type=provider_type,
        ),
        "network_status": sample_network_status(rng, provider_config),
        "is_primary_care": is_primary_care,
        "attribution_eligible": attribution_eligible,
        "source_system": "MedFabric Synthetic Data Engine",
        "record_status": "Active",
        "created_at": pd.Timestamp.today().normalize(),
        "updated_at": pd.Timestamp.today().normalize(),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_provider_dataset(
    context,
    global_config: Dict[str, Any],
    provider_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate the complete canonical Provider dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed generation.yaml.

    provider_config:
        Parsed providers.yaml.

    reference_data:
        Loaded reference DataFrames.

    Returns
    -------
    pandas.DataFrame
        Generated Provider dataset.
    """
    logger = context.get_logger(MODULE_NAME)

    population_config = require_config_section(
        provider_config,
        "provider_population",
        "providers.yaml",
    )

    total_providers = int(
        require_config_value(
            population_config,
            "total_providers",
            "providers.provider_population",
        )
    )

    random_seed = resolve_random_seed(
        global_config=global_config,
        provider_config=provider_config,
    )

    rng = np.random.default_rng(random_seed)

    organization_assignments = build_organization_assignments(
        rng=rng,
        provider_config=provider_config,
        total_providers=total_providers,
        organization_reference=reference_data["organization"],
    )

    logger.info("Generating Provider dataset. Expected rows: %s", total_providers)

    records: List[Dict[str, Any]] = []

    for sequence_number in range(total_providers):
        records.append(
            build_provider_record(
                rng=rng,
                provider_config=provider_config,
                reference_data=reference_data,
                sequence_number=sequence_number,
                organization_row=organization_assignments[sequence_number],
            )
        )

    dataframe = pd.DataFrame(records)

    logger.info("Generated Provider dataset. Actual rows: %s", len(dataframe))

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible validation rules from providers.yaml.

    Parameters
    ----------
    provider_config:
        Parsed providers.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        provider_config,
        "validation",
        "providers.yaml",
    )

    required_columns = ["provider_id"]
    no_null_columns = ["provider_id"]

    if bool(validation_config.get("require_unique_npi", False)):
        required_columns.append("npi")
        no_null_columns.append("npi")

    if bool(validation_config.get("require_provider_type", False)):
        required_columns.append("provider_type")
        no_null_columns.append("provider_type")

    if bool(validation_config.get("require_specialty", False)):
        required_columns.append("specialty_code")
        no_null_columns.append("specialty_code")

    if bool(validation_config.get("require_taxonomy_code", False)):
        required_columns.append("taxonomy_code")
        no_null_columns.append("taxonomy_code")

    if bool(validation_config.get("require_zip_code", False)):
        required_columns.append("zip_code")
        no_null_columns.append("zip_code")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_provider_ids", False)):
        validation_rules["primary_key"] = ["provider_id"]

    return validation_rules


def validate_provider_dataset(
    context,
    dataframe: pd.DataFrame,
    provider_config: Dict[str, Any],
) -> None:
    """
    Validate generated Provider dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Provider DataFrame.

    provider_config:
        Parsed providers.yaml.
    """
    validation_rules = build_validation_rules(provider_config)

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=validation_rules,
        dataset_name="raw.providers",
    )

    validation_config = require_config_section(
        provider_config,
        "validation",
        "providers.yaml",
    )

    if bool(validation_config.get("require_unique_npi", False)):
        context.validation.validate_duplicates(
            dataframe=dataframe,
            key_columns=["npi"],
            dataset_name="raw.providers",
        )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_provider_output_path(provider_config: Dict[str, Any]) -> str:
    """
    Resolve Provider output path from providers.yaml.

    Parameters
    ----------
    provider_config:
        Parsed providers.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(provider_config, "output", "providers.yaml")

    file_name = require_config_value(
        output_config,
        "file_name",
        "providers.output",
    )

    return f"data/raw/{file_name}"


def write_provider_dataset(
    context,
    dataframe: pd.DataFrame,
    provider_config: Dict[str, Any],
) -> str:
    """
    Write Provider dataset to configured output location.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Provider DataFrame.

    provider_config:
        Parsed providers.yaml.

    Returns
    -------
    str
        Written output path.
    """
    output_path = resolve_provider_output_path(provider_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.providers",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_provider_metadata(
    context,
    dataframe: pd.DataFrame,
    provider_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Provider metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Provider DataFrame.

    provider_config:
        Parsed providers.yaml.

    output_path:
        Provider output path.
    """
    metadata_config = require_config_section(provider_config, "metadata", "providers.yaml")

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.providers",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="provider",
            primary_key=["provider_id"],
            description="Canonical MedFabric synthetic Provider dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/providers_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.providers",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/providers_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.providers",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/providers_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.providers",
            source_datasets=[
                "reference.provider_specialty",
                "reference.provider_taxonomy",
                "reference.provider_organization",
                "reference.primary_care_specialty",
                "reference.facility",
                "reference.geography",
            ],
            output_dataset="raw.providers",
            transformation_name="generate_provider_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/providers_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_provider_generation(context) -> pd.DataFrame:
    """
    Execute complete Provider generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Provider dataset.
    """
    global_config = load_global_generation_config(context)
    provider_config = load_provider_generation_config(context)

    reference_data = load_provider_reference_data(
        context=context,
        provider_config=provider_config,
    )

    provider_dataframe = generate_provider_dataset(
        context=context,
        global_config=global_config,
        provider_config=provider_config,
        reference_data=reference_data,
    )

    validate_provider_dataset(
        context=context,
        dataframe=provider_dataframe,
        provider_config=provider_config,
    )

    output_path = write_provider_dataset(
        context=context,
        dataframe=provider_dataframe,
        provider_config=provider_config,
    )

    write_provider_metadata(
        context=context,
        dataframe=provider_dataframe,
        provider_config=provider_config,
        output_path=output_path,
    )

    return provider_dataframe


def main() -> None:
    """
    Main entry point for Provider generation.

    Run Command
    -----------
    python -m src.data_generation.generators.provider_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        provider_dataframe = run_provider_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Provider generation completed successfully. Rows: %s",
            len(provider_dataframe),
        )

        print("MedFabric provider generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Provider generation failed.")
        logger.exception("Provider generation failed.")
        raise PipelineError("Provider generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()