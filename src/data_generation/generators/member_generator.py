###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/member_generator.py
#
# Purpose:
#     Generates the canonical Member dataset used by the MedFabric Synthetic Data
#     Engine.
#
# Business Context:
#     The Member object is the foundational person-level entity in MedFabric.
#     Enrollment, encounters, claims, pharmacy events, laboratory results, SDOH
#     records, provider attribution, care management, registries, risk models,
#     Member 360 analytics, and payer reporting all depend on a complete,
#     consistent, reproducible, and validated Member dataset.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/members.yaml
#     reference/demographics/first_names.parquet
#     reference/demographics/last_names.parquet
#     reference/demographics/race_reference.parquet
#     reference/demographics/ethnicity_reference.parquet
#     reference/demographics/language_reference.parquet
#     reference/geography/us_geography_reference.parquet
#
# Outputs:
#     data/raw/members.parquet
#     data/metadata/members_dataset_metadata.json
#     data/metadata/members_column_metadata.csv
#     data/metadata/members_statistics.json
#     data/metadata/members_lineage.json
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
#     5. Do not place business values in Python.
#     6. Dataset-specific settings come from config/data_generation/members.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.member_generator
#
# Expected Output:
#     Canonical Member dataset and metadata written to configured output paths.
###############################################################################

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.member"
STEP_NAME = "Generate Member Dataset"


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


def load_member_generation_config(context) -> Dict[str, Any]:
    """
    Load Member-specific generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/members.yaml.

    Notes
    -----
    This file intentionally returns the full members.yaml content because the
    approved configuration does not wrap settings inside a top-level `member:`
    section.
    """
    return context.configuration.load_yaml("data_generation/members.yaml")


###############################################################################
# REFERENCE DATA LOADING
###############################################################################


def load_member_reference_data(
    context,
    member_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load reference datasets required for Member generation.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    member_config:
        Parsed config/data_generation/members.yaml.

    Returns
    -------
    dict
        Dictionary of loaded reference DataFrames.

    Notes
    -----
    Reference paths come from members.yaml because this is the Member-specific
    dataset configuration file.
    """
    logger = context.get_logger(MODULE_NAME)

    demographics_config = require_config_section(
        member_config,
        "demographics",
        "config/data_generation/members.yaml",
    )

    reference_paths = {
        "first_names": require_config_value(
            demographics_config,
            "first_name_source",
            "members.demographics",
        ),
        "last_names": require_config_value(
            demographics_config,
            "last_name_source",
            "members.demographics",
        ),
        "race": require_config_value(
            demographics_config,
            "race_source",
            "members.demographics",
        ),
        "ethnicity": require_config_value(
            demographics_config,
            "ethnicity_source",
            "members.demographics",
        ),
        "language": require_config_value(
            demographics_config,
            "language_source",
            "members.demographics",
        ),
        "geography": require_config_value(
            demographics_config,
            "geography_source",
            "members.demographics",
        ),
    }

    reference_data: Dict[str, pd.DataFrame] = {}

    for reference_name, reference_path in reference_paths.items():
        dataframe = context.storage.read_parquet(reference_path)

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
            "Loaded Member reference data: %s | Rows: %s | Path: %s",
            reference_name,
            len(dataframe),
            reference_path,
        )

    return reference_data


###############################################################################
# SAMPLING HELPERS
###############################################################################


def normalize_weights(weights: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Normalize configured or reference-provided weights.

    Parameters
    ----------
    weights:
        Numeric weight array.

    dataset_name:
        Name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Normalized probability array.
    """
    total_weight = float(weights.sum())

    if total_weight <= 0:
        raise PipelineError(f"Weights must sum to a positive value for {dataset_name}.")

    return weights / total_weight


def sample_from_weighted_dataframe(
    rng: np.random.Generator,
    dataframe: pd.DataFrame,
    value_column: str,
    weight_column: str,
    dataset_name: str,
) -> Any:
    """
    Sample one value from a DataFrame using a weight column when available.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    dataframe:
        Reference DataFrame.

    value_column:
        Column to sample.

    weight_column:
        Weight column expected in reference data.

    dataset_name:
        Dataset name used in errors.

    Returns
    -------
    Any
        Sampled value.
    """
    if value_column not in dataframe.columns:
        raise PipelineError(
            f"Column '{value_column}' was not found in reference dataset {dataset_name}."
        )

    if weight_column in dataframe.columns:
        sample_frame = dataframe[[value_column, weight_column]].dropna()

        if sample_frame.empty:
            raise PipelineError(f"No valid weighted rows found in {dataset_name}.")

        values = sample_frame[value_column].to_numpy()
        weights = sample_frame[weight_column].astype(float).to_numpy()
        probabilities = normalize_weights(weights, dataset_name)

        return rng.choice(values, p=probabilities)

    values = dataframe[value_column].dropna().to_numpy()

    if len(values) == 0:
        raise PipelineError(f"No valid values found in {dataset_name}.{value_column}.")

    return rng.choice(values)


def sample_weighted_label(
    rng: np.random.Generator,
    weights_by_label: Dict[str, float],
    rule_name: str,
) -> str:
    """
    Sample a configured label from a dictionary of label weights.

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
    if not weights_by_label:
        raise PipelineError(f"No weights configured for {rule_name}.")

    labels = list(weights_by_label.keys())
    weights = np.array([float(weights_by_label[label]) for label in labels])
    probabilities = normalize_weights(weights, rule_name)

    return str(rng.choice(labels, p=probabilities))


def sample_weighted_geography_row(
    rng: np.random.Generator,
    geography_reference: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Sample one geography row.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    geography_reference:
        Geography reference DataFrame.

    Returns
    -------
    dict
        Sampled geography row.
    """
    if geography_reference.empty:
        raise PipelineError("Geography reference dataset is empty.")

    if "population_weight" in geography_reference.columns:
        weights = geography_reference["population_weight"].astype(float).to_numpy()
        probabilities = normalize_weights(weights, "reference.geography")
        selected_index = rng.choice(geography_reference.index.to_numpy(), p=probabilities)

        return geography_reference.loc[selected_index].to_dict()

    selected_position = int(rng.integers(0, len(geography_reference)))

    return geography_reference.iloc[selected_position].to_dict()


###############################################################################
# MEMBER BUSINESS GENERATION HELPERS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    member_config: Dict[str, Any],
) -> int:
    """
    Resolve the random seed for reproducible Member generation.

    Parameters
    ----------
    global_config:
        Parsed generation.yaml.

    member_config:
        Parsed members.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        member_config,
        "reproducibility",
        "members.yaml",
    )

    use_global_random_seed = bool(
        require_config_value(
            reproducibility_config,
            "use_global_random_seed",
            "members.reproducibility",
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
            "members.reproducibility",
        )
    )


def build_member_id(member_config: Dict[str, Any], sequence_number: int) -> str:
    """
    Build a configured Member identifier.

    Parameters
    ----------
    member_config:
        Parsed members.yaml.

    sequence_number:
        Zero-based sequence number.

    Returns
    -------
    str
        Generated member identifier.
    """
    identifier_config = require_config_section(
        member_config,
        "member_identifier",
        "members.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "members.member_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "members.member_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "members.member_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def build_household_id(
    member_config: Dict[str, Any],
    household_sequence_number: int,
) -> str:
    """
    Build a configured Household identifier.

    Parameters
    ----------
    member_config:
        Parsed members.yaml.

    household_sequence_number:
        Zero-based household sequence number.

    Returns
    -------
    str
        Generated household identifier.
    """
    household_identifier_config = require_config_section(
        member_config,
        "household_identifier",
        "members.yaml",
    )

    prefix = require_config_value(
        household_identifier_config,
        "prefix",
        "members.household_identifier",
    )

    starting_sequence = int(
        require_config_value(
            household_identifier_config,
            "starting_sequence",
            "members.household_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            household_identifier_config,
            "padding_length",
            "members.household_identifier",
        )
    )

    numeric_value = starting_sequence + household_sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def build_household_assignments(
    rng: np.random.Generator,
    member_config: Dict[str, Any],
    total_members: int,
) -> List[str]:
    """
    Build household IDs for generated members.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    member_config:
        Parsed members.yaml.

    total_members:
        Number of members to generate.

    Returns
    -------
    list[str]
        Household ID assigned to each generated member.

    Notes
    -----
    If household generation is disabled, each member receives a unique household.
    If household generation is enabled, households are created from configured
    average household size.
    """
    population_config = require_config_section(
        member_config,
        "member_population",
        "members.yaml",
    )

    generate_family_households = bool(
        require_config_value(
            population_config,
            "generate_family_households",
            "members.member_population",
        )
    )

    if not generate_family_households:
        return [
            build_household_id(member_config, household_sequence_number)
            for household_sequence_number in range(total_members)
        ]

    average_household_size = float(
        require_config_value(
            population_config,
            "average_household_size",
            "members.member_population",
        )
    )

    if average_household_size <= 0:
        raise PipelineError("average_household_size must be greater than zero.")

    household_assignments: List[str] = []
    household_sequence_number = 0

    while len(household_assignments) < total_members:
        household_id = build_household_id(member_config, household_sequence_number)

        household_size = max(
            1,
            int(rng.poisson(average_household_size)),
        )

        for _ in range(household_size):
            if len(household_assignments) < total_members:
                household_assignments.append(household_id)

        household_sequence_number += 1

    return household_assignments


def sample_age(
    rng: np.random.Generator,
    member_config: Dict[str, Any],
) -> int:
    """
    Generate a member age from configured age distribution.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    member_config:
        Parsed members.yaml.

    Returns
    -------
    int
        Generated age.
    """
    demographics_config = require_config_section(
        member_config,
        "demographics",
        "members.yaml",
    )

    minimum_age = int(
        require_config_value(
            demographics_config,
            "minimum_age",
            "members.demographics",
        )
    )

    maximum_age = int(
        require_config_value(
            demographics_config,
            "maximum_age",
            "members.demographics",
        )
    )

    if minimum_age > maximum_age:
        raise PipelineError("minimum_age cannot be greater than maximum_age.")

    age_distribution = require_config_section(
        demographics_config,
        "age_distribution",
        "members.demographics",
    )

    age_band = sample_weighted_label(
        rng=rng,
        weights_by_label=age_distribution,
        rule_name="members.demographics.age_distribution",
    )

    if age_band == "pediatric":
        lower_bound = minimum_age
        upper_bound = min(maximum_age, 17)
    elif age_band == "senior":
        lower_bound = max(minimum_age, 65)
        upper_bound = maximum_age
    else:
        lower_bound = max(minimum_age, 18)
        upper_bound = min(maximum_age, 64)

    if lower_bound > upper_bound:
        lower_bound = minimum_age
        upper_bound = maximum_age

    return int(rng.integers(lower_bound, upper_bound + 1))


def build_birth_date(
    rng: np.random.Generator,
    age: int,
) -> date:
    """
    Build a birth date from generated age.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    age:
        Generated member age.

    Returns
    -------
    datetime.date
        Generated birth date.
    """
    current_date = pd.Timestamp.today().normalize()
    random_day_offset = int(rng.integers(0, 365))

    birth_timestamp = (
        current_date
        - pd.DateOffset(years=int(age))
        - pd.DateOffset(days=random_day_offset)
    )

    return birth_timestamp.date()


def sample_gender(
    rng: np.random.Generator,
    member_config: Dict[str, Any],
) -> str:
    """
    Sample gender from configured gender weights.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    member_config:
        Parsed members.yaml.

    Returns
    -------
    str
        Generated gender label.
    """
    gender_config = require_config_section(member_config, "gender", "members.yaml")

    return sample_weighted_label(
        rng=rng,
        weights_by_label=gender_config,
        rule_name="members.gender",
    )


def sample_first_name(
    rng: np.random.Generator,
    first_name_reference: pd.DataFrame,
    gender: str,
) -> str:
    """
    Sample first name from first name reference data.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    first_name_reference:
        First name reference DataFrame.

    gender:
        Generated member gender.

    Returns
    -------
    str
        Sampled first name.
    """
    candidate_reference = first_name_reference

    if "gender_hint" in first_name_reference.columns:
        filtered = first_name_reference[
            first_name_reference["gender_hint"].astype(str).str.lower()
            == str(gender).lower()
        ]

        if not filtered.empty:
            candidate_reference = filtered

    return str(
        sample_from_weighted_dataframe(
            rng=rng,
            dataframe=candidate_reference,
            value_column="first_name",
            weight_column="selection_weight",
            dataset_name="reference.first_names",
        )
    )


def sample_last_name(
    rng: np.random.Generator,
    last_name_reference: pd.DataFrame,
) -> str:
    """
    Sample last name from last name reference data.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    last_name_reference:
        Last name reference DataFrame.

    Returns
    -------
    str
        Sampled last name.
    """
    return str(
        sample_from_weighted_dataframe(
            rng=rng,
            dataframe=last_name_reference,
            value_column="last_name",
            weight_column="selection_weight",
            dataset_name="reference.last_names",
        )
    )


def sample_race(
    rng: np.random.Generator,
    race_reference: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Sample race from race reference data.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    race_reference:
        Race reference DataFrame.

    Returns
    -------
    dict
        Race code and description.
    """
    race_code = sample_from_weighted_dataframe(
        rng=rng,
        dataframe=race_reference,
        value_column="race_code",
        weight_column="selection_weight",
        dataset_name="reference.race",
    )

    matched_rows = race_reference[race_reference["race_code"] == race_code]

    race_description = (
        matched_rows.iloc[0]["race_description"]
        if "race_description" in race_reference.columns and not matched_rows.empty
        else race_code
    )

    return {
        "race_code": race_code,
        "race_description": race_description,
    }


def sample_ethnicity(
    rng: np.random.Generator,
    ethnicity_reference: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Sample ethnicity from ethnicity reference data.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    ethnicity_reference:
        Ethnicity reference DataFrame.

    Returns
    -------
    dict
        Ethnicity code and description.
    """
    ethnicity_code = sample_from_weighted_dataframe(
        rng=rng,
        dataframe=ethnicity_reference,
        value_column="ethnicity_code",
        weight_column="selection_weight",
        dataset_name="reference.ethnicity",
    )

    matched_rows = ethnicity_reference[
        ethnicity_reference["ethnicity_code"] == ethnicity_code
    ]

    ethnicity_description = (
        matched_rows.iloc[0]["ethnicity_description"]
        if "ethnicity_description" in ethnicity_reference.columns and not matched_rows.empty
        else ethnicity_code
    )

    return {
        "ethnicity_code": ethnicity_code,
        "ethnicity_description": ethnicity_description,
    }


def sample_language(
    rng: np.random.Generator,
    language_reference: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Sample preferred language from language reference data.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    language_reference:
        Language reference DataFrame.

    Returns
    -------
    dict
        Language code and description.
    """
    language_code = sample_from_weighted_dataframe(
        rng=rng,
        dataframe=language_reference,
        value_column="language_code",
        weight_column="selection_weight",
        dataset_name="reference.language",
    )

    matched_rows = language_reference[
        language_reference["language_code"] == language_code
    ]

    language_description = (
        matched_rows.iloc[0]["language_description"]
        if "language_description" in language_reference.columns and not matched_rows.empty
        else language_code
    )

    return {
        "preferred_language_code": language_code,
        "preferred_language_description": language_description,
    }


def build_phone_number(
    rng: np.random.Generator,
    enabled: bool,
) -> str | None:
    """
    Generate a synthetic phone number when enabled.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    enabled:
        Whether phone number generation is enabled.

    Returns
    -------
    str or None
        Generated phone number.
    """
    if not enabled:
        return None

    area_code = int(rng.integers(200, 1000))
    prefix = int(rng.integers(200, 1000))
    line_number = int(rng.integers(0, 10000))

    return f"{area_code}-{prefix}-{line_number:04d}"


def build_email_address(
    enabled: bool,
    first_name: str,
    last_name: str,
    member_id: str,
) -> str | None:
    """
    Generate a synthetic email address when enabled.

    Parameters
    ----------
    enabled:
        Whether email generation is enabled.

    first_name:
        Generated first name.

    last_name:
        Generated last name.

    member_id:
        Generated member ID.

    Returns
    -------
    str or None
        Generated email address.
    """
    if not enabled:
        return None

    normalized_first_name = str(first_name).lower().replace(" ", "")
    normalized_last_name = str(last_name).lower().replace(" ", "")
    normalized_member_id = str(member_id).lower()

    return f"{normalized_first_name}.{normalized_last_name}.{normalized_member_id}@medfabric.example"


def build_member_record(
    rng: np.random.Generator,
    member_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
    sequence_number: int,
    household_id: str,
) -> Dict[str, Any]:
    """
    Build a single canonical Member record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    member_config:
        Parsed members.yaml.

    reference_data:
        Loaded reference DataFrames.

    sequence_number:
        Zero-based member sequence number.

    household_id:
        Household identifier assigned to this member.

    Returns
    -------
    dict
        Generated canonical Member record.
    """
    contact_config = require_config_section(
        member_config,
        "contact_information",
        "members.yaml",
    )

    member_id = build_member_id(member_config, sequence_number)
    gender = sample_gender(rng, member_config)
    age = sample_age(rng, member_config)
    birth_date = build_birth_date(rng, age)

    first_name = sample_first_name(
        rng=rng,
        first_name_reference=reference_data["first_names"],
        gender=gender,
    )

    last_name = sample_last_name(
        rng=rng,
        last_name_reference=reference_data["last_names"],
    )

    race = sample_race(
        rng=rng,
        race_reference=reference_data["race"],
    )

    ethnicity = sample_ethnicity(
        rng=rng,
        ethnicity_reference=reference_data["ethnicity"],
    )

    language = sample_language(
        rng=rng,
        language_reference=reference_data["language"],
    )

    geography = sample_weighted_geography_row(
        rng=rng,
        geography_reference=reference_data["geography"],
    )

    generate_phone_numbers = bool(
        require_config_value(
            contact_config,
            "generate_phone_numbers",
            "members.contact_information",
        )
    )

    generate_email_addresses = bool(
        require_config_value(
            contact_config,
            "generate_email_addresses",
            "members.contact_information",
        )
    )

    return {
        "member_id": member_id,
        "household_id": household_id,
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
        "gender": gender,
        "birth_date": birth_date,
        "age": age,
        "race_code": race["race_code"],
        "race_description": race["race_description"],
        "ethnicity_code": ethnicity["ethnicity_code"],
        "ethnicity_description": ethnicity["ethnicity_description"],
        "preferred_language_code": language["preferred_language_code"],
        "preferred_language_description": language["preferred_language_description"],
        "zip_code": geography.get("zip_code"),
        "city": geography.get("city"),
        "county": geography.get("county_name") or geography.get("county"),
        "state_code": geography.get("state_code") or geography.get("state"),
        "state_name": geography.get("state_name"),
        "region": geography.get("region"),
        "phone_number": build_phone_number(
            rng=rng,
            enabled=generate_phone_numbers,
        ),
        "email_address": build_email_address(
            enabled=generate_email_addresses,
            first_name=first_name,
            last_name=last_name,
            member_id=member_id,
        ),
        "source_system": "MedFabric Synthetic Data Engine",
        "record_status": "Active",
        "created_at": pd.Timestamp.today().normalize(),
        "updated_at": pd.Timestamp.today().normalize(),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_member_dataset(
    context,
    global_config: Dict[str, Any],
    member_config: Dict[str, Any],
    reference_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate the complete canonical Member dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed generation.yaml.

    member_config:
        Parsed members.yaml.

    reference_data:
        Loaded reference DataFrames.

    Returns
    -------
    pandas.DataFrame
        Generated canonical Member dataset.
    """
    logger = context.get_logger(MODULE_NAME)

    population_config = require_config_section(
        member_config,
        "member_population",
        "members.yaml",
    )

    total_members = int(
        require_config_value(
            population_config,
            "total_members",
            "members.member_population",
        )
    )

    random_seed = resolve_random_seed(
        global_config=global_config,
        member_config=member_config,
    )

    rng = np.random.default_rng(random_seed)

    household_assignments = build_household_assignments(
        rng=rng,
        member_config=member_config,
        total_members=total_members,
    )

    logger.info("Generating Member dataset. Expected rows: %s", total_members)

    records: List[Dict[str, Any]] = []

    for sequence_number in range(total_members):
        records.append(
            build_member_record(
                rng=rng,
                member_config=member_config,
                reference_data=reference_data,
                sequence_number=sequence_number,
                household_id=household_assignments[sequence_number],
            )
        )

    dataframe = pd.DataFrame(records)

    logger.info("Generated Member dataset. Actual rows: %s", len(dataframe))

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(member_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible validation rules from members.yaml.

    Parameters
    ----------
    member_config:
        Parsed members.yaml.

    Returns
    -------
    dict
        Validation rules for context.validation.validate_dataset().
    """
    validation_config = require_config_section(
        member_config,
        "validation",
        "members.yaml",
    )

    required_columns = ["member_id", "household_id"]

    no_null_columns = ["member_id", "household_id"]

    if bool(validation_config.get("require_birth_date", False)):
        required_columns.append("birth_date")
        no_null_columns.append("birth_date")

    if bool(validation_config.get("require_gender", False)):
        required_columns.append("gender")
        no_null_columns.append("gender")

    if bool(validation_config.get("require_zip_code", False)):
        required_columns.append("zip_code")
        no_null_columns.append("zip_code")

    if not bool(validation_config.get("allow_missing_email", True)):
        required_columns.append("email_address")
        no_null_columns.append("email_address")

    if not bool(validation_config.get("allow_missing_phone", True)):
        required_columns.append("phone_number")
        no_null_columns.append("phone_number")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_member_ids", False)):
        validation_rules["primary_key"] = ["member_id"]

    return validation_rules


def validate_member_dataset(
    context,
    dataframe: pd.DataFrame,
    member_config: Dict[str, Any],
) -> None:
    """
    Validate generated Member dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Member DataFrame.

    member_config:
        Parsed members.yaml.
    """
    validation_rules = build_validation_rules(member_config)

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=validation_rules,
        dataset_name="raw.members",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_member_output_path(member_config: Dict[str, Any]) -> str:
    """
    Resolve Member output path from members.yaml.

    Parameters
    ----------
    member_config:
        Parsed members.yaml.

    Returns
    -------
    str
        Output path for generated Member dataset.
    """
    output_config = require_config_section(member_config, "output", "members.yaml")

    file_name = require_config_value(
        output_config,
        "file_name",
        "members.output",
    )

    return f"data/raw/{file_name}"


def write_member_dataset(
    context,
    dataframe: pd.DataFrame,
    member_config: Dict[str, Any],
) -> str:
    """
    Write Member dataset to configured output location.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Member DataFrame.

    member_config:
        Parsed members.yaml.

    Returns
    -------
    str
        Written output path.
    """
    output_path = resolve_member_output_path(member_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.members",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_member_metadata(
    context,
    dataframe: pd.DataFrame,
    member_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Member metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Member DataFrame.

    member_config:
        Parsed members.yaml.

    output_path:
        Member dataset output path.
    """
    metadata_config = require_config_section(member_config, "metadata", "members.yaml")

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.members",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="member",
            primary_key=["member_id"],
            description="Canonical MedFabric synthetic Member dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/members_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.members",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/members_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.members",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/members_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.members",
            source_datasets=[
                "reference.first_names",
                "reference.last_names",
                "reference.race",
                "reference.ethnicity",
                "reference.language",
                "reference.geography",
            ],
            output_dataset="raw.members",
            transformation_name="generate_member_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/members_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_member_generation(context) -> pd.DataFrame:
    """
    Execute complete Member generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Member dataset.
    """
    global_config = load_global_generation_config(context)
    member_config = load_member_generation_config(context)

    reference_data = load_member_reference_data(
        context=context,
        member_config=member_config,
    )

    member_dataframe = generate_member_dataset(
        context=context,
        global_config=global_config,
        member_config=member_config,
        reference_data=reference_data,
    )

    validate_member_dataset(
        context=context,
        dataframe=member_dataframe,
        member_config=member_config,
    )

    output_path = write_member_dataset(
        context=context,
        dataframe=member_dataframe,
        member_config=member_config,
    )

    write_member_metadata(
        context=context,
        dataframe=member_dataframe,
        member_config=member_config,
        output_path=output_path,
    )

    return member_dataframe


def main() -> None:
    """
    Main entry point for Member generation.

    Run Command
    -----------
    python -m src.data_generation.generators.member_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        member_dataframe = run_member_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Member generation completed successfully. Rows: %s",
            len(member_dataframe),
        )

        print("MedFabric member generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Member generation failed.")
        logger.exception("Member generation failed.")
        raise PipelineError("Member generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()