###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/sdoh_generator.py
#
# Purpose:
#     Generates the canonical Social Determinants of Health (SDOH) dataset used
#     by the MedFabric Synthetic Data Engine.
#
# Business Context:
#     SDOH data captures non-clinical member risk factors such as income,
#     education, employment, housing, food security, transportation, social
#     support, digital access, and language barriers. These factors are critical
#     for population health, health equity analytics, care management, risk
#     stratification, and Member 360.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/sdoh.yaml
#     data/raw/members.parquet
#     reference/sdoh/income_reference.parquet
#     reference/sdoh/education_reference.parquet
#     reference/sdoh/employment_reference.parquet
#     reference/sdoh/housing_reference.parquet
#     reference/sdoh/food_security_reference.parquet
#     reference/sdoh/transportation_reference.parquet
#     reference/sdoh/social_support_reference.parquet
#     reference/sdoh/digital_access_reference.parquet
#     reference/sdoh/language_barrier_reference.parquet
#
# Outputs:
#     data/raw/sdoh.parquet
#     data/metadata/sdoh_dataset_metadata.json
#     data/metadata/sdoh_column_metadata.csv
#     data/metadata/sdoh_statistics.json
#     data/metadata/sdoh_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/sdoh.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.sdoh_generator
#
# Expected Output:
#     Canonical SDOH dataset and metadata written to configured output paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.sdoh"
STEP_NAME = "Generate SDOH Dataset"


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


def load_sdoh_generation_config(context) -> Dict[str, Any]:
    """
    Load SDOH-specific synthetic data generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/sdoh.yaml.
    """
    return context.configuration.load_yaml("data_generation/sdoh.yaml")


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


def sample_dataframe_value(
    rng: np.random.Generator,
    dataframe: pd.DataFrame,
    dataset_name: str,
    value_column: str = "category",
    weight_column: str = "selection_weight",
) -> Any:
    """
    Sample one value from a reference DataFrame.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    dataframe:
        Source reference DataFrame.

    dataset_name:
        Logical dataset name used in error messages.

    value_column:
        Column containing the value to sample.

    weight_column:
        Optional weight column. If present, weighted sampling is used.

    Returns
    -------
    Any
        Sampled reference value.

    Raises
    ------
    PipelineError
        Raised when the reference DataFrame is empty or required value column is
        missing.
    """
    if dataframe.empty:
        raise PipelineError(f"Dataset is empty: {dataset_name}")

    if value_column not in dataframe.columns:
        raise PipelineError(
            f"Column '{value_column}' not found in reference dataset {dataset_name}."
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


###############################################################################
# INPUT AND REFERENCE LOADING
###############################################################################


def load_input_and_reference_data(
    context,
    sdoh_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load Member input and SDOH reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    dict
        Dictionary of loaded input and reference DataFrames.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(sdoh_config, "inputs", "sdoh.yaml")

    domain_sections = [
        "income",
        "education",
        "employment",
        "housing",
        "food_security",
        "transportation",
        "social_support",
        "digital_access",
        "language_barriers",
    ]

    paths = {
        "members": require_config_value(
            inputs_config,
            "members_dataset",
            "sdoh.inputs",
        )
    }

    for domain_name in domain_sections:
        domain_config = require_config_section(sdoh_config, domain_name, "sdoh.yaml")
        paths[domain_name] = require_config_value(
            domain_config,
            "source",
            f"sdoh.{domain_name}",
        )

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
            "Loaded SDOH input/reference: %s | Rows: %s | Path: %s",
            dataset_name,
            len(dataframe),
            path,
        )

    return datasets


###############################################################################
# IDENTIFIERS AND DATES
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    sdoh_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for reproducible SDOH generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        sdoh_config,
        "reproducibility",
        "sdoh.yaml",
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
            "sdoh.reproducibility",
        )
    )


def build_sdoh_id(
    sdoh_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build configured SDOH identifier.

    Parameters
    ----------
    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    sequence_number:
        Zero-based SDOH sequence number.

    Returns
    -------
    str
        Generated SDOH identifier.
    """
    identifier_config = require_config_section(
        sdoh_config,
        "sdoh_identifier",
        "sdoh.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "sdoh.sdoh_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "sdoh.sdoh_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "sdoh.sdoh_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def sample_assessment_date(
    rng: np.random.Generator,
    sdoh_config: Dict[str, Any],
) -> pd.Timestamp:
    """
    Generate synthetic SDOH assessment date.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    pandas.Timestamp
        Generated assessment date.
    """
    assessment_config = require_config_section(
        sdoh_config,
        "assessment_dates",
        "sdoh.yaml",
    )

    start_year = int(
        require_config_value(
            assessment_config,
            "assessment_start_year",
            "sdoh.assessment_dates",
        )
    )

    end_year = int(
        require_config_value(
            assessment_config,
            "assessment_end_year",
            "sdoh.assessment_dates",
        )
    )

    if start_year > end_year:
        raise PipelineError(
            "assessment_start_year cannot be greater than assessment_end_year."
        )

    assessment_year = int(rng.integers(start_year, end_year + 1))
    assessment_month = int(rng.integers(1, 13))
    assessment_day = int(rng.integers(1, 29))

    return pd.Timestamp(
        year=assessment_year,
        month=assessment_month,
        day=assessment_day,
    )


###############################################################################
# SDOH RISK SCORING
###############################################################################


def score_domain_risk(value: Any) -> int:
    """
    Score one SDOH domain value.

    Parameters
    ----------
    value:
        Sampled SDOH category value.

    Returns
    -------
    int
        Risk score contribution.

    Processing Notes
    ----------------
    SDOH reference values are categorical. This scoring function converts common
    adverse categories into a simple risk score. Later versions can move scoring
    rules into YAML or dedicated risk models.
    """
    normalized_value = str(value).strip().lower()

    high_risk_terms = [
        "low",
        "less than high school",
        "unemployed",
        "disabled",
        "unstable",
        "homeless risk",
        "insecure",
        "barrier",
        "limited",
        "limited access",
        "high",
    ]

    moderate_risk_terms = [
        "moderate",
        "at risk",
        "mobile only",
        "some college",
    ]

    if normalized_value in high_risk_terms:
        return 2

    if normalized_value in moderate_risk_terms:
        return 1

    return 0


def build_sdoh_risk_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build aggregate SDOH risk summary fields.

    Parameters
    ----------
    record:
        SDOH record dictionary.

    Returns
    -------
    dict
        Risk score and risk band.
    """
    domain_columns = [
        "income_category",
        "education_category",
        "employment_category",
        "housing_category",
        "food_security_category",
        "transportation_category",
        "social_support_category",
        "digital_access_category",
        "language_barrier_category",
    ]

    score = sum(score_domain_risk(record.get(column)) for column in domain_columns)

    if score >= 8:
        risk_band = "High"
    elif score >= 4:
        risk_band = "Medium"
    else:
        risk_band = "Low"

    return {
        "sdoh_risk_score": score,
        "sdoh_risk_band": risk_band,
    }


###############################################################################
# SDOH RECORD GENERATION
###############################################################################


def determine_assessment_count(
    rng: np.random.Generator,
    sdoh_config: Dict[str, Any],
) -> int:
    """
    Determine how many SDOH assessments to generate for one member.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    int
        Number of assessments for one member.
    """
    population_config = require_config_section(
        sdoh_config,
        "sdoh_population",
        "sdoh.yaml",
    )

    generate_record_for_every_member = bool(
        require_config_value(
            population_config,
            "generate_record_for_every_member",
            "sdoh.sdoh_population",
        )
    )

    maximum_assessments = int(
        require_config_value(
            population_config,
            "maximum_assessments_per_member",
            "sdoh.sdoh_population",
        )
    )

    if maximum_assessments <= 0:
        raise PipelineError("maximum_assessments_per_member must be positive.")

    if generate_record_for_every_member:
        return int(rng.integers(1, maximum_assessments + 1))

    return int(rng.integers(0, maximum_assessments + 1))


def build_sdoh_record(
    rng: np.random.Generator,
    sdoh_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    member_row: Dict[str, Any],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical SDOH record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    datasets:
        Loaded input and reference datasets.

    member_row:
        Member row dictionary.

    sequence_number:
        Zero-based SDOH sequence number.

    Returns
    -------
    dict
        Generated SDOH record.
    """
    audit_config = require_config_section(sdoh_config, "audit", "sdoh.yaml")

    record = {
        "sdoh_id": build_sdoh_id(sdoh_config, sequence_number),
        "member_id": member_row.get("member_id"),
        "assessment_date": sample_assessment_date(rng, sdoh_config).date(),
        "income_category": sample_dataframe_value(
            rng,
            datasets["income"],
            "reference.sdoh.income",
        ),
        "education_category": sample_dataframe_value(
            rng,
            datasets["education"],
            "reference.sdoh.education",
        ),
        "employment_category": sample_dataframe_value(
            rng,
            datasets["employment"],
            "reference.sdoh.employment",
        ),
        "housing_category": sample_dataframe_value(
            rng,
            datasets["housing"],
            "reference.sdoh.housing",
        ),
        "food_security_category": sample_dataframe_value(
            rng,
            datasets["food_security"],
            "reference.sdoh.food_security",
        ),
        "transportation_category": sample_dataframe_value(
            rng,
            datasets["transportation"],
            "reference.sdoh.transportation",
        ),
        "social_support_category": sample_dataframe_value(
            rng,
            datasets["social_support"],
            "reference.sdoh.social_support",
        ),
        "digital_access_category": sample_dataframe_value(
            rng,
            datasets["digital_access"],
            "reference.sdoh.digital_access",
        ),
        "language_barrier_category": sample_dataframe_value(
            rng,
            datasets["language_barriers"],
            "reference.sdoh.language_barriers",
        ),
        "source_system": require_config_value(
            audit_config,
            "source_system",
            "sdoh.audit",
        ),
        "record_status": require_config_value(
            audit_config,
            "record_status",
            "sdoh.audit",
        ),
        "created_at": pd.Timestamp.today().normalize(),
        "updated_at": pd.Timestamp.today().normalize(),
    }

    record.update(build_sdoh_risk_summary(record))

    return record


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_sdoh_dataset(
    context,
    global_config: Dict[str, Any],
    sdoh_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate canonical SDOH dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    datasets:
        Loaded input and reference DataFrames.

    Returns
    -------
    pandas.DataFrame
        Generated SDOH DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    random_seed = resolve_random_seed(
        global_config=global_config,
        sdoh_config=sdoh_config,
    )

    rng = np.random.default_rng(random_seed)
    members_dataframe = datasets["members"]

    logger.info(
        "Generating SDOH records from Member dataset. Member rows: %s",
        len(members_dataframe),
    )

    records: List[Dict[str, Any]] = []
    sequence_number = 0

    for member_row in members_dataframe.to_dict("records"):
        assessment_count = determine_assessment_count(
            rng=rng,
            sdoh_config=sdoh_config,
        )

        for _ in range(assessment_count):
            records.append(
                build_sdoh_record(
                    rng=rng,
                    sdoh_config=sdoh_config,
                    datasets=datasets,
                    member_row=member_row,
                    sequence_number=sequence_number,
                )
            )

            sequence_number += 1

    dataframe = pd.DataFrame(records)

    logger.info(
        "Generated SDOH dataset. Rows: %s",
        len(dataframe),
    )

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(sdoh_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible validation rules.

    Parameters
    ----------
    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        sdoh_config,
        "validation",
        "sdoh.yaml",
    )

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_unique_sdoh_ids", False)):
        required_columns.append("sdoh_id")
        no_null_columns.append("sdoh_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_assessment_date", False)):
        required_columns.append("assessment_date")
        no_null_columns.append("assessment_date")

    if bool(validation_config.get("require_income", False)):
        required_columns.append("income_category")
        no_null_columns.append("income_category")

    if bool(validation_config.get("require_education", False)):
        required_columns.append("education_category")
        no_null_columns.append("education_category")

    if bool(validation_config.get("require_housing", False)):
        required_columns.append("housing_category")
        no_null_columns.append("housing_category")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_sdoh_ids", False)):
        validation_rules["primary_key"] = ["sdoh_id"]

    return validation_rules


def validate_sdoh_dataset(
    context,
    dataframe: pd.DataFrame,
    sdoh_config: Dict[str, Any],
) -> None:
    """
    Validate generated SDOH dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated SDOH DataFrame.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.
    """
    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=build_validation_rules(sdoh_config),
        dataset_name="raw.sdoh",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_output_path(sdoh_config: Dict[str, Any]) -> str:
    """
    Resolve SDOH output path.

    Parameters
    ----------
    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(sdoh_config, "output", "sdoh.yaml")

    file_name = require_config_value(
        output_config,
        "file_name",
        "sdoh.output",
    )

    return f"data/raw/{file_name}"


def write_sdoh_dataset(
    context,
    dataframe: pd.DataFrame,
    sdoh_config: Dict[str, Any],
) -> str:
    """
    Write SDOH dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated SDOH DataFrame.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    Returns
    -------
    str
        Written output path.
    """
    output_path = resolve_output_path(sdoh_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.sdoh",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_sdoh_metadata(
    context,
    dataframe: pd.DataFrame,
    sdoh_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write SDOH metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated SDOH DataFrame.

    sdoh_config:
        Parsed config/data_generation/sdoh.yaml.

    output_path:
        Written raw output path.
    """
    metadata_config = require_config_section(sdoh_config, "metadata", "sdoh.yaml")

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.sdoh",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="sdoh",
            primary_key=["sdoh_id"],
            description="Canonical MedFabric synthetic SDOH dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/sdoh_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.sdoh",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/sdoh_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.sdoh",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/sdoh_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.sdoh",
            source_datasets=[
                "raw.members",
                "reference.sdoh.income",
                "reference.sdoh.education",
                "reference.sdoh.employment",
                "reference.sdoh.housing",
                "reference.sdoh.food_security",
                "reference.sdoh.transportation",
                "reference.sdoh.social_support",
                "reference.sdoh.digital_access",
                "reference.sdoh.language_barriers",
            ],
            output_dataset="raw.sdoh",
            transformation_name="generate_sdoh_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/sdoh_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_sdoh_generation(context) -> pd.DataFrame:
    """
    Execute the complete SDOH generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated SDOH dataset.
    """
    global_config = load_global_generation_config(context)
    sdoh_config = load_sdoh_generation_config(context)

    datasets = load_input_and_reference_data(
        context=context,
        sdoh_config=sdoh_config,
    )

    sdoh_dataframe = generate_sdoh_dataset(
        context=context,
        global_config=global_config,
        sdoh_config=sdoh_config,
        datasets=datasets,
    )

    validate_sdoh_dataset(
        context=context,
        dataframe=sdoh_dataframe,
        sdoh_config=sdoh_config,
    )

    output_path = write_sdoh_dataset(
        context=context,
        dataframe=sdoh_dataframe,
        sdoh_config=sdoh_config,
    )

    write_sdoh_metadata(
        context=context,
        dataframe=sdoh_dataframe,
        sdoh_config=sdoh_config,
        output_path=output_path,
    )

    return sdoh_dataframe


def main() -> None:
    """
    Main entry point for SDOH generation.

    Run Command
    -----------
    python -m src.data_generation.generators.sdoh_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        sdoh_dataframe = run_sdoh_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric SDOH generation completed successfully. Rows: %s",
            len(sdoh_dataframe),
        )

        print("MedFabric SDOH generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "SDOH generation failed.")
        logger.exception("SDOH generation failed.")
        raise PipelineError("SDOH generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()