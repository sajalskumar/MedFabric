###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/claims_generator.py
#
# Purpose:
#     Generates canonical medical Claim Header and Claim Line datasets used by
#     the MedFabric Synthetic Data Engine.
#
# Business Context:
#     Claims are the financial and utilization backbone of payer analytics.
#     They connect members, enrollment, providers, facilities, diagnoses,
#     procedures, revenue codes, place of service, allowed amounts, paid amounts,
#     and member liability.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/claims.yaml
#     data/raw/members.parquet
#     data/raw/providers.parquet
#     data/raw/facilities.parquet
#     data/raw/enrollment.parquet
#     reference/terminology/encounter_type_reference.parquet
#     reference/terminology/icd10_reference.parquet
#     reference/terminology/cpt_reference.parquet
#     reference/terminology/hcpcs_reference.parquet
#     reference/terminology/revenue_code_reference.parquet
#     reference/terminology/place_of_service_reference.parquet
#
# Outputs:
#     data/raw/claim_headers.parquet
#     data/raw/claim_lines.parquet
#     data/metadata/claim_headers_dataset_metadata.json
#     data/metadata/claim_headers_column_metadata.csv
#     data/metadata/claim_headers_statistics.json
#     data/metadata/claim_headers_lineage.json
#     data/metadata/claim_lines_dataset_metadata.json
#     data/metadata/claim_lines_column_metadata.csv
#     data/metadata/claim_lines_statistics.json
#     data/metadata/claim_lines_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/claims.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.claims_generator
#
# Expected Output:
#     Canonical Claim Header and Claim Line datasets written to configured raw
#     output paths, with metadata written under data/metadata/.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.claims"
STEP_NAME = "Generate Claims Datasets"


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


def load_claims_generation_config(context) -> Dict[str, Any]:
    """
    Load Claims-specific generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/claims.yaml.
    """
    return context.configuration.load_yaml("data_generation/claims.yaml")


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


def sample_weighted_label(
    rng: np.random.Generator,
    weights_by_label: Dict[str, float],
    rule_name: str,
) -> str:
    """
    Sample one label from configured label weights.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    weights_by_label:
        Dictionary where keys are labels and values are weights.

    rule_name:
        Human-readable rule name used in validation errors.

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
        Source DataFrame.

    dataset_name:
        Logical dataset name used in errors.

    weight_column:
        Optional weight column. If present, weighted sampling is used.

    Returns
    -------
    dict
        Sampled row.
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
    claims_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all raw inputs and terminology reference datasets for claims.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    claims_config:
        Parsed config/data_generation/claims.yaml.

    Returns
    -------
    dict
        Dictionary of input and reference DataFrames.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(claims_config, "inputs", "claims.yaml")
    encounter_config = require_config_section(claims_config, "encounter_types", "claims.yaml")
    diagnosis_config = require_config_section(claims_config, "diagnosis", "claims.yaml")
    procedures_config = require_config_section(claims_config, "procedures", "claims.yaml")
    revenue_config = require_config_section(claims_config, "revenue_codes", "claims.yaml")
    pos_config = require_config_section(claims_config, "place_of_service", "claims.yaml")

    paths = {
        "members": require_config_value(inputs_config, "members_dataset", "claims.inputs"),
        "providers": require_config_value(inputs_config, "providers_dataset", "claims.inputs"),
        "facilities": require_config_value(inputs_config, "facilities_dataset", "claims.inputs"),
        "enrollment": require_config_value(inputs_config, "enrollment_dataset", "claims.inputs"),
        "encounter_types": require_config_value(encounter_config, "source", "claims.encounter_types"),
        "icd10": require_config_value(diagnosis_config, "icd10_source", "claims.diagnosis"),
        "cpt": require_config_value(procedures_config, "cpt_source", "claims.procedures"),
        "hcpcs": require_config_value(procedures_config, "hcpcs_source", "claims.procedures"),
        "revenue_codes": require_config_value(revenue_config, "source", "claims.revenue_codes"),
        "place_of_service": require_config_value(pos_config, "source", "claims.place_of_service"),
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
            "Loaded Claims input/reference: %s | Rows: %s | Path: %s",
            dataset_name,
            len(dataframe),
            path,
        )

    return datasets


###############################################################################
# IDENTIFIERS, DATES, AND FINANCIALS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    claims_config: Dict[str, Any],
) -> int:
    """
    Resolve the random seed used for reproducible Claims generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    claims_config:
        Parsed config/data_generation/claims.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        claims_config,
        "reproducibility",
        "claims.yaml",
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
            "claims.reproducibility",
        )
    )


def build_configured_identifier(
    identifier_config: Dict[str, Any],
    sequence_number: int,
    config_name: str,
) -> str:
    """
    Build a configured sequence-based identifier.

    Parameters
    ----------
    identifier_config:
        Identifier configuration.

    sequence_number:
        Zero-based sequence number.

    config_name:
        Human-readable configuration name used in errors.

    Returns
    -------
    str
        Generated identifier.
    """
    prefix = require_config_value(identifier_config, "prefix", config_name)

    starting_sequence = int(
        require_config_value(identifier_config, "starting_sequence", config_name)
    )

    padding_length = int(
        require_config_value(identifier_config, "padding_length", config_name)
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def build_claim_id(claims_config: Dict[str, Any], sequence_number: int) -> str:
    """
    Build configured claim_id.

    Parameters
    ----------
    claims_config:
        Parsed config/data_generation/claims.yaml.

    sequence_number:
        Zero-based claim sequence number.

    Returns
    -------
    str
        Generated claim identifier.
    """
    identifier_config = require_config_section(
        claims_config,
        "claim_identifier",
        "claims.yaml",
    )

    return build_configured_identifier(
        identifier_config=identifier_config,
        sequence_number=sequence_number,
        config_name="claims.claim_identifier",
    )


def build_claim_line_id(claims_config: Dict[str, Any], sequence_number: int) -> str:
    """
    Build configured claim_line_id.

    Parameters
    ----------
    claims_config:
        Parsed config/data_generation/claims.yaml.

    sequence_number:
        Zero-based claim line sequence number.

    Returns
    -------
    str
        Generated claim line identifier.
    """
    identifier_config = require_config_section(
        claims_config,
        "claim_line_identifier",
        "claims.yaml",
    )

    return build_configured_identifier(
        identifier_config=identifier_config,
        sequence_number=sequence_number,
        config_name="claims.claim_line_identifier",
    )


def sample_claim_type(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
) -> str:
    """
    Sample professional or institutional claim type.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    Returns
    -------
    str
        Claim type label.
    """
    claim_type_config = require_config_section(claims_config, "claim_types", "claims.yaml")

    return sample_weighted_label(
        rng=rng,
        weights_by_label={
            "Professional": float(
                require_config_value(
                    claim_type_config,
                    "professional_probability",
                    "claims.claim_types",
                )
            ),
            "Institutional": float(
                require_config_value(
                    claim_type_config,
                    "institutional_probability",
                    "claims.claim_types",
                )
            ),
        },
        rule_name="claims.claim_types",
    )


def sample_service_date(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
) -> pd.Timestamp:
    """
    Generate a synthetic service date.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    Returns
    -------
    pandas.Timestamp
        Generated service date.
    """
    dates_config = require_config_section(claims_config, "dates", "claims.yaml")

    minimum_service_year = int(
        require_config_value(dates_config, "minimum_service_year", "claims.dates")
    )

    maximum_service_year = int(
        require_config_value(dates_config, "maximum_service_year", "claims.dates")
    )

    if minimum_service_year > maximum_service_year:
        raise PipelineError("minimum_service_year cannot be greater than maximum_service_year.")

    service_year = int(rng.integers(minimum_service_year, maximum_service_year + 1))
    service_month = int(rng.integers(1, 13))
    service_day = int(rng.integers(1, 29))

    return pd.Timestamp(year=service_year, month=service_month, day=service_day)


def build_admission_discharge_dates(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
    claim_type: str,
    service_date: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Build admission and discharge dates for institutional claims.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    claim_type:
        Claim type label.

    service_date:
        Generated service date.

    Returns
    -------
    dict
        Admission and discharge dates. Professional claims return null values.
    """
    dates_config = require_config_section(claims_config, "dates", "claims.yaml")

    if claim_type != "Institutional":
        return {
            "admission_date": None,
            "discharge_date": None,
        }

    max_los = int(
        require_config_value(
            dates_config,
            "maximum_inpatient_length_of_stay_days",
            "claims.dates",
        )
    )

    length_of_stay = int(rng.integers(1, max_los + 1))

    return {
        "admission_date": service_date.date(),
        "discharge_date": (service_date + pd.DateOffset(days=length_of_stay)).date(),
    }


def build_financials(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Build synthetic allowed, paid, and member liability amounts.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    Returns
    -------
    dict
        Financial amount fields.
    """
    financial_config = require_config_section(claims_config, "financials", "claims.yaml")

    minimum_allowed = float(
        require_config_value(
            financial_config,
            "minimum_allowed_amount",
            "claims.financials",
        )
    )

    maximum_allowed = float(
        require_config_value(
            financial_config,
            "maximum_allowed_amount",
            "claims.financials",
        )
    )

    paid_ratio_minimum = float(
        require_config_value(
            financial_config,
            "paid_amount_ratio_minimum",
            "claims.financials",
        )
    )

    paid_ratio_maximum = float(
        require_config_value(
            financial_config,
            "paid_amount_ratio_maximum",
            "claims.financials",
        )
    )

    member_liability_minimum = float(
        require_config_value(
            financial_config,
            "member_liability_ratio_minimum",
            "claims.financials",
        )
    )

    member_liability_maximum = float(
        require_config_value(
            financial_config,
            "member_liability_ratio_maximum",
            "claims.financials",
        )
    )

    allowed_amount = round(float(rng.uniform(minimum_allowed, maximum_allowed)), 2)
    paid_amount = round(
        allowed_amount * float(rng.uniform(paid_ratio_minimum, paid_ratio_maximum)),
        2,
    )
    member_liability = round(
        allowed_amount * float(rng.uniform(member_liability_minimum, member_liability_maximum)),
        2,
    )

    deductible_amount = round(member_liability * 0.40, 2)
    copay_amount = round(member_liability * 0.25, 2)
    coinsurance_amount = round(member_liability - deductible_amount - copay_amount, 2)

    return {
        "allowed_amount": allowed_amount,
        "paid_amount": paid_amount,
        "member_liability_amount": member_liability,
        "deductible_amount": deductible_amount,
        "copay_amount": copay_amount,
        "coinsurance_amount": coinsurance_amount,
    }


###############################################################################
# CLAIM RECORD BUILDERS
###############################################################################


def sample_claim_count(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
) -> int:
    """
    Sample number of claims for a member.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    Returns
    -------
    int
        Claim count for one member.
    """
    volume_config = require_config_section(claims_config, "claim_volume", "claims.yaml")

    average_claims = float(
        require_config_value(
            volume_config,
            "average_claims_per_member_per_year",
            "claims.claim_volume",
        )
    )

    minimum_claims = int(
        require_config_value(
            volume_config,
            "minimum_claims_per_member",
            "claims.claim_volume",
        )
    )

    maximum_claims = int(
        require_config_value(
            volume_config,
            "maximum_claims_per_member",
            "claims.claim_volume",
        )
    )

    sampled_count = int(rng.poisson(average_claims))

    return max(minimum_claims, min(maximum_claims, sampled_count))


def sample_primary_diagnosis(
    rng: np.random.Generator,
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Sample primary diagnosis from ICD-10 reference.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    datasets:
        Loaded reference datasets.

    Returns
    -------
    dict
        Diagnosis code row.
    """
    return sample_dataframe_row(
        rng=rng,
        dataframe=datasets["icd10"],
        dataset_name="reference.icd10",
    )


def sample_procedure(
    rng: np.random.Generator,
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Sample procedure code from CPT or HCPCS references.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    datasets:
        Loaded reference datasets.

    Returns
    -------
    dict
        Procedure row.
    """
    procedure_source = str(rng.choice(["cpt", "hcpcs"]))

    return sample_dataframe_row(
        rng=rng,
        dataframe=datasets[procedure_source],
        dataset_name=f"reference.{procedure_source}",
    )


def build_claim_header_record(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    member_row: Dict[str, Any],
    claim_sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one claim header record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    datasets:
        Loaded input and reference datasets.

    member_row:
        Member row dictionary.

    claim_sequence_number:
        Zero-based claim sequence number.

    Returns
    -------
    dict
        Claim header record.
    """
    audit_config = require_config_section(claims_config, "audit", "claims.yaml")
    status_config = require_config_section(claims_config, "claim_status", "claims.yaml")

    claim_id = build_claim_id(claims_config, claim_sequence_number)
    claim_type = sample_claim_type(rng, claims_config)
    service_date = sample_service_date(rng, claims_config)
    admit_discharge = build_admission_discharge_dates(
        rng=rng,
        claims_config=claims_config,
        claim_type=claim_type,
        service_date=service_date,
    )

    provider_row = sample_dataframe_row(rng, datasets["providers"], "raw.providers")
    facility_row = sample_dataframe_row(rng, datasets["facilities"], "raw.facilities")
    enrollment_row = sample_dataframe_row(rng, datasets["enrollment"], "raw.enrollment")
    encounter_row = sample_dataframe_row(rng, datasets["encounter_types"], "reference.encounter_types")
    diagnosis_row = sample_primary_diagnosis(rng, datasets)

    financials = build_financials(rng, claims_config)

    return {
        "claim_id": claim_id,
        "member_id": member_row.get("member_id"),
        "enrollment_id": enrollment_row.get("enrollment_id"),
        "provider_id": provider_row.get("provider_id"),
        "facility_id": facility_row.get("facility_id"),
        "claim_type": claim_type,
        "encounter_type": encounter_row.get("encounter_type"),
        "service_date": service_date.date(),
        "admission_date": admit_discharge["admission_date"],
        "discharge_date": admit_discharge["discharge_date"],
        "primary_diagnosis_code": diagnosis_row.get("code"),
        "primary_diagnosis_description": diagnosis_row.get("description"),
        "allowed_amount": financials["allowed_amount"],
        "paid_amount": financials["paid_amount"],
        "member_liability_amount": financials["member_liability_amount"],
        "deductible_amount": financials["deductible_amount"],
        "copay_amount": financials["copay_amount"],
        "coinsurance_amount": financials["coinsurance_amount"],
        "claim_status": require_config_value(
            status_config,
            "default_claim_status",
            "claims.claim_status",
        ),
        "adjudication_status": require_config_value(
            status_config,
            "default_adjudication_status",
            "claims.claim_status",
        ),
        "source_system": require_config_value(audit_config, "source_system", "claims.audit"),
        "record_status": require_config_value(audit_config, "record_status", "claims.audit"),
        "created_at": pd.Timestamp(require_config_value(audit_config, "created_at", "claims.audit")),
        "updated_at": pd.Timestamp(require_config_value(audit_config, "updated_at", "claims.audit")),
    }


def build_claim_line_records(
    rng: np.random.Generator,
    claims_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    claim_header: Dict[str, Any],
    starting_claim_line_sequence_number: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Build claim line records for one claim header.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    claims_config:
        Parsed claims configuration.

    datasets:
        Loaded input and reference datasets.

    claim_header:
        Generated claim header dictionary.

    starting_claim_line_sequence_number:
        First available claim line sequence number.

    Returns
    -------
    tuple[list[dict], int]
        Claim line records and next available claim line sequence number.
    """
    procedures_config = require_config_section(claims_config, "procedures", "claims.yaml")

    maximum_procedures = int(
        require_config_value(
            procedures_config,
            "maximum_procedures_per_claim",
            "claims.procedures",
        )
    )

    line_count = int(rng.integers(1, max(2, maximum_procedures + 1)))
    claim_lines: List[Dict[str, Any]] = []
    next_sequence = starting_claim_line_sequence_number

    for line_number in range(1, line_count + 1):
        procedure_row = sample_procedure(rng, datasets)
        revenue_row = sample_dataframe_row(rng, datasets["revenue_codes"], "reference.revenue_codes")
        pos_row = sample_dataframe_row(rng, datasets["place_of_service"], "reference.place_of_service")

        line_allowed_amount = round(float(claim_header["allowed_amount"]) / line_count, 2)
        line_paid_amount = round(float(claim_header["paid_amount"]) / line_count, 2)

        claim_lines.append(
            {
                "claim_line_id": build_claim_line_id(claims_config, next_sequence),
                "claim_id": claim_header["claim_id"],
                "claim_line_number": line_number,
                "member_id": claim_header["member_id"],
                "provider_id": claim_header["provider_id"],
                "facility_id": claim_header["facility_id"],
                "service_date": claim_header["service_date"],
                "procedure_code": procedure_row.get("code"),
                "procedure_description": procedure_row.get("description"),
                "revenue_code": revenue_row.get("code"),
                "revenue_code_description": revenue_row.get("description"),
                "place_of_service_code": pos_row.get("code"),
                "place_of_service_description": pos_row.get("description"),
                "diagnosis_code": claim_header["primary_diagnosis_code"],
                "allowed_amount": line_allowed_amount,
                "paid_amount": line_paid_amount,
                "source_system": claim_header["source_system"],
                "record_status": claim_header["record_status"],
                "created_at": claim_header["created_at"],
                "updated_at": claim_header["updated_at"],
            }
        )

        next_sequence += 1

    return claim_lines, next_sequence


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_claim_datasets(
    context,
    global_config: Dict[str, Any],
    claims_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Claim Header and Claim Line datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed generation.yaml.

    claims_config:
        Parsed claims.yaml.

    datasets:
        Loaded input and reference DataFrames.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Claim header and claim line DataFrames.
    """
    logger = context.get_logger(MODULE_NAME)

    random_seed = resolve_random_seed(
        global_config=global_config,
        claims_config=claims_config,
    )

    rng = np.random.default_rng(random_seed)
    claim_headers: List[Dict[str, Any]] = []
    claim_lines: List[Dict[str, Any]] = []

    claim_sequence_number = 0
    claim_line_sequence_number = 0

    members_dataframe = datasets["members"]

    logger.info("Generating Claims from Member dataset. Member rows: %s", len(members_dataframe))

    for member_row in members_dataframe.to_dict("records"):
        claim_count = sample_claim_count(rng, claims_config)

        for _ in range(claim_count):
            claim_header = build_claim_header_record(
                rng=rng,
                claims_config=claims_config,
                datasets=datasets,
                member_row=member_row,
                claim_sequence_number=claim_sequence_number,
            )

            claim_headers.append(claim_header)

            generated_lines, claim_line_sequence_number = build_claim_line_records(
                rng=rng,
                claims_config=claims_config,
                datasets=datasets,
                claim_header=claim_header,
                starting_claim_line_sequence_number=claim_line_sequence_number,
            )

            claim_lines.extend(generated_lines)
            claim_sequence_number += 1

    claim_header_dataframe = pd.DataFrame(claim_headers)
    claim_line_dataframe = pd.DataFrame(claim_lines)

    logger.info(
        "Generated Claims. Header rows: %s | Line rows: %s",
        len(claim_header_dataframe),
        len(claim_line_dataframe),
    )

    return claim_header_dataframe, claim_line_dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_claim_header_validation_rules(claims_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build validation rules for Claim Header dataset.

    Parameters
    ----------
    claims_config:
        Parsed claims.yaml.

    Returns
    -------
    dict
        Validation rules.
    """
    validation_config = require_config_section(claims_config, "validation", "claims.yaml")

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_claim_id", False)):
        required_columns.append("claim_id")
        no_null_columns.append("claim_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_provider_id", False)):
        required_columns.append("provider_id")
        no_null_columns.append("provider_id")

    if bool(validation_config.get("require_service_date", False)):
        required_columns.append("service_date")
        no_null_columns.append("service_date")

    if bool(validation_config.get("require_allowed_amount", False)):
        required_columns.append("allowed_amount")
        no_null_columns.append("allowed_amount")

    if bool(validation_config.get("require_paid_amount", False)):
        required_columns.append("paid_amount")
        no_null_columns.append("paid_amount")

    if bool(validation_config.get("require_primary_diagnosis", False)):
        required_columns.append("primary_diagnosis_code")
        no_null_columns.append("primary_diagnosis_code")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_claim_id", False)):
        validation_rules["primary_key"] = ["claim_id"]

    return validation_rules


def build_claim_line_validation_rules(claims_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build validation rules for Claim Line dataset.

    Parameters
    ----------
    claims_config:
        Parsed claims.yaml.

    Returns
    -------
    dict
        Validation rules.
    """
    validation_config = require_config_section(claims_config, "validation", "claims.yaml")

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_claim_line_id", False)):
        required_columns.append("claim_line_id")
        no_null_columns.append("claim_line_id")

    if bool(validation_config.get("require_claim_id", False)):
        required_columns.append("claim_id")
        no_null_columns.append("claim_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_provider_id", False)):
        required_columns.append("provider_id")
        no_null_columns.append("provider_id")

    if bool(validation_config.get("require_service_date", False)):
        required_columns.append("service_date")
        no_null_columns.append("service_date")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_claim_line_id", False)):
        validation_rules["primary_key"] = ["claim_line_id"]

    return validation_rules


def validate_claim_outputs(
    context,
    claims_config: Dict[str, Any],
    claim_header_dataframe: pd.DataFrame,
    claim_line_dataframe: pd.DataFrame,
) -> None:
    """
    Validate Claim Header and Claim Line outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    claims_config:
        Parsed claims.yaml.

    claim_header_dataframe:
        Generated claim header DataFrame.

    claim_line_dataframe:
        Generated claim line DataFrame.
    """
    context.validation.validate_dataset(
        dataframe=claim_header_dataframe,
        validation_rules=build_claim_header_validation_rules(claims_config),
        dataset_name="raw.claim_headers",
    )

    context.validation.validate_dataset(
        dataframe=claim_line_dataframe,
        validation_rules=build_claim_line_validation_rules(claims_config),
        dataset_name="raw.claim_lines",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_output_paths(claims_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve output paths for claim headers and claim lines.

    Parameters
    ----------
    claims_config:
        Parsed claims.yaml.

    Returns
    -------
    dict
        Output path dictionary.
    """
    output_config = require_config_section(claims_config, "output", "claims.yaml")

    return {
        "claim_headers": f"data/raw/{require_config_value(output_config, 'claim_header_file_name', 'claims.output')}",
        "claim_lines": f"data/raw/{require_config_value(output_config, 'claim_line_file_name', 'claims.output')}",
    }


def write_claim_outputs(
    context,
    claims_config: Dict[str, Any],
    claim_header_dataframe: pd.DataFrame,
    claim_line_dataframe: pd.DataFrame,
) -> Dict[str, str]:
    """
    Write Claim Header and Claim Line datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    claims_config:
        Parsed claims.yaml.

    claim_header_dataframe:
        Generated claim header DataFrame.

    claim_line_dataframe:
        Generated claim line DataFrame.

    Returns
    -------
    dict
        Written output paths.
    """
    output_paths = resolve_output_paths(claims_config)

    written_header_path = context.storage.write_parquet(
        dataframe=claim_header_dataframe,
        path=output_paths["claim_headers"],
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.claim_headers",
        row_count=len(claim_header_dataframe),
        column_count=len(claim_header_dataframe.columns),
        path=written_header_path,
    )

    written_line_path = context.storage.write_parquet(
        dataframe=claim_line_dataframe,
        path=output_paths["claim_lines"],
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.claim_lines",
        row_count=len(claim_line_dataframe),
        column_count=len(claim_line_dataframe.columns),
        path=written_line_path,
    )

    return {
        "claim_headers": str(written_header_path),
        "claim_lines": str(written_line_path),
    }


def write_single_dataset_metadata(
    context,
    dataframe: pd.DataFrame,
    dataset_name: str,
    domain: str,
    output_path: str,
    primary_key: List[str],
    description: str,
    metadata_prefix: str,
    source_datasets: List[str],
    transformation_name: str,
) -> None:
    """
    Write metadata outputs for one dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Dataset DataFrame.

    dataset_name:
        Logical dataset name.

    domain:
        Business domain.

    output_path:
        Written raw output path.

    primary_key:
        Primary key columns.

    description:
        Dataset description.

    metadata_prefix:
        Prefix for metadata output files.

    source_datasets:
        Lineage source datasets.

    transformation_name:
        Transformation name.
    """
    dataset_metadata = context.metadata.build_dataset_metadata(
        dataset_name=dataset_name,
        dataframe=dataframe,
        output_path=output_path,
        layer="raw",
        domain=domain,
        primary_key=primary_key,
        description=description,
    )

    context.metadata.write_metadata(
        metadata=dataset_metadata,
        output_path=f"data/metadata/{metadata_prefix}_dataset_metadata.json",
    )

    column_metadata = context.metadata.build_column_metadata(
        dataset_name=dataset_name,
        dataframe=dataframe,
    )

    context.metadata.write_metadata(
        metadata=column_metadata,
        output_path=f"data/metadata/{metadata_prefix}_column_metadata.csv",
    )

    statistics = context.metadata.build_statistics(
        dataset_name=dataset_name,
        dataframe=dataframe,
    )

    context.metadata.write_metadata(
        metadata=statistics,
        output_path=f"data/metadata/{metadata_prefix}_statistics.json",
    )

    lineage = context.metadata.build_lineage(
        dataset_name=dataset_name,
        source_datasets=source_datasets,
        output_dataset=dataset_name,
        transformation_name=transformation_name,
        module_name=MODULE_NAME,
    )

    context.metadata.write_metadata(
        metadata=lineage,
        output_path=f"data/metadata/{metadata_prefix}_lineage.json",
    )


def write_claim_metadata(
    context,
    claims_config: Dict[str, Any],
    claim_header_dataframe: pd.DataFrame,
    claim_line_dataframe: pd.DataFrame,
    output_paths: Dict[str, str],
) -> None:
    """
    Write metadata for claim header and claim line datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    claims_config:
        Parsed claims.yaml.

    claim_header_dataframe:
        Generated claim header DataFrame.

    claim_line_dataframe:
        Generated claim line DataFrame.

    output_paths:
        Written raw output paths.
    """
    metadata_config = require_config_section(claims_config, "metadata", "claims.yaml")

    if not any(
        bool(metadata_config.get(key, False))
        for key in [
            "generate_dataset_metadata",
            "generate_column_metadata",
            "generate_statistics",
            "generate_lineage",
        ]
    ):
        return

    common_sources = [
        "raw.members",
        "raw.providers",
        "raw.facilities",
        "raw.enrollment",
        "reference.encounter_types",
        "reference.icd10",
        "reference.cpt",
        "reference.hcpcs",
        "reference.revenue_codes",
        "reference.place_of_service",
    ]

    write_single_dataset_metadata(
        context=context,
        dataframe=claim_header_dataframe,
        dataset_name="raw.claim_headers",
        domain="claims",
        output_path=output_paths["claim_headers"],
        primary_key=["claim_id"],
        description="Canonical MedFabric synthetic Claim Header dataset.",
        metadata_prefix="claim_headers",
        source_datasets=common_sources,
        transformation_name="generate_claim_header_dataset",
    )

    write_single_dataset_metadata(
        context=context,
        dataframe=claim_line_dataframe,
        dataset_name="raw.claim_lines",
        domain="claims",
        output_path=output_paths["claim_lines"],
        primary_key=["claim_line_id"],
        description="Canonical MedFabric synthetic Claim Line dataset.",
        metadata_prefix="claim_lines",
        source_datasets=["raw.claim_headers"] + common_sources,
        transformation_name="generate_claim_line_dataset",
    )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_claims_generation(context) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete Claims generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Claim Header and Claim Line DataFrames.
    """
    global_config = load_global_generation_config(context)
    claims_config = load_claims_generation_config(context)

    datasets = load_input_and_reference_data(
        context=context,
        claims_config=claims_config,
    )

    claim_header_dataframe, claim_line_dataframe = generate_claim_datasets(
        context=context,
        global_config=global_config,
        claims_config=claims_config,
        datasets=datasets,
    )

    validate_claim_outputs(
        context=context,
        claims_config=claims_config,
        claim_header_dataframe=claim_header_dataframe,
        claim_line_dataframe=claim_line_dataframe,
    )

    output_paths = write_claim_outputs(
        context=context,
        claims_config=claims_config,
        claim_header_dataframe=claim_header_dataframe,
        claim_line_dataframe=claim_line_dataframe,
    )

    write_claim_metadata(
        context=context,
        claims_config=claims_config,
        claim_header_dataframe=claim_header_dataframe,
        claim_line_dataframe=claim_line_dataframe,
        output_paths=output_paths,
    )

    return claim_header_dataframe, claim_line_dataframe


def main() -> None:
    """
    Main entry point for Claims generation.

    Run Command
    -----------
    python -m src.data_generation.generators.claims_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        claim_header_dataframe, claim_line_dataframe = run_claims_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Claims generation completed successfully. "
            "Claim Header rows: %s | Claim Line rows: %s",
            len(claim_header_dataframe),
            len(claim_line_dataframe),
        )

        print("MedFabric claims generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Claims generation failed.")
        logger.exception("Claims generation failed.")
        raise PipelineError("Claims generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()