###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/pharmacy_generator.py
#
# Purpose:
#     Generates the canonical Pharmacy Claims dataset used by the MedFabric
#     Synthetic Data Engine.
#
# Business Context:
#     Pharmacy claims represent prescription drug utilization, medication
#     adherence, chronic medication use, member drug cost exposure, provider
#     prescribing behavior, and pharmacy benefit activity. Pharmacy claims are
#     foundational for population health, chronic condition analytics, risk
#     modeling, medication adherence, quality measures, and Member 360.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/pharmacy.yaml
#     data/raw/members.parquet
#     data/raw/providers.parquet
#     data/raw/enrollment.parquet
#     reference/pharmacy/rxnorm_reference.parquet
#     reference/pharmacy/drug_class_reference.parquet
#     reference/pharmacy/generic_brand_reference.parquet
#
# Outputs:
#     data/raw/pharmacy_claims.parquet
#     data/metadata/pharmacy_claims_dataset_metadata.json
#     data/metadata/pharmacy_claims_column_metadata.csv
#     data/metadata/pharmacy_claims_statistics.json
#     data/metadata/pharmacy_claims_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/pharmacy.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.pharmacy_generator
#
# Expected Output:
#     Canonical Pharmacy Claims dataset and metadata written to configured
#     output paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.pharmacy"
STEP_NAME = "Generate Pharmacy Claims Dataset"


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


def load_pharmacy_generation_config(context) -> Dict[str, Any]:
    """
    Load Pharmacy-specific synthetic data generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/pharmacy.yaml.
    """
    return context.configuration.load_yaml("data_generation/pharmacy.yaml")


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
        Probability array.

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
        Logical dataset name used in error messages.

    weight_column:
        Optional weight column. If present, weighted sampling is used.

    Returns
    -------
    dict
        Sampled row as dictionary.

    Raises
    ------
    PipelineError
        Raised when input DataFrame is empty.
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
    pharmacy_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all raw inputs and pharmacy reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    dict
        Dictionary of loaded input and reference DataFrames.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(pharmacy_config, "inputs", "pharmacy.yaml")
    rxnorm_config = require_config_section(pharmacy_config, "rxnorm", "pharmacy.yaml")
    drug_class_config = require_config_section(pharmacy_config, "drug_class", "pharmacy.yaml")
    generic_brand_config = require_config_section(pharmacy_config, "generic_brand", "pharmacy.yaml")

    paths = {
        "members": require_config_value(inputs_config, "members_dataset", "pharmacy.inputs"),
        "providers": require_config_value(inputs_config, "providers_dataset", "pharmacy.inputs"),
        "enrollment": require_config_value(inputs_config, "enrollment_dataset", "pharmacy.inputs"),
        "rxnorm": require_config_value(rxnorm_config, "source", "pharmacy.rxnorm"),
        "drug_class": require_config_value(drug_class_config, "source", "pharmacy.drug_class"),
        "generic_brand": require_config_value(generic_brand_config, "source", "pharmacy.generic_brand"),
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
            "Loaded Pharmacy input/reference: %s | Rows: %s | Path: %s",
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
    pharmacy_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for reproducible Pharmacy generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        pharmacy_config,
        "reproducibility",
        "pharmacy.yaml",
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
            "pharmacy.reproducibility",
        )
    )


def build_pharmacy_claim_id(
    pharmacy_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build configured pharmacy_claim_id.

    Parameters
    ----------
    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    sequence_number:
        Zero-based pharmacy claim sequence number.

    Returns
    -------
    str
        Generated pharmacy claim identifier.
    """
    identifier_config = require_config_section(
        pharmacy_config,
        "pharmacy_claim_identifier",
        "pharmacy.yaml",
    )

    prefix = require_config_value(
        identifier_config,
        "prefix",
        "pharmacy.pharmacy_claim_identifier",
    )

    starting_sequence = int(
        require_config_value(
            identifier_config,
            "starting_sequence",
            "pharmacy.pharmacy_claim_identifier",
        )
    )

    padding_length = int(
        require_config_value(
            identifier_config,
            "padding_length",
            "pharmacy.pharmacy_claim_identifier",
        )
    )

    numeric_value = starting_sequence + sequence_number

    return f"{prefix}{numeric_value:0{padding_length}d}"


def sample_fill_date(
    rng: np.random.Generator,
    pharmacy_config: Dict[str, Any],
) -> pd.Timestamp:
    """
    Generate synthetic pharmacy claim fill date.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    pandas.Timestamp
        Generated fill date.
    """
    dates_config = require_config_section(pharmacy_config, "dates", "pharmacy.yaml")

    minimum_fill_year = int(
        require_config_value(dates_config, "minimum_fill_year", "pharmacy.dates")
    )

    maximum_fill_year = int(
        require_config_value(dates_config, "maximum_fill_year", "pharmacy.dates")
    )

    if minimum_fill_year > maximum_fill_year:
        raise PipelineError("minimum_fill_year cannot be greater than maximum_fill_year.")

    fill_year = int(rng.integers(minimum_fill_year, maximum_fill_year + 1))
    fill_month = int(rng.integers(1, 13))
    fill_day = int(rng.integers(1, 29))

    return pd.Timestamp(year=fill_year, month=fill_month, day=fill_day)


def build_prescription_characteristics(
    rng: np.random.Generator,
    pharmacy_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build prescription-level operational characteristics.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    dict
        Days supply, quantity dispensed, refill flag, and refill number.
    """
    prescription_config = require_config_section(
        pharmacy_config,
        "prescription",
        "pharmacy.yaml",
    )

    minimum_days_supply = int(
        require_config_value(
            prescription_config,
            "minimum_days_supply",
            "pharmacy.prescription",
        )
    )

    maximum_days_supply = int(
        require_config_value(
            prescription_config,
            "maximum_days_supply",
            "pharmacy.prescription",
        )
    )

    minimum_quantity = int(
        require_config_value(
            prescription_config,
            "minimum_quantity_dispensed",
            "pharmacy.prescription",
        )
    )

    maximum_quantity = int(
        require_config_value(
            prescription_config,
            "maximum_quantity_dispensed",
            "pharmacy.prescription",
        )
    )

    refill_probability = float(
        require_config_value(
            prescription_config,
            "refill_probability",
            "pharmacy.prescription",
        )
    )

    if minimum_days_supply > maximum_days_supply:
        raise PipelineError("minimum_days_supply cannot be greater than maximum_days_supply.")

    if minimum_quantity > maximum_quantity:
        raise PipelineError(
            "minimum_quantity_dispensed cannot be greater than maximum_quantity_dispensed."
        )

    is_refill = bool(rng.random() < refill_probability)
    refill_number = int(rng.integers(1, 6)) if is_refill else 0

    return {
        "days_supply": int(rng.integers(minimum_days_supply, maximum_days_supply + 1)),
        "quantity_dispensed": int(rng.integers(minimum_quantity, maximum_quantity + 1)),
        "is_refill": is_refill,
        "refill_number": refill_number,
    }


def build_financials(
    rng: np.random.Generator,
    pharmacy_config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Build synthetic pharmacy financial amounts.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    dict
        Allowed, paid, and member liability amount fields.
    """
    financial_config = require_config_section(pharmacy_config, "financials", "pharmacy.yaml")

    minimum_allowed = float(
        require_config_value(
            financial_config,
            "minimum_allowed_amount",
            "pharmacy.financials",
        )
    )

    maximum_allowed = float(
        require_config_value(
            financial_config,
            "maximum_allowed_amount",
            "pharmacy.financials",
        )
    )

    paid_ratio_minimum = float(
        require_config_value(
            financial_config,
            "paid_amount_ratio_minimum",
            "pharmacy.financials",
        )
    )

    paid_ratio_maximum = float(
        require_config_value(
            financial_config,
            "paid_amount_ratio_maximum",
            "pharmacy.financials",
        )
    )

    if minimum_allowed > maximum_allowed:
        raise PipelineError("minimum_allowed_amount cannot be greater than maximum_allowed_amount.")

    allowed_amount = round(float(rng.uniform(minimum_allowed, maximum_allowed)), 2)
    paid_amount = round(
        allowed_amount * float(rng.uniform(paid_ratio_minimum, paid_ratio_maximum)),
        2,
    )
    member_liability_amount = round(max(0.0, allowed_amount - paid_amount), 2)

    return {
        "allowed_amount": allowed_amount,
        "paid_amount": paid_amount,
        "member_liability_amount": member_liability_amount,
    }


###############################################################################
# PHARMACY CLAIM RECORD GENERATION
###############################################################################


def sample_claim_count(
    rng: np.random.Generator,
    pharmacy_config: Dict[str, Any],
) -> int:
    """
    Sample number of pharmacy claims for one member.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    int
        Pharmacy claim count for a member.
    """
    volume_config = require_config_section(pharmacy_config, "claim_volume", "pharmacy.yaml")

    average_prescriptions = float(
        require_config_value(
            volume_config,
            "average_prescriptions_per_member_per_year",
            "pharmacy.claim_volume",
        )
    )

    minimum_prescriptions = int(
        require_config_value(
            volume_config,
            "minimum_prescriptions_per_member",
            "pharmacy.claim_volume",
        )
    )

    maximum_prescriptions = int(
        require_config_value(
            volume_config,
            "maximum_prescriptions_per_member",
            "pharmacy.claim_volume",
        )
    )

    sampled_count = int(rng.poisson(average_prescriptions))

    return max(minimum_prescriptions, min(maximum_prescriptions, sampled_count))


def find_matching_drug_class(
    rxnorm_row: Dict[str, Any],
    drug_class_reference: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Resolve drug class attributes for a selected RxNorm drug.

    Parameters
    ----------
    rxnorm_row:
        Sampled RxNorm reference row.

    drug_class_reference:
        Drug class reference DataFrame.

    Returns
    -------
    dict
        Drug class attributes.

    Processing Notes
    ----------------
    If the RxNorm reference already carries a drug_class value, it is used as
    the join key. Otherwise the generator samples a drug class uniformly from
    the drug class reference. This keeps the generator tolerant of reference
    schema variation while still staying reference-data driven.
    """
    if "drug_class" in rxnorm_row and "drug_class" in drug_class_reference.columns:
        matched = drug_class_reference[
            drug_class_reference["drug_class"].astype(str)
            == str(rxnorm_row["drug_class"])
        ]

        if not matched.empty:
            return matched.iloc[0].to_dict()

    if drug_class_reference.empty:
        return {}

    return drug_class_reference.iloc[0].to_dict()


def build_pharmacy_claim_record(
    rng: np.random.Generator,
    pharmacy_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    member_row: Dict[str, Any],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical Pharmacy Claim record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    datasets:
        Loaded input and reference datasets.

    member_row:
        Member row dictionary.

    sequence_number:
        Zero-based pharmacy claim sequence number.

    Returns
    -------
    dict
        Generated Pharmacy Claim record.
    """
    audit_config = require_config_section(pharmacy_config, "audit", "pharmacy.yaml")

    provider_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["providers"],
        dataset_name="raw.providers",
    )

    enrollment_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["enrollment"],
        dataset_name="raw.enrollment",
    )

    rxnorm_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["rxnorm"],
        dataset_name="reference.rxnorm",
    )

    generic_brand_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["generic_brand"],
        dataset_name="reference.generic_brand",
    )

    drug_class_row = find_matching_drug_class(
        rxnorm_row=rxnorm_row,
        drug_class_reference=datasets["drug_class"],
    )

    prescription = build_prescription_characteristics(
        rng=rng,
        pharmacy_config=pharmacy_config,
    )

    financials = build_financials(
        rng=rng,
        pharmacy_config=pharmacy_config,
    )

    fill_date = sample_fill_date(
        rng=rng,
        pharmacy_config=pharmacy_config,
    )

    return {
        "pharmacy_claim_id": build_pharmacy_claim_id(pharmacy_config, sequence_number),
        "member_id": member_row.get("member_id"),
        "enrollment_id": enrollment_row.get("enrollment_id"),
        "prescribing_provider_id": provider_row.get("provider_id"),
        "rxnorm_code": rxnorm_row.get("rxnorm_code"),
        "drug_name": rxnorm_row.get("drug_name"),
        "drug_class": rxnorm_row.get("drug_class") or drug_class_row.get("drug_class"),
        "chronic_use_flag": drug_class_row.get("chronic_use_flag"),
        "generic_brand": generic_brand_row.get("generic_brand"),
        "fill_date": fill_date.date(),
        "days_supply": prescription["days_supply"],
        "quantity_dispensed": prescription["quantity_dispensed"],
        "is_refill": prescription["is_refill"],
        "refill_number": prescription["refill_number"],
        "allowed_amount": financials["allowed_amount"],
        "paid_amount": financials["paid_amount"],
        "member_liability_amount": financials["member_liability_amount"],
        "source_system": require_config_value(
            audit_config,
            "source_system",
            "pharmacy.audit",
        ),
        "record_status": require_config_value(
            audit_config,
            "record_status",
            "pharmacy.audit",
        ),
        "created_at": pd.Timestamp(
            require_config_value(audit_config, "created_at", "pharmacy.audit")
        ),
        "updated_at": pd.Timestamp(
            require_config_value(audit_config, "updated_at", "pharmacy.audit")
        ),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_pharmacy_dataset(
    context,
    global_config: Dict[str, Any],
    pharmacy_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate canonical Pharmacy Claims dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    datasets:
        Loaded input and reference DataFrames.

    Returns
    -------
    pandas.DataFrame
        Generated Pharmacy Claims DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    random_seed = resolve_random_seed(
        global_config=global_config,
        pharmacy_config=pharmacy_config,
    )

    rng = np.random.default_rng(random_seed)
    members_dataframe = datasets["members"]

    logger.info(
        "Generating Pharmacy Claims from Member dataset. Member rows: %s",
        len(members_dataframe),
    )

    records: List[Dict[str, Any]] = []
    sequence_number = 0

    for member_row in members_dataframe.to_dict("records"):
        claim_count = sample_claim_count(
            rng=rng,
            pharmacy_config=pharmacy_config,
        )

        for _ in range(claim_count):
            records.append(
                build_pharmacy_claim_record(
                    rng=rng,
                    pharmacy_config=pharmacy_config,
                    datasets=datasets,
                    member_row=member_row,
                    sequence_number=sequence_number,
                )
            )

            sequence_number += 1

    dataframe = pd.DataFrame(records)

    logger.info(
        "Generated Pharmacy Claims dataset. Rows: %s",
        len(dataframe),
    )

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(pharmacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible rules for Pharmacy Claims.

    Parameters
    ----------
    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        pharmacy_config,
        "validation",
        "pharmacy.yaml",
    )

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_unique_pharmacy_claim_ids", False)):
        required_columns.append("pharmacy_claim_id")
        no_null_columns.append("pharmacy_claim_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_provider_id", False)):
        required_columns.append("prescribing_provider_id")
        no_null_columns.append("prescribing_provider_id")

    if bool(validation_config.get("require_rxnorm_code", False)):
        required_columns.append("rxnorm_code")
        no_null_columns.append("rxnorm_code")

    if bool(validation_config.get("require_fill_date", False)):
        required_columns.append("fill_date")
        no_null_columns.append("fill_date")

    if bool(validation_config.get("require_days_supply", False)):
        required_columns.append("days_supply")
        no_null_columns.append("days_supply")

    if bool(validation_config.get("require_quantity_dispensed", False)):
        required_columns.append("quantity_dispensed")
        no_null_columns.append("quantity_dispensed")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_pharmacy_claim_ids", False)):
        validation_rules["primary_key"] = ["pharmacy_claim_id"]

    return validation_rules


def validate_pharmacy_dataset(
    context,
    dataframe: pd.DataFrame,
    pharmacy_config: Dict[str, Any],
) -> None:
    """
    Validate generated Pharmacy Claims dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Pharmacy Claims DataFrame.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.
    """
    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=build_validation_rules(pharmacy_config),
        dataset_name="raw.pharmacy_claims",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_output_path(pharmacy_config: Dict[str, Any]) -> str:
    """
    Resolve Pharmacy Claims output path.

    Parameters
    ----------
    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(pharmacy_config, "output", "pharmacy.yaml")

    file_name = require_config_value(
        output_config,
        "file_name",
        "pharmacy.output",
    )

    return f"data/raw/{file_name}"


def write_pharmacy_dataset(
    context,
    dataframe: pd.DataFrame,
    pharmacy_config: Dict[str, Any],
) -> str:
    """
    Write Pharmacy Claims dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Pharmacy Claims DataFrame.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    Returns
    -------
    str
        Written output path.
    """
    output_path = resolve_output_path(pharmacy_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.pharmacy_claims",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_pharmacy_metadata(
    context,
    dataframe: pd.DataFrame,
    pharmacy_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Pharmacy Claims metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Pharmacy Claims DataFrame.

    pharmacy_config:
        Parsed config/data_generation/pharmacy.yaml.

    output_path:
        Written raw output path.
    """
    metadata_config = require_config_section(
        pharmacy_config,
        "metadata",
        "pharmacy.yaml",
    )

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.pharmacy_claims",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="pharmacy",
            primary_key=["pharmacy_claim_id"],
            description="Canonical MedFabric synthetic Pharmacy Claims dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/pharmacy_claims_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.pharmacy_claims",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/pharmacy_claims_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.pharmacy_claims",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/pharmacy_claims_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.pharmacy_claims",
            source_datasets=[
                "raw.members",
                "raw.providers",
                "raw.enrollment",
                "reference.rxnorm",
                "reference.drug_class",
                "reference.generic_brand",
            ],
            output_dataset="raw.pharmacy_claims",
            transformation_name="generate_pharmacy_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/pharmacy_claims_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_pharmacy_generation(context) -> pd.DataFrame:
    """
    Execute the complete Pharmacy Claims generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Pharmacy Claims dataset.
    """
    global_config = load_global_generation_config(context)
    pharmacy_config = load_pharmacy_generation_config(context)

    datasets = load_input_and_reference_data(
        context=context,
        pharmacy_config=pharmacy_config,
    )

    pharmacy_dataframe = generate_pharmacy_dataset(
        context=context,
        global_config=global_config,
        pharmacy_config=pharmacy_config,
        datasets=datasets,
    )

    validate_pharmacy_dataset(
        context=context,
        dataframe=pharmacy_dataframe,
        pharmacy_config=pharmacy_config,
    )

    output_path = write_pharmacy_dataset(
        context=context,
        dataframe=pharmacy_dataframe,
        pharmacy_config=pharmacy_config,
    )

    write_pharmacy_metadata(
        context=context,
        dataframe=pharmacy_dataframe,
        pharmacy_config=pharmacy_config,
        output_path=output_path,
    )

    return pharmacy_dataframe


def main() -> None:
    """
    Main entry point for Pharmacy Claims generation.

    Run Command
    -----------
    python -m src.data_generation.generators.pharmacy_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        pharmacy_dataframe = run_pharmacy_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Pharmacy generation completed successfully. Rows: %s",
            len(pharmacy_dataframe),
        )

        print("MedFabric pharmacy generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Pharmacy generation failed.")
        logger.exception("Pharmacy generation failed.")
        raise PipelineError("Pharmacy generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()