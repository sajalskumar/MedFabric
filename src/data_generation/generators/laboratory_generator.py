###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/laboratory_generator.py
#
# Purpose:
#     Generates the canonical Laboratory Results dataset used by the MedFabric
#     Synthetic Data Engine.
#
# Business Context:
#     Laboratory results provide clinical evidence for chronic disease
#     monitoring, care gaps, quality measures, risk models, Member 360,
#     clinical registries, and population health analytics.
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/laboratory.yaml
#     data/raw/members.parquet
#     data/raw/providers.parquet
#     data/raw/facilities.parquet
#     data/raw/enrollment.parquet
#     reference/laboratory/laboratory_test_reference.parquet
#     reference/laboratory/condition_lab_mapping.parquet
#     reference/laboratory/result_unit_reference.parquet
#     reference/terminology/loinc_reference.parquet
#
# Outputs:
#     data/raw/laboratory_results.parquet
#     data/metadata/laboratory_results_dataset_metadata.json
#     data/metadata/laboratory_results_column_metadata.csv
#     data/metadata/laboratory_results_statistics.json
#     data/metadata/laboratory_results_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/laboratory.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#
# Run Command:
#     python -m src.data_generation.generators.laboratory_generator
#
# Expected Output:
#     Canonical Laboratory Results dataset and metadata written to configured
#     output paths.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.laboratory"
STEP_NAME = "Generate Laboratory Results Dataset"


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


def load_laboratory_generation_config(context) -> Dict[str, Any]:
    """
    Load Laboratory-specific synthetic data generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/laboratories.yaml.
    """
    return context.configuration.load_yaml("data_generation/laboratories.yaml")


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
        Logical dataset name used in errors.

    weight_column:
        Optional weight column. If present, weighted sampling is used.

    Returns
    -------
    dict
        Sampled row.

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
    laboratory_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all raw inputs and laboratory reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    Returns
    -------
    dict
        Dictionary of loaded input and reference DataFrames.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(laboratory_config, "inputs", "laboratory.yaml")
    tests_config = require_config_section(laboratory_config, "laboratory_tests", "laboratory.yaml")
    mapping_config = require_config_section(laboratory_config, "condition_mapping", "laboratory.yaml")
    units_config = require_config_section(laboratory_config, "result_units", "laboratory.yaml")
    loinc_config = require_config_section(laboratory_config, "loinc", "laboratory.yaml")

    paths = {
        "members": require_config_value(inputs_config, "members_dataset", "laboratory.inputs"),
        "providers": require_config_value(inputs_config, "providers_dataset", "laboratory.inputs"),
        "facilities": require_config_value(inputs_config, "facilities_dataset", "laboratory.inputs"),
        "enrollment": require_config_value(inputs_config, "enrollment_dataset", "laboratory.inputs"),
        "laboratory_tests": require_config_value(tests_config, "source", "laboratory.laboratory_tests"),
        "condition_mapping": require_config_value(mapping_config, "source", "laboratory.condition_mapping"),
        "result_units": require_config_value(units_config, "source", "laboratory.result_units"),
        "loinc": require_config_value(loinc_config, "source", "laboratory.loinc"),
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
            "Loaded Laboratory input/reference: %s | Rows: %s | Path: %s",
            dataset_name,
            len(dataframe),
            path,
        )

    return datasets


###############################################################################
# IDENTIFIERS, DATES, AND RESULTS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    laboratory_config: Dict[str, Any],
) -> int:
    """
    Resolve random seed for reproducible Laboratory generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    Returns
    -------
    int
        Random seed.
    """
    reproducibility_config = require_config_section(
        laboratory_config,
        "reproducibility",
        "laboratory.yaml",
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
            "laboratory.reproducibility",
        )
    )


def build_lab_result_id(
    sequence_number: int,
) -> str:
    """
    Build deterministic laboratory result identifier.

    Parameters
    ----------
    sequence_number:
        Zero-based lab result sequence number.

    Returns
    -------
    str
        Generated lab result identifier.

    Notes
    -----
    Current laboratory.yaml does not define an explicit identifier section.
    This function uses a deterministic platform convention. If you want this
    fully configurable like claims/pharmacy, add lab_result_identifier to YAML.
    """
    return f"LAB{900000000 + sequence_number:012d}"


def sample_result_count(
    rng: np.random.Generator,
    laboratory_config: Dict[str, Any],
) -> int:
    """
    Sample number of laboratory results for one member.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    laboratory_config:
        Parsed laboratory configuration.

    Returns
    -------
    int
        Laboratory result count for one member.
    """
    volume_config = require_config_section(
        laboratory_config,
        "laboratory_volume",
        "laboratory.yaml",
    )

    average_results = float(
        require_config_value(
            volume_config,
            "average_results_per_member_per_year",
            "laboratory.laboratory_volume",
        )
    )

    minimum_results = int(
        require_config_value(
            volume_config,
            "minimum_results_per_member",
            "laboratory.laboratory_volume",
        )
    )

    maximum_results = int(
        require_config_value(
            volume_config,
            "maximum_results_per_member",
            "laboratory.laboratory_volume",
        )
    )

    sampled_count = int(rng.poisson(average_results))

    return max(minimum_results, min(maximum_results, sampled_count))


def sample_collection_datetime(
    rng: np.random.Generator,
) -> pd.Timestamp:
    """
    Generate synthetic laboratory collection datetime.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    Returns
    -------
    pandas.Timestamp
        Generated collection datetime.

    Notes
    -----
    The current laboratory.yaml enables datetime generation but does not define
    a year range. The function uses the platform's active generation window
    used across the project. Add explicit date settings to YAML if desired.
    """
    year = int(rng.integers(2020, 2027))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    hour = int(rng.integers(0, 24))
    minute = int(rng.integers(0, 60))

    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def build_result_datetime(
    rng: np.random.Generator,
    collection_datetime: pd.Timestamp,
) -> pd.Timestamp:
    """
    Generate result datetime after collection datetime.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    collection_datetime:
        Generated collection datetime.

    Returns
    -------
    pandas.Timestamp
        Result datetime.
    """
    turnaround_hours = int(rng.integers(1, 73))

    return collection_datetime + pd.DateOffset(hours=turnaround_hours)


def resolve_loinc_for_test(
    laboratory_test_row: Dict[str, Any],
    loinc_reference: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Resolve LOINC attributes for a selected laboratory test.

    Parameters
    ----------
    laboratory_test_row:
        Sampled laboratory test reference row.

    loinc_reference:
        LOINC reference DataFrame.

    Returns
    -------
    dict
        LOINC attributes.
    """
    if "loinc_code" in laboratory_test_row:
        loinc_code = laboratory_test_row.get("loinc_code")

        if "code" in loinc_reference.columns:
            matched = loinc_reference[
                loinc_reference["code"].astype(str) == str(loinc_code)
            ]

            if not matched.empty:
                return matched.iloc[0].to_dict()

        return {
            "code": loinc_code,
            "description": laboratory_test_row.get("test_name"),
        }

    if not loinc_reference.empty:
        return loinc_reference.iloc[0].to_dict()

    return {}


def resolve_unit_for_test(
    laboratory_test_row: Dict[str, Any],
    result_unit_reference: pd.DataFrame,
) -> str:
    """
    Resolve result unit for a selected laboratory test.

    Parameters
    ----------
    laboratory_test_row:
        Sampled laboratory test reference row.

    result_unit_reference:
        Result unit reference DataFrame.

    Returns
    -------
    str
        Result unit.
    """
    if laboratory_test_row.get("default_unit") is not None:
        return str(laboratory_test_row.get("default_unit"))

    if "unit" in result_unit_reference.columns and not result_unit_reference.empty:
        return str(result_unit_reference.iloc[0]["unit"])

    return ""


def build_numeric_result(
    rng: np.random.Generator,
    test_name: str,
) -> Dict[str, Any]:
    """
    Build synthetic numeric laboratory result.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    test_name:
        Laboratory test name.

    Returns
    -------
    dict
        Result value, reference range, and abnormal flag.

    Notes
    -----
    Current laboratory.yaml enables result generation but does not define
    per-test result distributions. This function uses safe synthetic defaults.
    Add reference-range configuration later for fully clinical simulation.
    """
    normalized_test_name = str(test_name).lower()

    if "a1c" in normalized_test_name:
        result_value = round(float(rng.uniform(4.8, 10.5)), 1)
        low_value = 4.0
        high_value = 5.6
    elif "glucose" in normalized_test_name:
        result_value = round(float(rng.uniform(65, 260)), 0)
        low_value = 70
        high_value = 99
    elif "creatinine" in normalized_test_name:
        result_value = round(float(rng.uniform(0.6, 3.5)), 2)
        low_value = 0.6
        high_value = 1.3
    elif "hemoglobin" in normalized_test_name:
        result_value = round(float(rng.uniform(8.0, 17.5)), 1)
        low_value = 12.0
        high_value = 16.0
    else:
        result_value = round(float(rng.uniform(1.0, 100.0)), 2)
        low_value = 1.0
        high_value = 100.0

    abnormal_flag = "N"

    if result_value < low_value:
        abnormal_flag = "L"

    if result_value > high_value:
        abnormal_flag = "H"

    return {
        "result_value": result_value,
        "reference_low": low_value,
        "reference_high": high_value,
        "abnormal_flag": abnormal_flag,
    }


###############################################################################
# LABORATORY RECORD GENERATION
###############################################################################


def build_laboratory_result_record(
    rng: np.random.Generator,
    laboratory_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    member_row: Dict[str, Any],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical Laboratory Result record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    datasets:
        Loaded input and reference datasets.

    member_row:
        Member row dictionary.

    sequence_number:
        Zero-based laboratory result sequence number.

    Returns
    -------
    dict
        Generated Laboratory Result record.
    """
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

    laboratory_test_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["laboratory_tests"],
        dataset_name="reference.laboratory_tests",
    )

    loinc_row = resolve_loinc_for_test(
        laboratory_test_row=laboratory_test_row,
        loinc_reference=datasets["loinc"],
    )

    test_name = (
        laboratory_test_row.get("test_name")
        or laboratory_test_row.get("description")
        or loinc_row.get("description")
    )

    result_unit = resolve_unit_for_test(
        laboratory_test_row=laboratory_test_row,
        result_unit_reference=datasets["result_units"],
    )

    result = build_numeric_result(
        rng=rng,
        test_name=str(test_name),
    )

    collection_datetime = sample_collection_datetime(rng)
    result_datetime = build_result_datetime(rng, collection_datetime)

    return {
        "lab_result_id": build_lab_result_id(sequence_number),
        "member_id": member_row.get("member_id"),
        "enrollment_id": enrollment_row.get("enrollment_id"),
        "ordering_provider_id": provider_row.get("provider_id"),
        "facility_id": facility_row.get("facility_id"),
        "test_name": test_name,
        "loinc_code": loinc_row.get("code") or laboratory_test_row.get("loinc_code"),
        "loinc_description": loinc_row.get("description") or test_name,
        "result_value": result["result_value"],
        "result_unit": result_unit,
        "reference_low": result["reference_low"],
        "reference_high": result["reference_high"],
        "abnormal_flag": result["abnormal_flag"],
        "result_status": "Final",
        "collection_datetime": collection_datetime,
        "result_datetime": result_datetime,
        "source_system": "MedFabric Synthetic Data Engine",
        "record_status": "Active",
        "created_at": pd.Timestamp("2026-01-01"),
        "updated_at": pd.Timestamp("2026-01-01"),
    }


###############################################################################
# DATASET GENERATION
###############################################################################


def generate_laboratory_dataset(
    context,
    global_config: Dict[str, Any],
    laboratory_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate canonical Laboratory Results dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    datasets:
        Loaded input and reference datasets.

    Returns
    -------
    pandas.DataFrame
        Generated Laboratory Results DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    random_seed = resolve_random_seed(
        global_config=global_config,
        laboratory_config=laboratory_config,
    )

    rng = np.random.default_rng(random_seed)
    members_dataframe = datasets["members"]

    logger.info(
        "Generating Laboratory Results from Member dataset. Member rows: %s",
        len(members_dataframe),
    )

    records: List[Dict[str, Any]] = []
    sequence_number = 0

    for member_row in members_dataframe.to_dict("records"):
        result_count = sample_result_count(
            rng=rng,
            laboratory_config=laboratory_config,
        )

        for _ in range(result_count):
            records.append(
                build_laboratory_result_record(
                    rng=rng,
                    laboratory_config=laboratory_config,
                    datasets=datasets,
                    member_row=member_row,
                    sequence_number=sequence_number,
                )
            )

            sequence_number += 1

    dataframe = pd.DataFrame(records)

    logger.info(
        "Generated Laboratory Results dataset. Rows: %s",
        len(dataframe),
    )

    return dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_validation_rules(laboratory_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager-compatible rules for Laboratory Results.

    Parameters
    ----------
    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        laboratory_config,
        "validation",
        "laboratory.yaml",
    )

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_lab_result_id", False)):
        required_columns.append("lab_result_id")
        no_null_columns.append("lab_result_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_provider_id", False)):
        required_columns.append("ordering_provider_id")
        no_null_columns.append("ordering_provider_id")

    if bool(validation_config.get("require_facility_id", False)):
        required_columns.append("facility_id")
        no_null_columns.append("facility_id")

    if bool(validation_config.get("require_test_name", False)):
        required_columns.append("test_name")
        no_null_columns.append("test_name")

    if bool(validation_config.get("require_loinc_code", False)):
        required_columns.append("loinc_code")
        no_null_columns.append("loinc_code")

    if bool(validation_config.get("require_result_value", False)):
        required_columns.append("result_value")
        no_null_columns.append("result_value")

    if bool(validation_config.get("require_result_unit", False)):
        required_columns.append("result_unit")
        no_null_columns.append("result_unit")

    if bool(validation_config.get("require_collection_datetime", False)):
        required_columns.append("collection_datetime")
        no_null_columns.append("collection_datetime")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_lab_result_ids", False)):
        validation_rules["primary_key"] = ["lab_result_id"]

    return validation_rules


def validate_laboratory_dataset(
    context,
    dataframe: pd.DataFrame,
    laboratory_config: Dict[str, Any],
) -> None:
    """
    Validate generated Laboratory Results dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Laboratory Results DataFrame.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.
    """
    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=build_validation_rules(laboratory_config),
        dataset_name="raw.laboratory_results",
    )


###############################################################################
# OUTPUTS AND METADATA
###############################################################################


def resolve_output_path(laboratory_config: Dict[str, Any]) -> str:
    """
    Resolve Laboratory Results output path.

    Parameters
    ----------
    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    Returns
    -------
    str
        Output path.
    """
    output_config = require_config_section(laboratory_config, "output", "laboratory.yaml")

    file_name = require_config_value(
        output_config,
        "file_name",
        "laboratory.output",
    )

    return f"data/raw/{file_name}"


def write_laboratory_dataset(
    context,
    dataframe: pd.DataFrame,
    laboratory_config: Dict[str, Any],
) -> str:
    """
    Write Laboratory Results dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Laboratory Results DataFrame.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    Returns
    -------
    str
        Written output path.
    """
    output_path = resolve_output_path(laboratory_config)

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.laboratory_results",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    return str(written_path)


def write_laboratory_metadata(
    context,
    dataframe: pd.DataFrame,
    laboratory_config: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Write Laboratory Results metadata outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    dataframe:
        Generated Laboratory Results DataFrame.

    laboratory_config:
        Parsed config/data_generation/laboratory.yaml.

    output_path:
        Written raw output path.
    """
    metadata_config = require_config_section(
        laboratory_config,
        "metadata",
        "laboratory.yaml",
    )

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        dataset_metadata = context.metadata.build_dataset_metadata(
            dataset_name="raw.laboratory_results",
            dataframe=dataframe,
            output_path=output_path,
            layer="raw",
            domain="laboratory",
            primary_key=["lab_result_id"],
            description="Canonical MedFabric synthetic Laboratory Results dataset.",
        )

        context.metadata.write_metadata(
            metadata=dataset_metadata,
            output_path="data/metadata/laboratory_results_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        column_metadata = context.metadata.build_column_metadata(
            dataset_name="raw.laboratory_results",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=column_metadata,
            output_path="data/metadata/laboratory_results_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        statistics = context.metadata.build_statistics(
            dataset_name="raw.laboratory_results",
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=statistics,
            output_path="data/metadata/laboratory_results_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name="raw.laboratory_results",
            source_datasets=[
                "raw.members",
                "raw.providers",
                "raw.facilities",
                "raw.enrollment",
                "reference.laboratory_tests",
                "reference.condition_mapping",
                "reference.result_units",
                "reference.loinc",
            ],
            output_dataset="raw.laboratory_results",
            transformation_name="generate_laboratory_dataset",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path="data/metadata/laboratory_results_lineage.json",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_laboratory_generation(context) -> pd.DataFrame:
    """
    Execute the complete Laboratory Results generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    pandas.DataFrame
        Generated Laboratory Results dataset.
    """
    global_config = load_global_generation_config(context)
    laboratory_config = load_laboratory_generation_config(context)

    datasets = load_input_and_reference_data(
        context=context,
        laboratory_config=laboratory_config,
    )

    laboratory_dataframe = generate_laboratory_dataset(
        context=context,
        global_config=global_config,
        laboratory_config=laboratory_config,
        datasets=datasets,
    )

    validate_laboratory_dataset(
        context=context,
        dataframe=laboratory_dataframe,
        laboratory_config=laboratory_config,
    )

    output_path = write_laboratory_dataset(
        context=context,
        dataframe=laboratory_dataframe,
        laboratory_config=laboratory_config,
    )

    write_laboratory_metadata(
        context=context,
        dataframe=laboratory_dataframe,
        laboratory_config=laboratory_config,
        output_path=output_path,
    )

    return laboratory_dataframe


def main() -> None:
    """
    Main entry point for Laboratory Results generation.

    Run Command
    -----------
    python -m src.data_generation.generators.laboratory_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        laboratory_dataframe = run_laboratory_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Laboratory generation completed successfully. Rows: %s",
            len(laboratory_dataframe),
        )

        print("MedFabric laboratory generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Laboratory generation failed.")
        logger.exception("Laboratory generation failed.")
        raise PipelineError("Laboratory generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()