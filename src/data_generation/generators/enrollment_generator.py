###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/generators/enrollment_generator.py
#
# Purpose:
#     Generates canonical Enrollment and Member Month datasets used by the
#     MedFabric Synthetic Data Engine.
#
# Business Context:
#     Enrollment is the payer-side contract bridge between Members and health
#     plan coverage. It determines whether a member is eligible for services,
#     which payer/product/plan is responsible, what line of business applies,
#     and which months should count in PMPM, utilization, risk, quality, and
#     population health analytics.
#
#     The Enrollment dataset represents coverage spans.
#
#     The Member Month dataset expands those coverage spans into one row per
#     member per covered month. Member Month is essential for:
#
#         - PMPM analytics
#         - Risk adjustment denominators
#         - Quality measure denominators
#         - Utilization normalization
#         - Population health cohorts
#         - Enrollment continuity analysis
#         - Member 360 coverage history
#
# Inputs:
#     config/data_generation/generation.yaml
#     config/data_generation/enrollment.yaml
#     data/raw/members.parquet
#     reference/enrollment/payer_reference.parquet
#     reference/enrollment/line_of_business_reference.parquet
#     reference/enrollment/product_reference.parquet
#     reference/enrollment/plan_reference.parquet
#     reference/enrollment/coverage_type_reference.parquet
#
# Outputs:
#     data/raw/enrollment.parquet
#     data/raw/member_months.parquet
#     data/metadata/enrollment_dataset_metadata.json
#     data/metadata/enrollment_column_metadata.csv
#     data/metadata/enrollment_statistics.json
#     data/metadata/enrollment_lineage.json
#     data/metadata/member_months_dataset_metadata.json
#     data/metadata/member_months_column_metadata.csv
#     data/metadata/member_months_statistics.json
#     data/metadata/member_months_lineage.json
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
#     6. Dataset-specific settings come from config/data_generation/enrollment.yaml.
#     7. Global execution settings come from config/data_generation/generation.yaml.
#     8. Enrollment depends on Members and enrollment reference data.
#     9. Member Month must be derived from Enrollment coverage spans.
#
# Run Command:
#     python -m src.data_generation.generators.enrollment_generator
#
# Expected Output:
#     Canonical Enrollment and Member Month datasets written to configured raw
#     output paths, with metadata written under data/metadata/.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.data_generation.enrollment"
STEP_NAME = "Generate Enrollment And Member Month Datasets"


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
    Load platform-wide generation configuration.

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


def load_enrollment_generation_config(context) -> Dict[str, Any]:
    """
    Load Enrollment-specific generation configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    dict
        Parsed config/data_generation/enrollment.yaml.
    """
    return context.configuration.load_yaml("data_generation/enrollment.yaml")


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

    Raises
    ------
    PipelineError
        Raised when the configured weights are empty or invalid.
    """
    if not weights_by_label:
        raise PipelineError(f"No weights configured for {rule_name}.")

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
    Sample one row from a reference DataFrame.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    dataframe:
        Source reference DataFrame.

    dataset_name:
        Logical dataset name used in errors.

    weight_column:
        Optional weight column. If the column exists, sampling is weighted.
        If the column does not exist, sampling is uniform.

    Returns
    -------
    dict
        Sampled row as a dictionary.

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
# INPUT AND REFERENCE LOADING
###############################################################################


def load_input_and_reference_data(
    context,
    enrollment_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all inputs and reference datasets required by Enrollment generation.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    Returns
    -------
    dict
        Dictionary containing members and enrollment reference DataFrames.

    Raises
    ------
    PipelineError
        Raised when required configuration sections or source paths are missing.

    Processing Notes
    ----------------
    This generator intentionally does not generate members, plans, payers,
    products, coverage types, or lines of business. Those objects are upstream
    inputs or reference datasets. This file links them into payer eligibility
    records.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(
        enrollment_config,
        "inputs",
        "enrollment.yaml",
    )

    payer_config = require_config_section(
        enrollment_config,
        "payer",
        "enrollment.yaml",
    )

    line_of_business_config = require_config_section(
        enrollment_config,
        "line_of_business",
        "enrollment.yaml",
    )

    product_config = require_config_section(
        enrollment_config,
        "product",
        "enrollment.yaml",
    )

    plan_config = require_config_section(
        enrollment_config,
        "plan",
        "enrollment.yaml",
    )

    coverage_type_config = require_config_section(
        enrollment_config,
        "coverage_type",
        "enrollment.yaml",
    )

    paths = {
        "members": require_config_value(
            inputs_config,
            "members_dataset",
            "enrollment.inputs",
        ),
        "payer": require_config_value(
            payer_config,
            "source",
            "enrollment.payer",
        ),
        "line_of_business": require_config_value(
            line_of_business_config,
            "source",
            "enrollment.line_of_business",
        ),
        "product": require_config_value(
            product_config,
            "source",
            "enrollment.product",
        ),
        "plan": require_config_value(
            plan_config,
            "source",
            "enrollment.plan",
        ),
        "coverage_type": require_config_value(
            coverage_type_config,
            "source",
            "enrollment.coverage_type",
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
            "Loaded Enrollment input/reference: %s | Rows: %s | Path: %s",
            dataset_name,
            len(dataframe),
            path,
        )

    return datasets


###############################################################################
# IDENTIFIER AND DATE HELPERS
###############################################################################


def resolve_random_seed(
    global_config: Dict[str, Any],
    enrollment_config: Dict[str, Any],
) -> int:
    """
    Resolve the random seed used for reproducible Enrollment generation.

    Parameters
    ----------
    global_config:
        Parsed config/data_generation/generation.yaml.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    Returns
    -------
    int
        Random seed.

    Raises
    ------
    PipelineError
        Raised when seed configuration is missing.
    """
    reproducibility_config = require_config_section(
        enrollment_config,
        "reproducibility",
        "enrollment.yaml",
    )

    use_global_random_seed = bool(
        reproducibility_config.get("use_global_random_seed", True)
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
            "enrollment.reproducibility",
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
        Identifier configuration containing prefix, starting_sequence, and
        padding_length.

    sequence_number:
        Zero-based sequence number.

    config_name:
        Human-readable configuration name used in error messages.

    Returns
    -------
    str
        Generated identifier.

    Raises
    ------
    PipelineError
        Raised when identifier configuration is incomplete.
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


def build_enrollment_id(
    enrollment_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build a configured Enrollment identifier.

    Parameters
    ----------
    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    sequence_number:
        Zero-based enrollment sequence number.

    Returns
    -------
    str
        Generated enrollment_id.
    """
    identifier_config = require_config_section(
        enrollment_config,
        "enrollment_identifier",
        "enrollment.yaml",
    )

    return build_configured_identifier(
        identifier_config=identifier_config,
        sequence_number=sequence_number,
        config_name="enrollment.enrollment_identifier",
    )


def build_member_month_id(
    enrollment_config: Dict[str, Any],
    sequence_number: int,
) -> str:
    """
    Build a configured Member Month identifier.

    Parameters
    ----------
    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    sequence_number:
        Zero-based member-month sequence number.

    Returns
    -------
    str
        Generated member_month_id.
    """
    identifier_config = require_config_section(
        enrollment_config,
        "member_month_identifier",
        "enrollment.yaml",
    )

    return build_configured_identifier(
        identifier_config=identifier_config,
        sequence_number=sequence_number,
        config_name="enrollment.member_month_identifier",
    )


def month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    """
    Normalize any timestamp to the first day of its month.

    Parameters
    ----------
    timestamp:
        Input timestamp.

    Returns
    -------
    pandas.Timestamp
        Month-start timestamp.
    """
    return pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1)


def month_end(timestamp: pd.Timestamp) -> pd.Timestamp:
    """
    Normalize any timestamp to the last day of its month.

    Parameters
    ----------
    timestamp:
        Input timestamp.

    Returns
    -------
    pandas.Timestamp
        Month-end timestamp.
    """
    return month_start(timestamp) + pd.offsets.MonthEnd(0)


def choose_start_month(
    rng: np.random.Generator,
    start_year: int,
    end_year: int,
) -> pd.Timestamp:
    """
    Choose a random coverage start month within configured years.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    start_year:
        Minimum configured enrollment start year.

    end_year:
        Maximum configured enrollment end year.

    Returns
    -------
    pandas.Timestamp
        Coverage start month as first day of month.
    """
    start_month = pd.Timestamp(
        year=int(rng.integers(start_year, end_year + 1)),
        month=int(rng.integers(1, 13)),
        day=1,
    )

    return start_month


def build_coverage_period(
    rng: np.random.Generator,
    enrollment_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one synthetic coverage period.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    Returns
    -------
    dict
        Coverage period attributes.

    Raises
    ------
    PipelineError
        Raised when configured years are invalid.

    Processing Notes
    ----------------
    The current configuration defines coverage continuity probabilities but does
    not define explicit minimum and maximum coverage-month settings. Therefore,
    this function derives coverage spans from the configured start and end year
    window:
        - continuous_coverage_probability creates coverage through the configured
          end year.
        - partial_year_coverage_probability creates a shorter same-year coverage
          span.
        - coverage_gap_probability creates a shorter span and marks the span as
          gap-prone for downstream extension.
    """
    period_config = require_config_section(
        enrollment_config,
        "enrollment_period",
        "enrollment.yaml",
    )

    continuity_config = require_config_section(
        enrollment_config,
        "coverage_continuity",
        "enrollment.yaml",
    )

    start_year = int(
        require_config_value(
            period_config,
            "start_year",
            "enrollment.enrollment_period",
        )
    )

    end_year = int(
        require_config_value(
            period_config,
            "end_year",
            "enrollment.enrollment_period",
        )
    )

    if start_year > end_year:
        raise PipelineError("enrollment_period.start_year cannot be greater than end_year.")

    continuity_type = sample_weighted_label(
        rng=rng,
        weights_by_label={
            "Continuous": float(
                require_config_value(
                    continuity_config,
                    "continuous_coverage_probability",
                    "enrollment.coverage_continuity",
                )
            ),
            "Partial Year": float(
                require_config_value(
                    continuity_config,
                    "partial_year_coverage_probability",
                    "enrollment.coverage_continuity",
                )
            ),
            "Coverage Gap": float(
                require_config_value(
                    continuity_config,
                    "coverage_gap_probability",
                    "enrollment.coverage_continuity",
                )
            ),
        },
        rule_name="enrollment.coverage_continuity",
    )

    coverage_start_month = choose_start_month(
        rng=rng,
        start_year=start_year,
        end_year=end_year,
    )

    configured_end_month = pd.Timestamp(year=end_year, month=12, day=1)

    if continuity_type == "Continuous":
        coverage_end_month = configured_end_month

    elif continuity_type == "Partial Year":
        latest_possible_month = min(
            configured_end_month,
            pd.Timestamp(year=coverage_start_month.year, month=12, day=1),
        )

        possible_months = pd.period_range(
            coverage_start_month,
            latest_possible_month,
            freq="M",
        ).to_timestamp()

        selected_position = int(rng.integers(0, len(possible_months)))
        coverage_end_month = pd.Timestamp(possible_months[selected_position])

    else:
        latest_possible_month = min(
            configured_end_month,
            coverage_start_month + pd.DateOffset(months=11),
        )

        possible_months = pd.period_range(
            coverage_start_month,
            latest_possible_month,
            freq="M",
        ).to_timestamp()

        selected_position = int(rng.integers(0, len(possible_months)))
        coverage_end_month = pd.Timestamp(possible_months[selected_position])

    coverage_start_date = month_start(coverage_start_month)
    coverage_end_date = month_end(coverage_end_month)

    if coverage_end_date < coverage_start_date:
        coverage_end_date = month_end(coverage_start_date)

    return {
        "coverage_start_date": coverage_start_date.date(),
        "coverage_end_date": coverage_end_date.date(),
        "coverage_continuity_type": continuity_type,
    }


###############################################################################
# ENROLLMENT RECORD GENERATION
###############################################################################


def get_member_id(member_row: Dict[str, Any]) -> Any:
    """
    Read member_id from an input Member record.

    Parameters
    ----------
    member_row:
        Member row dictionary.

    Returns
    -------
    Any
        Member identifier.

    Raises
    ------
    PipelineError
        Raised when member_id is missing.
    """
    member_id = member_row.get("member_id")

    if member_id is None:
        raise PipelineError("Input members dataset must contain non-null member_id.")

    return member_id


def build_enrollment_record(
    rng: np.random.Generator,
    enrollment_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    member_row: Dict[str, Any],
    sequence_number: int,
) -> Dict[str, Any]:
    """
    Build one canonical Enrollment coverage-span record.

    Parameters
    ----------
    rng:
        Active deterministic random generator.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    datasets:
        Loaded input and reference datasets.

    member_row:
        Member row dictionary.

    sequence_number:
        Zero-based enrollment sequence number.

    Returns
    -------
    dict
        Generated Enrollment record.

    Processing Notes
    ----------------
    This version creates one enrollment span per member. The configuration
    already includes continuity controls, so each span carries a continuity type.
    Future versions can expand the same pattern into multiple spans per member
    using maximum_coverage_gaps_per_member.
    """
    enrollment_id = build_enrollment_id(
        enrollment_config=enrollment_config,
        sequence_number=sequence_number,
    )

    payer_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["payer"],
        dataset_name="reference.payer",
    )

    line_of_business_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["line_of_business"],
        dataset_name="reference.line_of_business",
    )

    product_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["product"],
        dataset_name="reference.product",
    )

    plan_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["plan"],
        dataset_name="reference.plan",
    )

    coverage_type_row = sample_dataframe_row(
        rng=rng,
        dataframe=datasets["coverage_type"],
        dataset_name="reference.coverage_type",
    )

    coverage_period = build_coverage_period(
        rng=rng,
        enrollment_config=enrollment_config,
    )

    return {
        "enrollment_id": enrollment_id,
        "member_id": get_member_id(member_row),
        "payer_code": payer_row.get("payer_code"),
        "payer_name": payer_row.get("payer_name"),
        "line_of_business": line_of_business_row.get("line_of_business"),
        "product_code": product_row.get("product_code"),
        "product_description": product_row.get("product_description"),
        "plan_id": plan_row.get("plan_id"),
        "plan_name": plan_row.get("plan_name"),
        "coverage_type": coverage_type_row.get("coverage_type"),
        "coverage_start_date": coverage_period["coverage_start_date"],
        "coverage_end_date": coverage_period["coverage_end_date"],
        "coverage_continuity_type": coverage_period["coverage_continuity_type"],
        "source_system": "MedFabric Synthetic Data Engine",
        "record_status": "Active",
        "created_at": pd.Timestamp.today().normalize(),
        "updated_at": pd.Timestamp.today().normalize(),
    }


def generate_enrollment_dataset(
    context,
    global_config: Dict[str, Any],
    enrollment_config: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate canonical Enrollment coverage-span dataset.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    global_config:
        Parsed config/data_generation/generation.yaml.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    datasets:
        Loaded input and reference datasets.

    Returns
    -------
    pandas.DataFrame
        Enrollment DataFrame.
    """
    logger = context.get_logger(MODULE_NAME)

    random_seed = resolve_random_seed(
        global_config=global_config,
        enrollment_config=enrollment_config,
    )

    rng = np.random.default_rng(random_seed)
    members_dataframe = datasets["members"]

    logger.info(
        "Generating Enrollment dataset from Member input. Member rows: %s",
        len(members_dataframe),
    )

    records: List[Dict[str, Any]] = []

    for sequence_number, member_row in enumerate(members_dataframe.to_dict("records")):
        records.append(
            build_enrollment_record(
                rng=rng,
                enrollment_config=enrollment_config,
                datasets=datasets,
                member_row=member_row,
                sequence_number=sequence_number,
            )
        )

    enrollment_dataframe = pd.DataFrame(records)

    logger.info(
        "Generated Enrollment dataset. Rows: %s",
        len(enrollment_dataframe),
    )

    return enrollment_dataframe


###############################################################################
# MEMBER MONTH GENERATION
###############################################################################


def iter_covered_months(
    coverage_start_date: Any,
    coverage_end_date: Any,
) -> List[pd.Timestamp]:
    """
    Expand coverage dates into covered month-start timestamps.

    Parameters
    ----------
    coverage_start_date:
        Coverage start date.

    coverage_end_date:
        Coverage end date.

    Returns
    -------
    list[pandas.Timestamp]
        Covered months represented as month-start timestamps.

    Raises
    ------
    PipelineError
        Raised when coverage dates are invalid.
    """
    start_timestamp = month_start(pd.Timestamp(coverage_start_date))
    end_timestamp = month_start(pd.Timestamp(coverage_end_date))

    if end_timestamp < start_timestamp:
        raise PipelineError(
            "coverage_end_date cannot be before coverage_start_date "
            "when building member months."
        )

    return list(pd.period_range(start_timestamp, end_timestamp, freq="M").to_timestamp())


def build_member_month_record(
    enrollment_config: Dict[str, Any],
    enrollment_row: Dict[str, Any],
    member_month_sequence_number: int,
    coverage_month: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Build one canonical Member Month record.

    Parameters
    ----------
    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    enrollment_row:
        Enrollment row dictionary.

    member_month_sequence_number:
        Zero-based member-month sequence number.

    coverage_month:
        Covered month represented as a month-start timestamp.

    Returns
    -------
    dict
        Generated Member Month record.
    """
    member_month_id = build_member_month_id(
        enrollment_config=enrollment_config,
        sequence_number=member_month_sequence_number,
    )

    return {
        "member_month_id": member_month_id,
        "enrollment_id": enrollment_row.get("enrollment_id"),
        "member_id": enrollment_row.get("member_id"),
        "coverage_month": coverage_month.date(),
        "coverage_year": int(coverage_month.year),
        "coverage_month_number": int(coverage_month.month),
        "payer_code": enrollment_row.get("payer_code"),
        "payer_name": enrollment_row.get("payer_name"),
        "line_of_business": enrollment_row.get("line_of_business"),
        "product_code": enrollment_row.get("product_code"),
        "product_description": enrollment_row.get("product_description"),
        "plan_id": enrollment_row.get("plan_id"),
        "plan_name": enrollment_row.get("plan_name"),
        "coverage_type": enrollment_row.get("coverage_type"),
        "coverage_start_date": enrollment_row.get("coverage_start_date"),
        "coverage_end_date": enrollment_row.get("coverage_end_date"),
        "coverage_continuity_type": enrollment_row.get("coverage_continuity_type"),
        "member_month_flag": 1,
        "source_system": enrollment_row.get("source_system"),
        "record_status": enrollment_row.get("record_status"),
        "created_at": enrollment_row.get("created_at"),
        "updated_at": enrollment_row.get("updated_at"),
    }


def generate_member_month_dataset(
    context,
    enrollment_config: Dict[str, Any],
    enrollment_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate canonical Member Month dataset from Enrollment spans.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    enrollment_dataframe:
        Generated Enrollment DataFrame.

    Returns
    -------
    pandas.DataFrame
        Member Month DataFrame.

    Processing Notes
    ----------------
    Member Month rows are derived from enrollment spans and should not be
    independently randomized. This preserves consistency between coverage spans
    and monthly denominator rows.
    """
    logger = context.get_logger(MODULE_NAME)

    period_config = require_config_section(
        enrollment_config,
        "enrollment_period",
        "enrollment.yaml",
    )

    generate_member_months = bool(
        require_config_value(
            period_config,
            "generate_member_months",
            "enrollment.enrollment_period",
        )
    )

    if not generate_member_months:
        logger.info("Member Month generation disabled by enrollment.yaml.")
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    member_month_sequence_number = 0

    for enrollment_row in enrollment_dataframe.to_dict("records"):
        covered_months = iter_covered_months(
            coverage_start_date=enrollment_row.get("coverage_start_date"),
            coverage_end_date=enrollment_row.get("coverage_end_date"),
        )

        for coverage_month in covered_months:
            records.append(
                build_member_month_record(
                    enrollment_config=enrollment_config,
                    enrollment_row=enrollment_row,
                    member_month_sequence_number=member_month_sequence_number,
                    coverage_month=coverage_month,
                )
            )

            member_month_sequence_number += 1

    member_month_dataframe = pd.DataFrame(records)

    logger.info(
        "Generated Member Month dataset. Rows: %s",
        len(member_month_dataframe),
    )

    return member_month_dataframe


###############################################################################
# VALIDATION
###############################################################################


def build_enrollment_validation_rules(enrollment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager rules for the Enrollment dataset.

    Parameters
    ----------
    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        enrollment_config,
        "validation",
        "enrollment.yaml",
    )

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_enrollment_id", False)):
        required_columns.append("enrollment_id")
        no_null_columns.append("enrollment_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_coverage_start_date", False)):
        required_columns.append("coverage_start_date")
        no_null_columns.append("coverage_start_date")

    if bool(validation_config.get("require_coverage_end_date", False)):
        required_columns.append("coverage_end_date")
        no_null_columns.append("coverage_end_date")

    if bool(validation_config.get("require_line_of_business", False)):
        required_columns.append("line_of_business")
        no_null_columns.append("line_of_business")

    if bool(validation_config.get("require_product", False)):
        required_columns.append("product_code")
        no_null_columns.append("product_code")

    if bool(validation_config.get("require_plan", False)):
        required_columns.append("plan_id")
        no_null_columns.append("plan_id")

    if bool(validation_config.get("require_payer", False)):
        required_columns.append("payer_code")
        no_null_columns.append("payer_code")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_enrollment_ids", False)):
        validation_rules["primary_key"] = ["enrollment_id"]

    return validation_rules


def build_member_month_validation_rules(enrollment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ValidationManager rules for the Member Month dataset.

    Parameters
    ----------
    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    Returns
    -------
    dict
        Validation rule dictionary.
    """
    validation_config = require_config_section(
        enrollment_config,
        "validation",
        "enrollment.yaml",
    )

    required_columns: List[str] = []
    no_null_columns: List[str] = []

    if bool(validation_config.get("require_member_month_id", False)):
        required_columns.append("member_month_id")
        no_null_columns.append("member_month_id")

    if bool(validation_config.get("require_member_id", False)):
        required_columns.append("member_id")
        no_null_columns.append("member_id")

    if bool(validation_config.get("require_enrollment_id", False)):
        required_columns.append("enrollment_id")
        no_null_columns.append("enrollment_id")

    if bool(validation_config.get("require_line_of_business", False)):
        required_columns.append("line_of_business")
        no_null_columns.append("line_of_business")

    if bool(validation_config.get("require_product", False)):
        required_columns.append("product_code")
        no_null_columns.append("product_code")

    if bool(validation_config.get("require_plan", False)):
        required_columns.append("plan_id")
        no_null_columns.append("plan_id")

    if bool(validation_config.get("require_payer", False)):
        required_columns.append("payer_code")
        no_null_columns.append("payer_code")

    validation_rules: Dict[str, Any] = {
        "allow_empty": False,
        "min_rows": 1,
        "required_columns": required_columns,
        "no_nulls": no_null_columns,
    }

    if bool(validation_config.get("require_unique_member_month_ids", False)):
        validation_rules["primary_key"] = ["member_month_id"]

    return validation_rules


def validate_generated_outputs(
    context,
    enrollment_config: Dict[str, Any],
    enrollment_dataframe: pd.DataFrame,
    member_month_dataframe: pd.DataFrame,
) -> None:
    """
    Validate Enrollment and Member Month outputs.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    enrollment_dataframe:
        Generated Enrollment DataFrame.

    member_month_dataframe:
        Generated Member Month DataFrame.
    """
    context.validation.validate_dataset(
        dataframe=enrollment_dataframe,
        validation_rules=build_enrollment_validation_rules(enrollment_config),
        dataset_name="raw.enrollment",
    )

    period_config = require_config_section(
        enrollment_config,
        "enrollment_period",
        "enrollment.yaml",
    )

    if bool(period_config.get("generate_member_months", False)):
        context.validation.validate_dataset(
            dataframe=member_month_dataframe,
            validation_rules=build_member_month_validation_rules(enrollment_config),
            dataset_name="raw.member_months",
        )


###############################################################################
# OUTPUT RESOLUTION
###############################################################################


def resolve_output_paths(enrollment_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve Enrollment and Member Month output paths.

    Parameters
    ----------
    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    Returns
    -------
    dict
        Output paths keyed by logical dataset.
    """
    output_config = require_config_section(
        enrollment_config,
        "output",
        "enrollment.yaml",
    )

    enrollment_file_name = require_config_value(
        output_config,
        "enrollment_file_name",
        "enrollment.output",
    )

    member_month_file_name = require_config_value(
        output_config,
        "member_month_file_name",
        "enrollment.output",
    )

    return {
        "enrollment": f"data/raw/{enrollment_file_name}",
        "member_months": f"data/raw/{member_month_file_name}",
    }


def write_generated_outputs(
    context,
    enrollment_config: Dict[str, Any],
    enrollment_dataframe: pd.DataFrame,
    member_month_dataframe: pd.DataFrame,
) -> Dict[str, str]:
    """
    Write Enrollment and Member Month datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    enrollment_dataframe:
        Generated Enrollment DataFrame.

    member_month_dataframe:
        Generated Member Month DataFrame.

    Returns
    -------
    dict
        Written output paths.
    """
    output_paths = resolve_output_paths(enrollment_config)

    written_enrollment_path = context.storage.write_parquet(
        dataframe=enrollment_dataframe,
        path=output_paths["enrollment"],
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.enrollment",
        row_count=len(enrollment_dataframe),
        column_count=len(enrollment_dataframe.columns),
        path=written_enrollment_path,
    )

    written_member_month_path = context.storage.write_parquet(
        dataframe=member_month_dataframe,
        path=output_paths["member_months"],
        index=False,
    )

    context.logging.log_dataset(
        dataset_name="raw.member_months",
        row_count=len(member_month_dataframe),
        column_count=len(member_month_dataframe.columns),
        path=written_member_month_path,
    )

    return {
        "enrollment": str(written_enrollment_path),
        "member_months": str(written_member_month_path),
    }


###############################################################################
# METADATA
###############################################################################


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
    Write dataset metadata, column metadata, statistics, and lineage.

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
        Written dataset path.

    primary_key:
        Primary key columns.

    description:
        Dataset description.

    metadata_prefix:
        Prefix used for metadata output file names.

    source_datasets:
        Source datasets used for lineage.

    transformation_name:
        Transformation name used for lineage metadata.
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


def write_generated_metadata(
    context,
    enrollment_config: Dict[str, Any],
    enrollment_dataframe: pd.DataFrame,
    member_month_dataframe: pd.DataFrame,
    output_paths: Dict[str, str],
) -> None:
    """
    Write metadata outputs when enabled by configuration.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    enrollment_config:
        Parsed config/data_generation/enrollment.yaml.

    enrollment_dataframe:
        Generated Enrollment DataFrame.

    member_month_dataframe:
        Generated Member Month DataFrame.

    output_paths:
        Written raw dataset paths.
    """
    metadata_config = require_config_section(
        enrollment_config,
        "metadata",
        "enrollment.yaml",
    )

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

    write_single_dataset_metadata(
        context=context,
        dataframe=enrollment_dataframe,
        dataset_name="raw.enrollment",
        domain="enrollment",
        output_path=output_paths["enrollment"],
        primary_key=["enrollment_id"],
        description="Canonical MedFabric synthetic Enrollment coverage-span dataset.",
        metadata_prefix="enrollment",
        source_datasets=[
            "raw.members",
            "reference.payer",
            "reference.line_of_business",
            "reference.product",
            "reference.plan",
            "reference.coverage_type",
        ],
        transformation_name="generate_enrollment_dataset",
    )

    period_config = require_config_section(
        enrollment_config,
        "enrollment_period",
        "enrollment.yaml",
    )

    if bool(period_config.get("generate_member_months", False)):
        write_single_dataset_metadata(
            context=context,
            dataframe=member_month_dataframe,
            dataset_name="raw.member_months",
            domain="enrollment",
            output_path=output_paths["member_months"],
            primary_key=["member_month_id"],
            description="Canonical MedFabric synthetic Member Month denominator dataset.",
            metadata_prefix="member_months",
            source_datasets=["raw.enrollment"],
            transformation_name="generate_member_month_dataset",
        )


###############################################################################
# ORCHESTRATION
###############################################################################


def run_enrollment_generation(context) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute complete Enrollment and Member Month generation lifecycle.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Generated Enrollment and Member Month DataFrames.
    """
    global_config = load_global_generation_config(context)
    enrollment_config = load_enrollment_generation_config(context)

    datasets = load_input_and_reference_data(
        context=context,
        enrollment_config=enrollment_config,
    )

    enrollment_dataframe = generate_enrollment_dataset(
        context=context,
        global_config=global_config,
        enrollment_config=enrollment_config,
        datasets=datasets,
    )

    member_month_dataframe = generate_member_month_dataset(
        context=context,
        enrollment_config=enrollment_config,
        enrollment_dataframe=enrollment_dataframe,
    )

    validate_generated_outputs(
        context=context,
        enrollment_config=enrollment_config,
        enrollment_dataframe=enrollment_dataframe,
        member_month_dataframe=member_month_dataframe,
    )

    output_paths = write_generated_outputs(
        context=context,
        enrollment_config=enrollment_config,
        enrollment_dataframe=enrollment_dataframe,
        member_month_dataframe=member_month_dataframe,
    )

    write_generated_metadata(
        context=context,
        enrollment_config=enrollment_config,
        enrollment_dataframe=enrollment_dataframe,
        member_month_dataframe=member_month_dataframe,
        output_paths=output_paths,
    )

    return enrollment_dataframe, member_month_dataframe


def main() -> None:
    """
    Main entry point for Enrollment generation.

    Run Command
    -----------
    python -m src.data_generation.generators.enrollment_generator
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        enrollment_dataframe, member_month_dataframe = run_enrollment_generation(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Enrollment generation completed successfully. "
            "Enrollment rows: %s | Member Month rows: %s",
            len(enrollment_dataframe),
            len(member_month_dataframe),
        )

        print("MedFabric enrollment generation completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Enrollment generation failed.")
        logger.exception("Enrollment generation failed.")
        raise PipelineError("Enrollment generation failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()