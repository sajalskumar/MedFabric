###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/feature_store/build_feature_store.py
#
# Purpose:
#     Builds the MedFabric Feature Store from Gold analytical marts.
#
# Business Context:
#     The Feature Store creates reusable, versioned machine-learning-ready
#     features from Gold datasets. These features support future predictive
#     models, population health analytics, risk scoring, provider analytics,
#     and care management analytics.
#
# Feature Store Philosophy:
#     Build Once • Reuse Everywhere
#
# Inputs:
#     config/feature_store/feature_store.yaml
#     data/gold/*.parquet
#
# Outputs:
#     data/feature_store/*.parquet
#     data/metadata/feature_store_*_dataset_metadata.json
#     data/metadata/feature_store_*_column_metadata.csv
#     data/metadata/feature_store_*_statistics.json
#     data/metadata/feature_store_*_lineage.json
#
# Dependencies:
#     pandas
#     src.common.pipeline_context.create_pipeline_context
#     src.common.exception_manager.PipelineError
#
# Architectural Rules:
#     1. Feature Store consumes Gold datasets only.
#     2. Feature Store does not read Raw, Bronze, or Silver datasets directly.
#     3. Feature Store creates reusable model-ready feature groups.
#     4. Feature groups must stay within the frozen roadmap scope.
#     5. Feature Store outputs must generate validation, metadata, lineage,
#        and logging.
#
# Run Command:
#     python -m src.feature_store.build_feature_store
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.feature_store"
STEP_NAME = "Build Feature Store"
FEATURE_STORE_CONFIG_PATH = "feature_store/feature_store.yaml"


def require_config_value(config: Dict[str, Any], key: str, config_name: str) -> Any:
    """
    Read a required configuration value.
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
    """
    value = require_config_value(config, key, config_name)

    if not isinstance(value, dict):
        raise PipelineError(
            f"Configuration section '{key}' in {config_name} must be a mapping."
        )

    return value


def load_feature_store_config(context) -> Dict[str, Any]:
    """
    Load config/feature_store/feature_store.yaml.
    """
    return context.configuration.load_yaml(FEATURE_STORE_CONFIG_PATH)


def load_feature_store_inputs(
    context,
    feature_store_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all configured Gold inputs for the Feature Store.
    """
    logger = context.get_logger(MODULE_NAME)

    inputs_config = require_config_section(
        feature_store_config,
        "inputs",
        "feature_store.yaml",
    )

    inputs: Dict[str, pd.DataFrame] = {}

    for input_name, input_path in inputs_config.items():
        dataframe = context.storage.read_parquet(input_path)

        context.validation.validate_dataset(
            dataframe=dataframe,
            validation_rules={
                "allow_empty": False,
                "min_rows": 1,
            },
            dataset_name=f"gold.{input_name}",
        )

        inputs[input_name] = dataframe

        logger.info(
            "Loaded Feature Store input: %s | Rows: %s | Path: %s",
            input_name,
            len(dataframe),
            input_path,
        )

    return inputs


def add_feature_store_audit_columns(
    context,
    dataframe: pd.DataFrame,
    feature_store_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add standard Feature Store audit columns.
    """
    audit_config = require_config_section(
        feature_store_config,
        "audit",
        "feature_store.yaml",
    )

    dataframe[
        require_config_value(
            audit_config,
            "feature_store_run_id_column",
            "feature_store.audit",
        )
    ] = context.run_id

    dataframe[
        require_config_value(
            audit_config,
            "feature_store_load_timestamp_column",
            "feature_store.audit",
        )
    ] = pd.Timestamp.utcnow()

    dataframe[
        require_config_value(
            audit_config,
            "feature_store_record_status_column",
            "feature_store.audit",
        )
    ] = require_config_value(
        audit_config,
        "default_record_status",
        "feature_store.audit",
    )

    return dataframe


def validate_feature_group(
    context,
    dataframe: pd.DataFrame,
    dataset_name: str,
    entity_key: List[str],
    feature_store_config: Dict[str, Any],
) -> None:
    """
    Validate one Feature Store feature group.
    """
    validation_config = require_config_section(
        feature_store_config,
        "validation",
        "feature_store.yaml",
    )

    validation_rules: Dict[str, Any] = {
        "allow_empty": bool(validation_config.get("allow_empty", False)),
        "required_columns": entity_key,
        "no_nulls": entity_key,
    }

    if not bool(validation_config.get("allow_empty", False)):
        validation_rules["min_rows"] = 1

    if entity_key:
        validation_rules["primary_key"] = entity_key

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=validation_rules,
        dataset_name=dataset_name,
    )


def write_feature_group_metadata(
    context,
    dataframe: pd.DataFrame,
    feature_group_key: str,
    dataset_name: str,
    output_path: str,
    entity_key: List[str],
    source_datasets: List[str],
    feature_store_config: Dict[str, Any],
) -> None:
    """
    Write metadata, statistics, and lineage for one feature group.
    """
    metadata_config = require_config_section(
        feature_store_config,
        "metadata",
        "feature_store.yaml",
    )

    metadata_prefix = f"feature_store_{feature_group_key}"

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        metadata = context.metadata.build_dataset_metadata(
            dataset_name=dataset_name,
            dataframe=dataframe,
            output_path=output_path,
            layer="feature_store",
            domain=feature_group_key,
            primary_key=entity_key,
            description=f"Reusable Feature Store feature group for {dataset_name}.",
        )

        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        metadata = context.metadata.build_column_metadata(
            dataset_name=dataset_name,
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        metadata = context.metadata.build_statistics(
            dataset_name=dataset_name,
            dataframe=dataframe,
        )

        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name=dataset_name,
            source_datasets=source_datasets,
            output_dataset=dataset_name,
            transformation_name=f"build_{feature_group_key}",
            module_name=MODULE_NAME,
        )

        context.metadata.write_metadata(
            metadata=lineage,
            output_path=f"data/metadata/{metadata_prefix}_lineage.json",
        )


def write_feature_group(
    context,
    dataframe: pd.DataFrame,
    feature_group_key: str,
    feature_group_config: Dict[str, Any],
    source_datasets: List[str],
    feature_store_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Validate, audit, write, and document one Feature Store feature group.
    """
    dataset_name = require_config_value(
        feature_group_config,
        "dataset_name",
        f"feature_store.feature_groups.{feature_group_key}",
    )

    output_path = require_config_value(
        feature_group_config,
        "output_path",
        f"feature_store.feature_groups.{feature_group_key}",
    )

    entity_key = feature_group_config.get("entity_key", [])

    dataframe = add_feature_store_audit_columns(
        context=context,
        dataframe=dataframe.copy(deep=True),
        feature_store_config=feature_store_config,
    )

    validate_feature_group(
        context=context,
        dataframe=dataframe,
        dataset_name=dataset_name,
        entity_key=entity_key,
        feature_store_config=feature_store_config,
    )

    written_path = context.storage.write_parquet(
        dataframe=dataframe,
        path=output_path,
        index=False,
    )

    context.logging.log_dataset(
        dataset_name=dataset_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    write_feature_group_metadata(
        context=context,
        dataframe=dataframe,
        feature_group_key=feature_group_key,
        dataset_name=dataset_name,
        output_path=str(written_path),
        entity_key=entity_key,
        source_datasets=source_datasets,
        feature_store_config=feature_store_config,
    )

    return dataframe


def safe_select_columns(
    dataframe: pd.DataFrame,
    requested_columns: List[str],
) -> pd.DataFrame:
    """
    Select columns that exist in the DataFrame.
    """
    available_columns = [
        column_name
        for column_name in requested_columns
        if column_name in dataframe.columns
    ]

    return dataframe[available_columns].copy()


def build_demographic_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member demographic features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "gender",
        "birth_date",
        "age_years",
        "race_code",
        "race_description",
        "ethnicity_code",
        "ethnicity_description",
        "language_code",
        "language_description",
        "zip_code",
        "city",
        "county",
        "state_code",
        "state_name",
        "region",
    ]

    features = safe_select_columns(member_360, requested_columns)

    if "age_years" in features.columns:
        features["age_band"] = pd.cut(
            features["age_years"],
            bins=[-1, 17, 44, 64, 200],
            labels=["0-17", "18-44", "45-64", "65+"],
        ).astype("string")

    return features


def build_enrollment_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member enrollment features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "enrollment_id",
        "line_of_business",
        "product_code",
        "product_description",
        "plan_id",
        "plan_name",
        "coverage_type",
        "coverage_start_date",
        "coverage_end_date",
        "enrollment_status",
    ]

    features = safe_select_columns(member_360, requested_columns)

    if "coverage_start_date" in features.columns and "coverage_end_date" in features.columns:
        features["coverage_start_date"] = pd.to_datetime(features["coverage_start_date"], errors="coerce")
        features["coverage_end_date"] = pd.to_datetime(features["coverage_end_date"], errors="coerce")
        features["coverage_days"] = (
            features["coverage_end_date"] - features["coverage_start_date"]
        ).dt.days

    return features


def build_claims_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member claims features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "total_claims",
        "total_allowed_amount",
        "total_paid_amount",
    ]

    features = safe_select_columns(member_360, requested_columns)

    if "total_claims" in features.columns:
        features["has_claim_activity"] = features["total_claims"].fillna(0) > 0

    return features


def build_cost_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member cost features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "total_allowed_amount",
        "total_paid_amount",
        "total_pharmacy_paid_amount",
    ]

    features = safe_select_columns(member_360, requested_columns)

    numeric_columns = [
        "total_allowed_amount",
        "total_paid_amount",
        "total_pharmacy_paid_amount",
    ]

    for column_name in numeric_columns:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    if "total_paid_amount" in features.columns:
        features["high_cost_member_flag"] = features["total_paid_amount"] >= features["total_paid_amount"].quantile(0.90)

    return features


def build_utilization_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member utilization features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "total_claims",
        "total_lab_results",
        "total_pharmacy_claims",
    ]

    features = safe_select_columns(member_360, requested_columns)

    for column_name in ["total_claims", "total_lab_results", "total_pharmacy_claims"]:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    return features


def build_laboratory_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member laboratory features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "total_lab_results",
        "abnormal_lab_results",
    ]

    features = safe_select_columns(member_360, requested_columns)

    for column_name in ["total_lab_results", "abnormal_lab_results"]:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    if "total_lab_results" in features.columns and "abnormal_lab_results" in features.columns:
        features["abnormal_lab_rate"] = features["abnormal_lab_results"] / features["total_lab_results"].replace(0, pd.NA)
        features["abnormal_lab_rate"] = features["abnormal_lab_rate"].fillna(0)

    return features


def build_pharmacy_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member pharmacy features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "total_pharmacy_claims",
        "total_pharmacy_paid_amount",
    ]

    features = safe_select_columns(member_360, requested_columns)

    for column_name in ["total_pharmacy_claims", "total_pharmacy_paid_amount"]:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    if "total_pharmacy_claims" in features.columns:
        features["has_pharmacy_activity"] = features["total_pharmacy_claims"] > 0

    return features


def build_sdoh_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build member SDOH features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "sdoh_assessments",
        "latest_sdoh_risk_score",
    ]

    features = safe_select_columns(member_360, requested_columns)

    for column_name in ["sdoh_assessments", "latest_sdoh_risk_score"]:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    if "latest_sdoh_risk_score" in features.columns:
        features["high_sdoh_risk_flag"] = features["latest_sdoh_risk_score"] >= 8

    return features


def build_provider_attribution_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build provider-level attribution features from Gold Provider Performance.
    """
    provider_performance = inputs["provider_performance"]

    requested_columns = [
        "provider_id",
        "claim_count",
        "encounter_count",
        "total_paid_amount",
        "total_allowed_amount",
    ]

    features = safe_select_columns(provider_performance, requested_columns)

    for column_name in ["claim_count", "encounter_count", "total_paid_amount", "total_allowed_amount"]:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    if "claim_count" in features.columns:
        features["active_provider_flag"] = features["claim_count"] > 0

    return features


def build_temporal_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build simple member temporal features from Gold Member 360.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "coverage_start_date",
        "coverage_end_date",
    ]

    features = safe_select_columns(member_360, requested_columns)

    for column_name in ["coverage_start_date", "coverage_end_date"]:
        if column_name in features.columns:
            features[column_name] = pd.to_datetime(features[column_name], errors="coerce")

    if "coverage_start_date" in features.columns:
        features["coverage_start_year"] = features["coverage_start_date"].dt.year
        features["coverage_start_month"] = features["coverage_start_date"].dt.month

    if "coverage_end_date" in features.columns:
        features["coverage_end_year"] = features["coverage_end_date"].dt.year
        features["coverage_end_month"] = features["coverage_end_date"].dt.month

    return features


def build_risk_features(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build reusable member risk features from multiple Feature Store concepts.
    """
    member_360 = inputs["member_360"]

    requested_columns = [
        "member_id",
        "age_years",
        "total_claims",
        "total_paid_amount",
        "total_pharmacy_claims",
        "total_lab_results",
        "abnormal_lab_results",
        "latest_sdoh_risk_score",
    ]

    features = safe_select_columns(member_360, requested_columns)

    numeric_columns = [
        "age_years",
        "total_claims",
        "total_paid_amount",
        "total_pharmacy_claims",
        "total_lab_results",
        "abnormal_lab_results",
        "latest_sdoh_risk_score",
    ]

    for column_name in numeric_columns:
        if column_name in features.columns:
            features[column_name] = features[column_name].fillna(0)

    if "total_paid_amount" in features.columns:
        features["high_cost_risk_signal"] = features["total_paid_amount"] >= features["total_paid_amount"].quantile(0.90)

    if "latest_sdoh_risk_score" in features.columns:
        features["sdoh_risk_signal"] = features["latest_sdoh_risk_score"] >= 8

    if "abnormal_lab_results" in features.columns:
        features["clinical_risk_signal"] = features["abnormal_lab_results"] > 0

    return features


def build_feature_group(
    feature_group_key: str,
    inputs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Dispatch feature group build by approved feature group key.
    """
    builders = {
        "demographic_features": build_demographic_features,
        "enrollment_features": build_enrollment_features,
        "claims_features": build_claims_features,
        "cost_features": build_cost_features,
        "utilization_features": build_utilization_features,
        "laboratory_features": build_laboratory_features,
        "pharmacy_features": build_pharmacy_features,
        "sdoh_features": build_sdoh_features,
        "provider_attribution_features": build_provider_attribution_features,
        "temporal_features": build_temporal_features,
        "risk_features": build_risk_features,
    }

    if feature_group_key not in builders:
        raise PipelineError(f"Unsupported Feature Store group key: {feature_group_key}")

    return builders[feature_group_key](inputs)


def build_feature_store(context) -> Dict[str, pd.DataFrame]:
    """
    Build all enabled Feature Store feature groups.
    """
    logger = context.get_logger(MODULE_NAME)

    feature_store_config = load_feature_store_config(context)

    configuration_config = require_config_section(
        feature_store_config,
        "configuration",
        "feature_store.yaml",
    )

    if not bool(configuration_config.get("enabled", True)):
        logger.warning("Feature Store is disabled in config/feature_store/feature_store.yaml.")
        return {}

    inputs = load_feature_store_inputs(
        context=context,
        feature_store_config=feature_store_config,
    )

    feature_groups_config = require_config_section(
        feature_store_config,
        "feature_groups",
        "feature_store.yaml",
    )

    outputs: Dict[str, pd.DataFrame] = {}

    for feature_group_key, feature_group_config in feature_groups_config.items():
        if not bool(feature_group_config.get("enabled", True)):
            continue

        logger.info("START Feature Group: %s", feature_group_key)

        feature_dataframe = build_feature_group(
            feature_group_key=feature_group_key,
            inputs=inputs,
        )

        outputs[feature_group_key] = write_feature_group(
            context=context,
            dataframe=feature_dataframe,
            feature_group_key=feature_group_key,
            feature_group_config=feature_group_config,
            source_datasets=[f"gold.{name}" for name in inputs.keys()],
            feature_store_config=feature_store_config,
        )

        logger.info(
            "COMPLETE Feature Group: %s | Rows: %s",
            feature_group_key,
            len(outputs[feature_group_key]),
        )

    return outputs


def main() -> None:
    """
    Main entry point for Feature Store build.

    Run Command
    -----------
    python -m src.feature_store.build_feature_store
    """
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        outputs = build_feature_store(context)

        context.logging.end_step(STEP_NAME)

        logger.info(
            "MedFabric Feature Store completed successfully. Feature groups built: %s",
            len(outputs),
        )

        print("MedFabric Feature Store completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Feature Store build failed.")
        logger.exception("Feature Store build failed.")
        raise PipelineError("Feature Store build failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()