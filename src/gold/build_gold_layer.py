###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/gold/build_gold_layer.py
#
# Purpose:
#     Builds the MedFabric Gold Layer analytical marts from Silver datasets.
#
# Business Context:
#     The Gold Layer provides business-ready healthcare analytics datasets used
#     for executive reporting, payer analytics, population health, provider
#     performance, utilization monitoring, PMPM reporting, cost analysis,
#     pharmacy analytics, laboratory analytics, and SDOH analytics.
#
# Gold Philosophy:
#     Gold = Summarize + Analyze
#
# Inputs:
#     config/gold/gold.yaml
#     data/silver/*.parquet
#
# Outputs:
#     data/gold/member_360.parquet
#     data/gold/enrollment_summary.parquet
#     data/gold/utilization_summary.parquet
#     data/gold/cost_summary.parquet
#     data/gold/pmpm_summary.parquet
#     data/gold/provider_performance.parquet
#     data/gold/organization_performance.parquet
#     data/gold/facility_performance.parquet
#     data/gold/clinical_summary.parquet
#     data/gold/pharmacy_summary.parquet
#     data/gold/laboratory_summary.parquet
#     data/gold/sdoh_summary.parquet
#
# Dependencies:
#     pandas
#     src.common.pipeline_context.create_pipeline_context
#     src.common.exception_manager.PipelineError
#
# Architectural Rules:
#     1. Gold consumes Silver datasets only.
#     2. Gold creates business-ready analytical marts.
#     3. Gold does not read Raw or Bronze datasets.
#     4. Gold calculations must stay within the frozen Gold roadmap.
#     5. Gold outputs must generate validation, metadata, lineage, and logging.
#
# Run Command:
#     python -m src.gold.build_gold_layer
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


MODULE_NAME = "medfabric.gold"
STEP_NAME = "Build Gold Layer"
GOLD_CONFIG_PATH = "gold/gold.yaml"


def require_config_value(config: Dict[str, Any], key: str, config_name: str) -> Any:
    """Read a required configuration value."""
    if key not in config:
        raise PipelineError(f"Missing required configuration value '{key}' in {config_name}.")
    return config[key]


def require_config_section(config: Dict[str, Any], key: str, config_name: str) -> Dict[str, Any]:
    """Read a required configuration section."""
    value = require_config_value(config, key, config_name)
    if not isinstance(value, dict):
        raise PipelineError(f"Configuration section '{key}' in {config_name} must be a mapping.")
    return value


def load_gold_config(context) -> Dict[str, Any]:
    """Load config/gold/gold.yaml."""
    return context.configuration.load_yaml(GOLD_CONFIG_PATH)


def load_gold_inputs(context, gold_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load all Silver inputs configured for Gold."""
    logger = context.get_logger(MODULE_NAME)
    inputs_config = require_config_section(gold_config, "inputs", "gold.yaml")

    inputs: Dict[str, pd.DataFrame] = {}

    for input_name, input_path in inputs_config.items():
        dataframe = context.storage.read_parquet(input_path)
        context.validation.validate_dataset(
            dataframe=dataframe,
            validation_rules={"allow_empty": False, "min_rows": 1},
            dataset_name=f"silver.{input_name}",
        )
        inputs[input_name] = dataframe
        logger.info("Loaded Gold input: %s | Rows: %s | Path: %s", input_name, len(dataframe), input_path)

    return inputs


def add_gold_audit_columns(context, dataframe: pd.DataFrame, gold_config: Dict[str, Any]) -> pd.DataFrame:
    """Add standard Gold audit columns."""
    audit_config = require_config_section(gold_config, "audit", "gold.yaml")

    dataframe[require_config_value(audit_config, "gold_run_id_column", "gold.audit")] = context.run_id
    dataframe[require_config_value(audit_config, "gold_load_timestamp_column", "gold.audit")] = pd.Timestamp.utcnow()
    dataframe[require_config_value(audit_config, "gold_record_status_column", "gold.audit")] = require_config_value(
        audit_config,
        "default_record_status",
        "gold.audit",
    )

    return dataframe


def validate_gold_dataset(
    context,
    dataframe: pd.DataFrame,
    dataset_name: str,
    primary_key: List[str],
    gold_config: Dict[str, Any],
) -> None:
    """Validate a Gold dataset."""
    validation_config = require_config_section(gold_config, "validation", "gold.yaml")

    validation_rules: Dict[str, Any] = {
        "allow_empty": bool(validation_config.get("allow_empty", False)),
        "required_columns": primary_key,
        "no_nulls": primary_key,
    }

    if not bool(validation_config.get("allow_empty", False)):
        validation_rules["min_rows"] = 1

    if primary_key:
        validation_rules["primary_key"] = primary_key

    context.validation.validate_dataset(
        dataframe=dataframe,
        validation_rules=validation_rules,
        dataset_name=dataset_name,
    )


def write_gold_metadata(
    context,
    dataframe: pd.DataFrame,
    mart_key: str,
    dataset_name: str,
    output_path: str,
    primary_key: List[str],
    source_datasets: List[str],
    gold_config: Dict[str, Any],
) -> None:
    """Write dataset metadata, column metadata, statistics, and lineage."""
    metadata_config = require_config_section(gold_config, "metadata", "gold.yaml")
    metadata_prefix = f"gold_{mart_key}"

    if bool(metadata_config.get("generate_dataset_metadata", False)):
        metadata = context.metadata.build_dataset_metadata(
            dataset_name=dataset_name,
            dataframe=dataframe,
            output_path=output_path,
            layer="gold",
            domain=mart_key,
            primary_key=primary_key,
            description=f"Gold analytical mart for {dataset_name}.",
        )
        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_dataset_metadata.json",
        )

    if bool(metadata_config.get("generate_column_metadata", False)):
        metadata = context.metadata.build_column_metadata(dataset_name=dataset_name, dataframe=dataframe)
        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_column_metadata.csv",
        )

    if bool(metadata_config.get("generate_statistics", False)):
        metadata = context.metadata.build_statistics(dataset_name=dataset_name, dataframe=dataframe)
        context.metadata.write_metadata(
            metadata=metadata,
            output_path=f"data/metadata/{metadata_prefix}_statistics.json",
        )

    if bool(metadata_config.get("generate_lineage", False)):
        lineage = context.metadata.build_lineage(
            dataset_name=dataset_name,
            source_datasets=source_datasets,
            output_dataset=dataset_name,
            transformation_name=f"build_{mart_key}",
            module_name=MODULE_NAME,
        )
        context.metadata.write_metadata(
            metadata=lineage,
            output_path=f"data/metadata/{metadata_prefix}_lineage.json",
        )


def write_gold_dataset(
    context,
    dataframe: pd.DataFrame,
    mart_key: str,
    mart_config: Dict[str, Any],
    source_datasets: List[str],
    gold_config: Dict[str, Any],
) -> pd.DataFrame:
    """Validate, audit, write, and document one Gold dataset."""
    dataset_name = require_config_value(mart_config, "dataset_name", f"gold.marts.{mart_key}")
    output_path = require_config_value(mart_config, "output_path", f"gold.marts.{mart_key}")
    primary_key = mart_config.get("primary_key", [])

    dataframe = add_gold_audit_columns(context, dataframe.copy(deep=True), gold_config)

    validate_gold_dataset(
        context=context,
        dataframe=dataframe,
        dataset_name=dataset_name,
        primary_key=primary_key,
        gold_config=gold_config,
    )

    written_path = context.storage.write_parquet(dataframe=dataframe, path=output_path, index=False)

    context.logging.log_dataset(
        dataset_name=dataset_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        path=written_path,
    )

    write_gold_metadata(
        context=context,
        dataframe=dataframe,
        mart_key=mart_key,
        dataset_name=dataset_name,
        output_path=str(written_path),
        primary_key=primary_key,
        source_datasets=source_datasets,
        gold_config=gold_config,
    )

    return dataframe


def build_member_360(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Member 360 mart."""
    members = inputs["dim_member"].copy()
    claims = inputs["fact_claim_header"]
    pharmacy = inputs["fact_pharmacy"]
    labs = inputs["fact_laboratory"]
    sdoh = inputs["fact_sdoh"]

    output = members.copy()

    if "member_id" in claims.columns:
        claim_summary = claims.groupby("member_id", as_index=False).agg(
            total_claims=("claim_id", "count"),
            total_allowed_amount=("allowed_amount", "sum"),
            total_paid_amount=("paid_amount", "sum"),
        )
        output = output.merge(claim_summary, on="member_id", how="left")

    if "member_id" in pharmacy.columns:
        pharmacy_summary = pharmacy.groupby("member_id", as_index=False).agg(
            total_pharmacy_claims=("pharmacy_claim_id", "count"),
            total_pharmacy_paid_amount=("paid_amount", "sum"),
        )
        output = output.merge(pharmacy_summary, on="member_id", how="left")

    if "member_id" in labs.columns:
        lab_summary = labs.groupby("member_id", as_index=False).agg(
            total_lab_results=("lab_result_id", "count"),
            abnormal_lab_results=("abnormal_flag", lambda s: int((s.astype(str) != "N").sum())),
        )
        output = output.merge(lab_summary, on="member_id", how="left")

    if "member_id" in sdoh.columns:
        sdoh_summary = sdoh.groupby("member_id", as_index=False).agg(
            sdoh_assessments=("sdoh_id", "count"),
            latest_sdoh_risk_score=("sdoh_risk_score", "max"),
        )
        output = output.merge(sdoh_summary, on="member_id", how="left")

    numeric_defaults = [
        "total_claims",
        "total_allowed_amount",
        "total_paid_amount",
        "total_pharmacy_claims",
        "total_pharmacy_paid_amount",
        "total_lab_results",
        "abnormal_lab_results",
        "sdoh_assessments",
        "latest_sdoh_risk_score",
    ]

    for column in numeric_defaults:
        if column in output.columns:
            output[column] = output[column].fillna(0)

    return output


def build_enrollment_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Enrollment Summary mart."""
    enrollment = inputs["fact_enrollment"].copy()

    group_columns = [column for column in ["line_of_business", "product_code", "plan_id"] if column in enrollment.columns]

    if not group_columns:
        group_columns = ["record_status"] if "record_status" in enrollment.columns else []

    summary = enrollment.groupby(group_columns, dropna=False).size().reset_index(name="enrollment_count") if group_columns else pd.DataFrame(
        {"enrollment_count": [len(enrollment)]}
    )

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_utilization_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Utilization Summary mart."""
    encounters = inputs["fact_encounter"].copy()

    group_columns = [column for column in ["encounter_year", "encounter_month", "encounter_type"] if column in encounters.columns]

    summary = encounters.groupby(group_columns, dropna=False).size().reset_index(name="encounter_count")

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_cost_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Cost Summary mart."""
    claims = inputs["fact_claim_header"].copy()

    group_columns = [column for column in ["claim_type", "encounter_type"] if column in claims.columns]

    summary = claims.groupby(group_columns, dropna=False).agg(
        claim_count=("claim_id", "count"),
        total_allowed_amount=("allowed_amount", "sum"),
        total_paid_amount=("paid_amount", "sum"),
        total_member_liability_amount=("member_liability_amount", "sum"),
    ).reset_index()

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_pmpm_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold PMPM Summary mart."""
    member_month = inputs["fact_member_month"].copy()
    claims = inputs["fact_claim_header"].copy()

    member_month_count = len(member_month)

    total_paid_amount = float(claims["paid_amount"].sum()) if "paid_amount" in claims.columns else 0.0
    total_allowed_amount = float(claims["allowed_amount"].sum()) if "allowed_amount" in claims.columns else 0.0

    pmpm_paid = total_paid_amount / member_month_count if member_month_count else 0.0
    pmpm_allowed = total_allowed_amount / member_month_count if member_month_count else 0.0

    return pd.DataFrame(
        {
            "summary_key": [1],
            "member_month_count": [member_month_count],
            "total_paid_amount": [round(total_paid_amount, 2)],
            "total_allowed_amount": [round(total_allowed_amount, 2)],
            "pmpm_paid_amount": [round(pmpm_paid, 2)],
            "pmpm_allowed_amount": [round(pmpm_allowed, 2)],
        }
    )


def build_provider_performance(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Provider Performance mart."""
    providers = inputs["dim_provider"].copy()
    claims = inputs["fact_claim_header"].copy()
    encounters = inputs["fact_encounter"].copy()

    output = providers.copy()

    if "provider_id" in claims.columns:
        claim_summary = claims.groupby("provider_id", as_index=False).agg(
            claim_count=("claim_id", "count"),
            total_paid_amount=("paid_amount", "sum"),
            total_allowed_amount=("allowed_amount", "sum"),
        )
        output = output.merge(claim_summary, on="provider_id", how="left")

    if "provider_id" in encounters.columns:
        encounter_summary = encounters.groupby("provider_id", as_index=False).agg(
            encounter_count=("encounter_id", "count"),
        )
        output = output.merge(encounter_summary, on="provider_id", how="left")

    for column in ["claim_count", "total_paid_amount", "total_allowed_amount", "encounter_count"]:
        if column in output.columns:
            output[column] = output[column].fillna(0)

    return output


def build_organization_performance(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Organization Performance mart."""
    organizations = inputs["dim_organization"].copy()
    organizations["organization_metric_placeholder"] = 1
    return organizations


def build_facility_performance(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Facility Performance mart."""
    facilities = inputs["dim_facility"].copy()
    claims = inputs["fact_claim_header"].copy()
    encounters = inputs["fact_encounter"].copy()
    labs = inputs["fact_laboratory"].copy()

    output = facilities.copy()

    if "facility_id" in claims.columns:
        claim_summary = claims.groupby("facility_id", as_index=False).agg(
            claim_count=("claim_id", "count"),
            total_paid_amount=("paid_amount", "sum"),
        )
        output = output.merge(claim_summary, on="facility_id", how="left")

    if "facility_id" in encounters.columns:
        encounter_summary = encounters.groupby("facility_id", as_index=False).agg(
            encounter_count=("encounter_id", "count"),
        )
        output = output.merge(encounter_summary, on="facility_id", how="left")

    if "facility_id" in labs.columns:
        lab_summary = labs.groupby("facility_id", as_index=False).agg(
            lab_result_count=("lab_result_id", "count"),
        )
        output = output.merge(lab_summary, on="facility_id", how="left")

    for column in ["claim_count", "total_paid_amount", "encounter_count", "lab_result_count"]:
        if column in output.columns:
            output[column] = output[column].fillna(0)

    return output


def build_clinical_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Clinical Summary mart."""
    claims = inputs["fact_claim_header"].copy()

    group_column = "primary_diagnosis_code"

    summary = claims.groupby(group_column, dropna=False).agg(
        claim_count=("claim_id", "count"),
        total_paid_amount=("paid_amount", "sum"),
    ).reset_index()

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_pharmacy_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Pharmacy Summary mart."""
    pharmacy = inputs["fact_pharmacy"].copy()

    group_columns = [column for column in ["drug_class", "generic_brand"] if column in pharmacy.columns]

    summary = pharmacy.groupby(group_columns, dropna=False).agg(
        pharmacy_claim_count=("pharmacy_claim_id", "count"),
        total_paid_amount=("paid_amount", "sum"),
        total_allowed_amount=("allowed_amount", "sum"),
    ).reset_index()

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_laboratory_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold Laboratory Summary mart."""
    labs = inputs["fact_laboratory"].copy()

    summary = labs.groupby("test_name", dropna=False).agg(
        lab_result_count=("lab_result_id", "count"),
        abnormal_result_count=("abnormal_flag", lambda s: int((s.astype(str) != "N").sum())),
    ).reset_index()

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_sdoh_summary(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build Gold SDOH Summary mart."""
    sdoh = inputs["fact_sdoh"].copy()

    group_column = "sdoh_risk_band" if "sdoh_risk_band" in sdoh.columns else "record_status"

    summary = sdoh.groupby(group_column, dropna=False).agg(
        sdoh_record_count=("sdoh_id", "count"),
        average_sdoh_risk_score=("sdoh_risk_score", "mean"),
    ).reset_index()

    summary["summary_key"] = range(1, len(summary) + 1)

    return summary


def build_gold_mart(mart_key: str, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dispatch Gold mart build by approved mart key."""
    builders = {
        "member_360": build_member_360,
        "enrollment_summary": build_enrollment_summary,
        "utilization_summary": build_utilization_summary,
        "cost_summary": build_cost_summary,
        "pmpm_summary": build_pmpm_summary,
        "provider_performance": build_provider_performance,
        "organization_performance": build_organization_performance,
        "facility_performance": build_facility_performance,
        "clinical_summary": build_clinical_summary,
        "pharmacy_summary": build_pharmacy_summary,
        "laboratory_summary": build_laboratory_summary,
        "sdoh_summary": build_sdoh_summary,
    }

    if mart_key not in builders:
        raise PipelineError(f"Unsupported Gold mart key: {mart_key}")

    return builders[mart_key](inputs)


def build_gold_layer(context) -> Dict[str, pd.DataFrame]:
    """Build all enabled Gold marts."""
    logger = context.get_logger(MODULE_NAME)

    gold_config = load_gold_config(context)
    configuration_config = require_config_section(gold_config, "configuration", "gold.yaml")

    if not bool(configuration_config.get("enabled", True)):
        logger.warning("Gold Layer is disabled in config/gold/gold.yaml.")
        return {}

    inputs = load_gold_inputs(context, gold_config)
    marts_config = require_config_section(gold_config, "marts", "gold.yaml")

    outputs: Dict[str, pd.DataFrame] = {}

    for mart_key, mart_config in marts_config.items():
        if not bool(mart_config.get("enabled", True)):
            continue

        logger.info("START Gold mart: %s", mart_key)

        mart_dataframe = build_gold_mart(mart_key=mart_key, inputs=inputs)

        outputs[mart_key] = write_gold_dataset(
            context=context,
            dataframe=mart_dataframe,
            mart_key=mart_key,
            mart_config=mart_config,
            source_datasets=[f"silver.{name}" for name in inputs.keys()],
            gold_config=gold_config,
        )

        logger.info("COMPLETE Gold mart: %s | Rows: %s", mart_key, len(outputs[mart_key]))

    return outputs


def main() -> None:
    """Main entry point for Gold Layer build."""
    context = create_pipeline_context()
    logger = context.get_logger(MODULE_NAME)

    try:
        context.logging.start_step(STEP_NAME)

        outputs = build_gold_layer(context)

        context.logging.end_step(STEP_NAME)

        logger.info("MedFabric Gold Layer completed successfully. Datasets built: %s", len(outputs))

        print("MedFabric Gold Layer completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Gold Layer build failed.")
        logger.exception("Gold Layer build failed.")
        raise PipelineError("Gold Layer build failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()