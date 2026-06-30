###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/data_generation/reference/build_reference_data.py
#
# Purpose:
#     Builds static reference datasets used by the MedFabric Synthetic Data
#     Engine.
#
# Business Context:
#     MedFabric generates canonical healthcare objects such as members,
#     providers, encounters, claims, pharmacy events, laboratory results, and
#     SDOH records. Those generators depend on standardized reference data such
#     as race, ethnicity, language, geography, provider specialties, enrollment
#     products, terminology, pharmacy classes, laboratory tests, and SDOH values.
#
# Inputs:
#     config/data_generation/reference_data.yaml
#
# Outputs:
#     reference/demographics/*.parquet
#     reference/geography/*.parquet
#     reference/providers/*.parquet
#     reference/facilities/*.parquet
#     reference/enrollment/*.parquet
#     reference/terminology/*.parquet
#     reference/pharmacy/*.parquet
#     reference/laboratory/*.parquet
#     reference/sdoh/*.parquet
#
# Dependencies:
#     pandas
#     src.common.pipeline_context.create_pipeline_context
#     src.common.exception_manager.PipelineError
#
# Run Command:
#     python -m src.data_generation.reference.build_reference_data
#
# Expected Output:
#     Reference datasets written under reference/
###############################################################################

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.common.exception_manager import PipelineError
from src.common.pipeline_context import create_pipeline_context


def build_demographics_reference() -> Dict[str, pd.DataFrame]:
    """
    Build demographics reference datasets.

    Returns
    -------
    dict
        Dictionary of demographics reference DataFrames.
    """
    return {
        "first_names": pd.DataFrame(
            {
                "first_name": [
                    "James", "Mary", "Robert", "Patricia", "John",
                    "Jennifer", "Michael", "Linda", "William", "Elizabeth",
                ],
                "gender_hint": [
                    "Male", "Female", "Male", "Female", "Male",
                    "Female", "Male", "Female", "Male", "Female",
                ],
            }
        ),
        "last_names": pd.DataFrame(
            {
                "last_name": [
                    "Smith", "Johnson", "Williams", "Brown", "Jones",
                    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                ]
            }
        ),
        "race": pd.DataFrame(
            {
                "race_code": ["WHITE", "BLACK", "ASIAN", "AIAN", "NHPI", "OTHER", "UNKNOWN"],
                "race_description": [
                    "White",
                    "Black or African American",
                    "Asian",
                    "American Indian or Alaska Native",
                    "Native Hawaiian or Other Pacific Islander",
                    "Other Race",
                    "Unknown",
                ],
                "selection_weight": [0.58, 0.13, 0.06, 0.02, 0.01, 0.12, 0.08],
            }
        ),
        "ethnicity": pd.DataFrame(
            {
                "ethnicity_code": ["HISPANIC", "NON_HISPANIC", "UNKNOWN"],
                "ethnicity_description": ["Hispanic or Latino", "Not Hispanic or Latino", "Unknown"],
                "selection_weight": [0.19, 0.76, 0.05],
            }
        ),
        "language": pd.DataFrame(
            {
                "language_code": ["EN", "ES", "ZH", "VI", "AR", "OTHER"],
                "language_description": ["English", "Spanish", "Chinese", "Vietnamese", "Arabic", "Other"],
                "selection_weight": [0.78, 0.14, 0.03, 0.02, 0.01, 0.02],
            }
        ),
    }


def build_geography_reference() -> Dict[str, pd.DataFrame]:
    """
    Build geography reference datasets.

    Returns
    -------
    dict
        Dictionary of geography reference DataFrames.
    """
    states = pd.DataFrame(
        {
            "state_code": ["AZ", "CA", "TX", "FL", "NY"],
            "state_name": ["Arizona", "California", "Texas", "Florida", "New York"],
            "region": ["West", "West", "South", "South", "Northeast"],
        }
    )

    counties = pd.DataFrame(
        {
            "state_code": ["AZ", "AZ", "CA", "TX", "FL", "NY"],
            "county_name": ["Maricopa", "Pima", "Los Angeles", "Harris", "Miami-Dade", "Kings"],
        }
    )

    zip_codes = pd.DataFrame(
        {
            "zip_code": ["85001", "85032", "85701", "90001", "77001", "33101", "11201"],
            "city": ["Phoenix", "Phoenix", "Tucson", "Los Angeles", "Houston", "Miami", "Brooklyn"],
            "state_code": ["AZ", "AZ", "AZ", "CA", "TX", "FL", "NY"],
            "county_name": ["Maricopa", "Maricopa", "Pima", "Los Angeles", "Harris", "Miami-Dade", "Kings"],
            "population_weight": [0.18, 0.14, 0.10, 0.20, 0.16, 0.11, 0.11],
        }
    )

    geography = zip_codes.merge(states, on="state_code", how="left")

    return {
        "states": states,
        "counties": counties,
        "zip_codes": zip_codes,
        "geography": geography,
    }


def build_provider_reference() -> Dict[str, pd.DataFrame]:
    """
    Build provider reference datasets.

    Returns
    -------
    dict
        Dictionary of provider reference DataFrames.
    """
    specialties = pd.DataFrame(
        {
            "specialty_code": ["FM", "IM", "PED", "CARD", "ENDO", "NEPH", "PULM", "ORTHO", "ED", "HOSP"],
            "specialty_description": [
                "Family Medicine",
                "Internal Medicine",
                "Pediatrics",
                "Cardiology",
                "Endocrinology",
                "Nephrology",
                "Pulmonology",
                "Orthopedics",
                "Emergency Medicine",
                "Hospitalist",
            ],
            "is_primary_care": [True, True, True, False, False, False, False, False, False, False],
            "selection_weight": [0.18, 0.16, 0.10, 0.10, 0.07, 0.06, 0.06, 0.08, 0.10, 0.09],
        }
    )

    taxonomy = pd.DataFrame(
        {
            "taxonomy_code": [
                "207Q00000X", "207R00000X", "208000000X", "207RC0000X",
                "207RE0101X", "207RN0300X", "207RP1001X", "207X00000X",
                "207P00000X", "208M00000X",
            ],
            "specialty_code": ["FM", "IM", "PED", "CARD", "ENDO", "NEPH", "PULM", "ORTHO", "ED", "HOSP"],
            "taxonomy_description": specialties["specialty_description"].tolist(),
        }
    )

    organizations = pd.DataFrame(
        {
            "organization_name": [
                "MedFabric Health Group",
                "Valley Integrated Physicians",
                "Desert Care Network",
                "Summit Specialty Partners",
            ],
            "organization_type": ["Provider Group", "IPA", "Health System", "Specialty Group"],
        }
    )

    primary_care_specialties = specialties[specialties["is_primary_care"]].copy()

    return {
        "taxonomy": taxonomy,
        "specialties": specialties,
        "organizations": organizations,
        "primary_care_specialties": primary_care_specialties,
    }


def build_facility_reference() -> Dict[str, pd.DataFrame]:
    """
    Build facility reference datasets.

    Returns
    -------
    dict
        Dictionary of facility reference DataFrames.
    """
    return {
        "facility_reference": pd.DataFrame(
            {
                "facility_type_code": ["HOSP", "CLINIC", "ASC", "LAB", "SNF", "UC"],
                "facility_type_description": [
                    "Hospital",
                    "Clinic",
                    "Ambulatory Surgical Center",
                    "Laboratory",
                    "Skilled Nursing Facility",
                    "Urgent Care",
                ],
                "selection_weight": [0.15, 0.35, 0.10, 0.10, 0.10, 0.20],
            }
        )
    }


def build_enrollment_reference() -> Dict[str, pd.DataFrame]:
    """
    Build enrollment reference datasets.

    Returns
    -------
    dict
        Dictionary of enrollment reference DataFrames.
    """
    return {
        "payer": pd.DataFrame(
            {
                "payer_code": ["MFHP"],
                "payer_name": ["MedFabric Health Plan"],
            }
        ),
        "line_of_business": pd.DataFrame(
            {
                "line_of_business": ["Commercial", "Medicare Advantage", "Medicaid"],
                "selection_weight": [0.55, 0.25, 0.20],
            }
        ),
        "product": pd.DataFrame(
            {
                "product_code": ["HMO", "PPO", "EPO", "MA", "MCD"],
                "product_description": ["HMO", "PPO", "EPO", "Medicare Advantage", "Medicaid Managed Care"],
            }
        ),
        "plan": pd.DataFrame(
            {
                "plan_id": ["PLAN001", "PLAN002", "PLAN003", "PLAN004"],
                "plan_name": ["MedFabric Silver", "MedFabric Gold", "MedFabric MA", "MedFabric Medicaid"],
            }
        ),
        "coverage_type": pd.DataFrame(
            {
                "coverage_type": ["Medical", "Pharmacy", "Medical + Pharmacy"],
                "selection_weight": [0.10, 0.05, 0.85],
            }
        ),
    }


def build_terminology_reference() -> Dict[str, pd.DataFrame]:
    """
    Build terminology reference datasets.

    Returns
    -------
    dict
        Dictionary of terminology reference DataFrames.
    """
    return {
        "icd10": pd.DataFrame(
            {
                "code": ["E11.9", "I10", "J44.9", "I50.9", "N18.9", "E78.5", "F32.9"],
                "description": [
                    "Type 2 diabetes mellitus without complications",
                    "Essential hypertension",
                    "Chronic obstructive pulmonary disease",
                    "Heart failure",
                    "Chronic kidney disease",
                    "Hyperlipidemia",
                    "Major depressive disorder",
                ],
                "code_system": ["ICD-10-CM"] * 7,
                "condition_group": ["Diabetes", "Hypertension", "COPD", "CHF", "CKD", "Hyperlipidemia", "Depression"],
            }
        ),
        "cpt": pd.DataFrame(
            {
                "code": ["99213", "99214", "99285", "83036", "80053", "93000"],
                "description": [
                    "Office visit established patient low complexity",
                    "Office visit established patient moderate complexity",
                    "Emergency department visit high severity",
                    "Hemoglobin A1c",
                    "Comprehensive metabolic panel",
                    "Electrocardiogram",
                ],
                "code_system": ["CPT"] * 6,
            }
        ),
        "hcpcs": pd.DataFrame(
            {
                "code": ["A0428", "G0008", "J1815"],
                "description": ["Ambulance transport", "Influenza vaccine administration", "Insulin injection"],
                "code_system": ["HCPCS"] * 3,
            }
        ),
        "loinc": pd.DataFrame(
            {
                "code": ["4548-4", "2345-7", "2160-0", "718-7"],
                "description": ["Hemoglobin A1c", "Glucose", "Creatinine", "Hemoglobin"],
                "code_system": ["LOINC"] * 4,
            }
        ),
        "revenue_codes": pd.DataFrame(
            {
                "code": ["0450", "0300", "0360", "0120"],
                "description": ["Emergency Room", "Laboratory", "Operating Room", "Room and Board"],
                "code_system": ["Revenue Code"] * 4,
            }
        ),
        "place_of_service": pd.DataFrame(
            {
                "code": ["11", "21", "22", "23", "81"],
                "description": ["Office", "Inpatient Hospital", "Outpatient Hospital", "Emergency Room", "Independent Laboratory"],
                "code_system": ["POS"] * 5,
            }
        ),
        "encounter_types": pd.DataFrame(
            {
                "encounter_type": ["Office Visit", "Emergency", "Inpatient", "Outpatient", "Lab", "Pharmacy"],
                "selection_weight": [0.45, 0.08, 0.07, 0.18, 0.12, 0.10],
            }
        ),
    }


def build_pharmacy_reference() -> Dict[str, pd.DataFrame]:
    """
    Build pharmacy reference datasets.

    Returns
    -------
    dict
        Dictionary of pharmacy reference DataFrames.
    """
    return {
        "rxnorm": pd.DataFrame(
            {
                "rxnorm_code": ["860975", "617314", "197361", "314077", "617320"],
                "drug_name": ["Metformin", "Lisinopril", "Atorvastatin", "Albuterol", "Amlodipine"],
                "drug_class": ["Antidiabetic", "ACE Inhibitor", "Statin", "Bronchodilator", "Calcium Channel Blocker"],
            }
        ),
        "drug_classes": pd.DataFrame(
            {
                "drug_class": ["Antidiabetic", "ACE Inhibitor", "Statin", "Bronchodilator", "Calcium Channel Blocker"],
                "chronic_use_flag": [True, True, True, True, True],
            }
        ),
        "generic_brand": pd.DataFrame(
            {
                "generic_brand": ["Generic", "Brand"],
                "selection_weight": [0.88, 0.12],
            }
        ),
    }


def build_laboratory_reference() -> Dict[str, pd.DataFrame]:
    """
    Build laboratory reference datasets.

    Returns
    -------
    dict
        Dictionary of laboratory reference DataFrames.
    """
    return {
        "laboratory_tests": pd.DataFrame(
            {
                "test_name": ["Hemoglobin A1c", "Glucose", "Creatinine", "Hemoglobin"],
                "loinc_code": ["4548-4", "2345-7", "2160-0", "718-7"],
                "default_unit": ["%", "mg/dL", "mg/dL", "g/dL"],
            }
        ),
        "result_units": pd.DataFrame(
            {
                "unit": ["%", "mg/dL", "g/dL", "mmol/L"],
                "unit_description": ["Percent", "Milligrams per deciliter", "Grams per deciliter", "Millimoles per liter"],
            }
        ),
        "condition_lab_mapping": pd.DataFrame(
            {
                "condition_group": ["Diabetes", "Diabetes", "CKD", "Anemia"],
                "test_name": ["Hemoglobin A1c", "Glucose", "Creatinine", "Hemoglobin"],
            }
        ),
    }


def build_sdoh_reference() -> Dict[str, pd.DataFrame]:
    """
    Build SDOH reference datasets.

    Returns
    -------
    dict
        Dictionary of SDOH reference DataFrames.
    """
    simple_references = {
        "income": ["Low", "Moderate", "High"],
        "education": ["Less than High School", "High School", "Some College", "College Graduate"],
        "employment": ["Employed", "Unemployed", "Retired", "Disabled"],
        "housing": ["Stable", "Unstable", "Homeless Risk"],
        "food_security": ["Secure", "At Risk", "Insecure"],
        "transportation": ["Reliable", "Limited", "Barrier"],
        "social_support": ["Strong", "Moderate", "Limited"],
        "digital_access": ["Broadband", "Mobile Only", "Limited Access"],
        "language_barriers": ["None", "Moderate", "High"],
    }

    return {
        name: pd.DataFrame(
            {
                "category": values,
                "selection_weight": [1 / len(values)] * len(values),
            }
        )
        for name, values in simple_references.items()
    }


def build_all_reference_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build all reference data groups.

    Returns
    -------
    dict
        Nested dictionary of reference group to reference DataFrames.
    """
    return {
        "demographics": build_demographics_reference(),
        "geography": build_geography_reference(),
        "providers": build_provider_reference(),
        "facilities": build_facility_reference(),
        "enrollment": build_enrollment_reference(),
        "terminology": build_terminology_reference(),
        "pharmacy": build_pharmacy_reference(),
        "laboratory": build_laboratory_reference(),
        "sdoh": build_sdoh_reference(),
    }


def write_reference_data(context, reference_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """
    Write reference datasets using paths from reference_data.yaml.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    reference_data:
        Nested dictionary of reference datasets.
    """
    logger = context.get_logger("medfabric.data_generation.reference")
    config = context.configuration.load_yaml("data_generation/reference_data.yaml")

    for group_name, datasets in reference_data.items():
        group_config = config.get(group_name, {})

        for dataset_name, dataframe in datasets.items():
            dataset_config = group_config.get(dataset_name)

            if dataset_config is None:
                logger.warning(
                    "Reference dataset has no config entry and will be skipped: %s.%s",
                    group_name,
                    dataset_name,
                )
                continue

            if not dataset_config.get("generate", True):
                logger.info("Skipping disabled reference dataset: %s.%s", group_name, dataset_name)
                continue

            output_path = dataset_config["output"]

            context.storage.write_parquet(dataframe, output_path)

            context.logging.log_dataset(
                dataset_name=f"{group_name}.{dataset_name}",
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                path=output_path,
            )

            metadata = context.metadata.build_dataset_metadata(
                dataset_name=f"{group_name}.{dataset_name}",
                dataframe=dataframe,
                output_path=output_path,
                layer="reference",
                domain=group_name,
                description=f"Reference dataset for {group_name}.{dataset_name}",
            )

            metadata_output_path = (
                context.paths.get_data_path("metadata")
                / f"reference_{group_name}_{dataset_name}_metadata.json"
            )

            context.metadata.write_metadata(metadata, metadata_output_path)


def validate_reference_data(context, reference_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """
    Validate generated reference datasets.

    Parameters
    ----------
    context:
        Active MedFabric PipelineContext.

    reference_data:
        Nested dictionary of reference datasets.
    """
    for group_name, datasets in reference_data.items():
        for dataset_name, dataframe in datasets.items():
            context.validation.validate_dataset(
                dataframe=dataframe,
                validation_rules={
                    "min_rows": 1,
                    "allow_empty": False,
                },
                dataset_name=f"{group_name}.{dataset_name}",
            )


def main() -> None:
    """
    Main entry point for reference data generation.

    Run Command
    -----------
    python -m src.data_generation.reference.build_reference_data
    """
    context = create_pipeline_context()
    logger = context.get_logger("medfabric.data_generation.reference")

    try:
        context.logging.start_step("Build Reference Data")

        reference_data = build_all_reference_data()

        validate_reference_data(context, reference_data)
        write_reference_data(context, reference_data)

        context.logging.end_step("Build Reference Data")

        print("MedFabric reference data build completed successfully.")

    except Exception as error:
        context.logging.log_exception(error, "Reference data build failed.")
        raise PipelineError("Reference data build failed.") from error

    finally:
        context.logging.close()


if __name__ == "__main__":
    main()