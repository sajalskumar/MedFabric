MedFabric/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── app.yaml
│   ├── paths.yaml
│   ├── pipeline.yaml
│   ├── logging.yaml
│   ├── data_quality.yaml
│   ├── governance.yaml
│   ├── data_generation.yaml
│   ├── bronze.yaml
│   ├── silver.yaml
│   ├── gold.yaml
│   ├── feature_store.yaml
│   ├── modeling.yaml
│   ├── scoring.yaml
│   └── models/
│       ├── high_cost.yaml
│       ├── readmission.yaml
│       ├── er_utilization.yaml
│       ├── rising_risk.yaml
│       ├── chronic_progression.yaml
│       ├── medication_non_adherence.yaml
│       ├── care_gap_closure.yaml
│       └── avoidable_admission.yaml
│
├── docs/
│   ├── 00_MedFabric_Enterprise_Blueprint.md
│   ├── 01_MedFabric_Final_Folder_Structure.md
│   ├── 02_Coding_Standards.md
│   ├── 03_Architecture.md
│   ├── 04_Data_Model.md
│   ├── 05_Governance_Strategy.md
│   ├── 06_Logging_and_Observability.md
│   ├── 07_Modeling_Standards.md
│   ├── 08_Cloud_Migration_Strategy.md
│   └── 09_Roadmap.md
│
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   ├── feature_store/
│   ├── modeling/
│   ├── scoring/
│   ├── metadata/
│   ├── quality/
│   └── audit/
│
├── logs/
│   ├── pipeline/
│   ├── modules/
│   ├── errors/
│   └── audit/
│
├── models/
│   ├── high_cost/
│   ├── readmission/
│   ├── er_utilization/
│   ├── rising_risk/
│   ├── chronic_progression/
│   ├── medication_non_adherence/
│   ├── care_gap_closure/
│   └── avoidable_admission/
│
├── notebooks/
│   ├── 00_platform_health_check.ipynb
│   ├── 01_data_quality_review.ipynb
│   ├── 02_member_360_analysis.ipynb
│   ├── 03_population_health_analysis.ipynb
│   ├── 04_feature_store_analysis.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_executive_dashboard.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── yaml_loader.py
│   │   ├── path_manager.py
│   │   ├── storage_manager.py
│   │   ├── dataframe_manager.py
│   │   ├── validation_manager.py
│   │   ├── logging_manager.py
│   │   ├── metadata_manager.py
│   │   ├── exception_manager.py
│   │   └── pipeline_context.py
│   │
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── generate_members.py
│   │   ├── generate_providers.py
│   │   ├── generate_enrollment.py
│   │   ├── generate_claims.py
│   │   ├── generate_pharmacy.py
│   │   ├── generate_labs.py
│   │   ├── generate_sdoh.py
│   │   ├── generate_clinical_terminology.py
│   │   └── run_data_generation.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── ingest_raw_to_bronze.py
│   │
│   ├── silver/
│   │   ├── __init__.py
│   │   ├── dimensions/
│   │   │   ├── build_dim_member.py
│   │   │   ├── build_dim_provider.py
│   │   │   ├── build_dim_date.py
│   │   │   └── build_dim_clinical_terminology.py
│   │   ├── facts/
│   │   │   ├── build_fact_claims.py
│   │   │   ├── build_fact_enrollment.py
│   │   │   ├── build_fact_member_month.py
│   │   │   ├── build_fact_pharmacy.py
│   │   │   ├── build_fact_labs.py
│   │   │   └── build_fact_sdoh.py
│   │   └── build_silver_layer.py
│   │
│   ├── gold/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── build_member_360.py
│   │   │   ├── build_pmpm_analytics.py
│   │   │   ├── build_utilization_analytics.py
│   │   │   └── build_provider_performance.py
│   │   ├── registries/
│   │   │   ├── build_diabetes_registry.py
│   │   │   ├── build_hypertension_registry.py
│   │   │   ├── build_copd_registry.py
│   │   │   ├── build_chf_registry.py
│   │   │   ├── build_ckd_registry.py
│   │   │   └── build_care_gap_registry.py
│   │   ├── clinical/
│   │   │   ├── build_lab_analytics.py
│   │   │   ├── build_pharmacy_analytics.py
│   │   │   └── build_medication_adherence.py
│   │   ├── attribution/
│   │   │   ├── build_provider_attribution.py
│   │   │   ├── build_pcp_attribution.py
│   │   │   └── build_attribution_quality_metrics.py
│   │   ├── population_management/
│   │   │   ├── build_population_segmentation.py
│   │   │   ├── build_cohort_management.py
│   │   │   ├── build_registry_performance.py
│   │   │   └── build_gap_closure_tracking.py
│   │   └── build_gold_layer.py
│   │
│   ├── feature_store/
│   │   ├── __init__.py
│   │   ├── build_member_feature_store.py
│   │   ├── build_model_feature_bases.py
│   │   ├── leakage_validator.py
│   │   └── feature_store_registry.py
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── common/
│   │   │   ├── model_trainer.py
│   │   │   ├── model_scorer.py
│   │   │   ├── model_evaluator.py
│   │   │   ├── model_registry.py
│   │   │   └── model_explainability.py
│   │   ├── high_cost/
│   │   ├── readmission/
│   │   ├── er_utilization/
│   │   ├── rising_risk/
│   │   ├── chronic_progression/
│   │   ├── medication_non_adherence/
│   │   ├── care_gap_closure/
│   │   └── avoidable_admission/
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── score_high_cost.py
│   │   ├── score_readmission.py
│   │   ├── score_er_utilization.py
│   │   ├── score_rising_risk.py
│   │   └── build_scoring_layer.py
│   │
│   ├── governance/
│   │   ├── __init__.py
│   │   ├── build_dataset_inventory.py
│   │   ├── build_column_dictionary.py
│   │   ├── build_metadata_catalog.py
│   │   ├── build_data_quality_scorecard.py
│   │   ├── build_config_validation.py
│   │   ├── build_pipeline_run_history.py
│   │   ├── build_model_registry.py
│   │   └── build_governance_layer.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── pipeline_monitor.py
│   │   ├── data_drift_monitor.py
│   │   ├── schema_drift_monitor.py
│   │   └── model_monitor.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── run_full_pipeline.py
│       ├── run_foundation_checks.py
│       ├── run_data_platform.py
│       ├── run_analytics_platform.py
│       └── run_governance_platform.py
│
└── tests/
    ├── test_config_loader.py
    ├── test_logging_manager.py
    ├── test_validation_manager.py
    ├── test_storage_manager.py
    ├── test_data_generation.py
    ├── test_bronze_ingestion.py
    ├── test_silver_layer.py
    ├── test_gold_layer.py
    ├── test_feature_store.py
    ├── test_modeling.py
    └── test_governance.py