# MedFabric Semantic Layer Architecture

**Project:** MedFabric – Enterprise Healthcare Data & AI Platform  
**Layer:** Layer 1G – Semantic Layer  
**Document:** Semantic Layer Architecture  
**Version:** 1.0  
**Status:** Draft

---

# 1. Purpose

The Semantic Layer is the final sublayer of the MedFabric Data Platform. Its purpose is to provide a business-friendly data model by defining governed business metrics, subject areas, measures, dimensions, KPI definitions, and calculation rules over approved Gold and Feature Store assets.

The Semantic Layer does not create new analytical data. It provides consistent business meaning for existing analytical assets.

---

# 2. Roadmap Alignment

Approved Layer 1G scope:

- Business Metrics
- Subject Areas
- Measures
- Dimensions
- KPI Definitions
- Calculation Rules

No additional capabilities are introduced in this layer.

---

# 3. Layer Position

```text
Reference Data
↓
Raw
↓
Bronze
↓
Silver
↓
Gold
↓
Feature Store
↓
Semantic Layer
```

---

# 4. Objectives

- Standardize business definitions.
- Eliminate inconsistent metric calculations.
- Provide reusable semantic artifacts.
- Support dashboards, notebooks, SQL and reporting.
- Maintain complete metadata and lineage.

---

# 5. Inputs

Primary inputs:

- Gold analytical marts
- Approved Feature Store feature groups
- Semantic Layer configuration

The Semantic Layer shall not read Raw, Bronze or Silver datasets directly.

---

# 6. Outputs

```text
data/semantic/
    business_metrics.parquet
    subject_areas.parquet
    measures.parquet
    dimensions.parquet
    kpi_definitions.parquet
    calculation_rules.parquet
    semantic_catalog.parquet
```

Metadata:

```text
data/metadata/
    semantic_*_dataset_metadata.json
    semantic_*_column_metadata.csv
    semantic_*_statistics.json
    semantic_*_lineage.json
```

---

# 7. Repository Structure

```text
config/
    semantic_layer/
        semantic_layer.yaml

src/
    semantic_layer/
        __init__.py
        build_semantic_layer.py

docs/
    Semantic_Layer_Architecture.md

data/
    semantic/
```

---

# 8. Components

## Business Metrics

Business-facing metrics used across MedFabric.

Examples:

- Member Count
- Active Member Count
- Claim Count
- Total Paid Amount
- Total Allowed Amount
- PMPM
- Average Claim Cost
- Pharmacy Paid Amount

Each metric shall contain:

- Name
- Business definition
- Subject area
- Source dataset
- Source columns
- Calculation rule
- Owner
- Certification status

---

## Subject Areas

Initial subject areas:

- Member
- Enrollment
- Claims
- Cost
- Utilization
- Provider
- Organization
- Facility
- Pharmacy
- Laboratory
- SDOH
- Risk

---

## Measures

Reusable numeric aggregations.

Examples:

- Count Members
- Count Claims
- Sum Paid Amount
- Sum Allowed Amount
- Sum Member Months
- Count Providers

Each measure shall define:

- Aggregation
- Source dataset
- Source column
- Data type

---

## Dimensions

Business slicing attributes.

Examples:

- Member
- Provider
- Organization
- Facility
- Date
- Geography
- Line of Business
- Product
- Plan
- Gender
- Age Band
- Race
- Ethnicity

---

## KPI Definitions

KPIs define business performance indicators.

Examples:

- PMPM
- Admission Rate
- ER Utilization Rate
- High Cost Member Rate
- Abnormal Lab Rate

Each KPI contains:

- Name
- Business definition
- Target
- Threshold
- Calculation rule
- Owner

---

## Calculation Rules

Examples:

```text
PMPM = Total Paid Amount / Member Months

Average Claim Cost =
Total Paid Amount / Claim Count

Abnormal Lab Rate =
Abnormal Lab Results / Total Lab Results
```

Each rule contains:

- Formula
- Source measures
- Source datasets
- Null handling
- Divide-by-zero handling

---

# 9. Validation

Validation shall verify:

- Required attributes
- Valid source datasets
- Valid source columns
- Non-empty outputs
- Valid calculation rules

---

# 10. Metadata

Generate:

- Dataset metadata
- Column metadata
- Statistics
- Lineage

---

# 11. Logging

Use the MedFabric centralized logging framework.

---

# 12. Execution Flow

1. Load configuration.
2. Load approved datasets.
3. Build subject areas.
4. Build dimensions.
5. Build measures.
6. Build business metrics.
7. Build KPI definitions.
8. Build calculation rules.
9. Build semantic catalog.
10. Validate outputs.
11. Generate metadata.
12. Generate lineage.
13. Write outputs.

---

# 13. Completion Criteria

Layer 1G is complete when:

- Configuration exists.
- Builder exists.
- Semantic outputs are generated.
- Metadata is generated.
- Validation passes.
- Documentation is complete.
- Layer executes successfully.

---

# 14. Layer Boundary

This layer does not include:

- Dashboards
- Predictive models
- Registries
- Care gaps
- Cloud deployment
- APIs

Those belong to later roadmap layers.

---

**END OF DOCUMENT**
