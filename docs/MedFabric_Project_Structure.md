###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     docs/MedFabric_Project_Structure.md
#
# Purpose:
#     Defines and freezes the MedFabric repository structure.
#
# Status:
#     FROZEN
#
# Last Updated:
#     June 30, 2026
#
# IMPORTANT
#     This document is the single source of truth for the MedFabric repository
#     structure. No new folders, layers, or top-level packages shall be added
#     unless this document is formally updated.
###############################################################################

# MedFabric Project Structure

---

# 1. Purpose

The purpose of this document is to permanently freeze the MedFabric repository
structure before further development.

One of the major lessons learned from previous projects was that changing the
repository structure during development creates unnecessary complexity,
confusion, broken imports, duplicated code, and architectural drift.

From this point forward:

- No new top-level folders will be created.
- No new architectural layers will be introduced.
- Existing folder purposes will not change.
- Any future structural changes must first update this document.

---

# 2. Architectural Principles

The repository structure is designed around the MedFabric architecture.

Each major layer owns its own configuration, source code, outputs, metadata,
and orchestration.

The folder hierarchy mirrors the platform architecture.

```
Foundation Platform

        ↓

Data Platform

        ↓

Analytics Platform

        ↓

Platform Operations
```

The folder structure must remain aligned with this architecture.

---

# 3. Root Repository Structure

```
MedFabric/

├── config/
├── data/
├── docs/
├── logs/
├── models/
├── notebooks/
├── reference/
├── src/
├── temp/
├── tests/
└── README.md
```

---

# 4. Configuration Structure

```
config/

├── app.yaml
├── logging.yaml
├── paths.yaml
├── pipeline.yaml

├── data_generation/
├── bronze/
├── silver/
├── gold/
├── feature_store/
├── governance/
└── models/
```

## Purpose

### app.yaml

Application configuration.

---

### logging.yaml

Logging configuration.

---

### pipeline.yaml

Pipeline execution configuration.

---

### paths.yaml

Platform paths.

---

### config/data_generation/

Configuration used by raw synthetic data generators.

Examples

- members.yaml
- providers.yaml
- organizations.yaml
- facilities.yaml
- enrollment.yaml
- encounters.yaml
- claims.yaml
- pharmacy.yaml
- laboratories.yaml
- sdoh.yaml
- terminology.yaml

---

### config/bronze/

Configuration for Bronze ingestion.

---

### config/silver/

Configuration for Silver transformations.

---

### config/gold/

Configuration for Gold marts.

---

### config/feature_store/

Feature Store configuration.

---

### config/governance/

Governance configuration.

---

### config/models/

Predictive model configuration.

---

# 5. Source Code Structure

```
src/

├── common/
├── data_generation/
├── ingestion/
├── silver/
├── gold/
├── feature_store/
├── governance/
├── modeling/
├── monitoring/
├── pipeline/
└── scoring/
```

---

## src/common/

Enterprise framework shared by every module.

Contains

- Configuration Manager
- Logging Manager
- Storage Manager
- Metadata Manager
- Validation Manager
- Exception Manager
- Pipeline Context
- Common Utilities

---

## src/data_generation/

Raw synthetic healthcare data generation.

Contains

- Reference builders
- Dataset generators
- Objects
- Templates
- Utilities
- Validators
- Raw orchestrator

---

## src/ingestion/

Bronze Layer.

Responsible for

- Raw ingestion
- Schema enforcement
- Audit columns
- Initial validation
- Bronze orchestration

---

## src/silver/

Silver Layer.

Responsible for

- Dimensions
- Facts
- Standardization
- Conformance
- Data cleansing
- Business transformations

---

## src/gold/

Gold Layer.

Responsible for

- Member 360
- Provider Analytics
- Clinical Analytics
- Population Analytics
- Pharmacy Analytics
- Gold Marts

---

## src/feature_store/

Reusable machine learning features.

---

## src/governance/

Governance components.

Contains

- Dataset Inventory
- Data Dictionary
- Metadata Catalog
- Lineage
- Quality Reports

---

## src/modeling/

Predictive models.

Contains

- Training
- Evaluation
- Feature Importance
- Metrics

---

## src/scoring/

Population scoring.

---

## src/pipeline/

Pipeline orchestration.

Responsible for

- Layer orchestration
- Full platform execution

---

## src/monitoring/

Monitoring and observability.

---

# 6. Data Structure

```
data/

├── raw/
├── bronze/
├── silver/
├── gold/
├── feature_store/
├── metadata/
├── audit/
├── quality/
├── modeling/
└── scoring/
```

---

## data/raw/

Synthetic source datasets.

---

## data/bronze/

Bronze datasets.

---

## data/silver/

Silver datasets.

---

## data/gold/

Gold datasets.

---

## data/feature_store/

Reusable feature datasets.

---

## data/metadata/

Dataset metadata.

Examples

- Dataset Metadata
- Column Metadata
- Statistics
- Lineage

---

## data/audit/

Audit outputs.

---

## data/quality/

Data Quality outputs.

---

## data/modeling/

Model outputs.

---

## data/scoring/

Population scoring outputs.

---

# 7. Reference Data Structure

```
reference/

├── demographics/
├── enrollment/
├── facilities/
├── geography/
├── laboratory/
├── pharmacy/
├── providers/
├── sdoh/
└── terminology/
```

Reference data is considered static lookup information.

Reference datasets are generated once and reused across the platform.

---

# 8. Documentation Structure

```
docs/
```

Contains

- Architecture
- Roadmaps
- Standards
- Design Documents
- User Guides
- Developer Guides
- Deployment Guides

---

# 9. Notebook Structure

```
notebooks/
```

Contains

- Exploration
- Validation
- Demonstrations
- Executive Dashboards

---

# 10. Logging Structure

```
logs/

├── audit/
├── errors/
├── modules/
└── pipeline/
```

Purpose

- Module Logs
- Pipeline Logs
- Error Logs
- Audit Logs

---

# 11. Model Structure

```
models/

├── high_cost/
├── readmission/
├── er_utilization/
├── rising_risk/
├── chronic_progression/
├── medication_non_adherence/
├── care_gap_closure/
└── avoidable_admission/
```

Each model folder contains

- Trained Model
- Metrics
- Feature Importance
- Training Summary

---

# 12. Development Ownership

## Layer 0

Uses

```
config/
src/common/
```

---

## Layer 1A

Uses

```
reference/
src/data_generation/reference/
```

---

## Layer 1B

Uses

```
config/data_generation/
src/data_generation/
data/raw/
```

---

## Layer 1C

Uses

```
config/bronze/
src/ingestion/
data/bronze/
```

---

## Layer 1D

Uses

```
config/silver/
src/silver/
data/silver/
```

---

## Layer 1E

Uses

```
config/gold/
src/gold/
data/gold/
```

---

## Layer 1F

Uses

```
config/feature_store/
src/feature_store/
data/feature_store/
```

---

## Layer 2

Uses

```
src/gold/
src/modeling/
src/scoring/
```

---

## Layer 3

Uses

```
src/governance/
src/monitoring/
```

---

# 13. Repository Rules

## Rule 1

Never create another top-level folder.

---

## Rule 2

Never create another architectural package.

Examples of prohibited additions

```
src/data_platform/
src/platform/
config/data_platform/
config/analytics/
```

---

## Rule 3

Every new module must belong to an existing layer.

---

## Rule 4

Configuration mirrors source code.

---

## Rule 5

Data mirrors source code.

---

## Rule 6

One orchestrator per architectural layer.

---

## Rule 7

No folder restructuring during active development.

If restructuring becomes necessary,

1. Update this document.
2. Obtain approval.
3. Refactor the repository.
4. Continue development.

---

# 14. Frozen Repository

The MedFabric repository structure is now considered frozen.

Future development will only add new files inside the existing folders defined in this document.

No structural redesign shall occur without updating this document first.

###############################################################################
# END OF DOCUMENT
###############################################################################