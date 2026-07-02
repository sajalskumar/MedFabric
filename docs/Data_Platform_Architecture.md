###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     docs/Bronze_Architecture.md
#
# Purpose:
#     Defines the architecture, design principles, implementation standards,
#     processing lifecycle, and responsibilities of the Bronze Layer within
#     the MedFabric Enterprise Healthcare Data Platform.
#
# Status:
#     FROZEN
#
# Version:
#     1.0
#
# Last Updated:
#     June 30, 2026
#
# Author:
#     MedFabric Architecture
#
###############################################################################

# Bronze Layer Architecture

---

# 1. Purpose

The Bronze Layer is the first permanent storage layer of the MedFabric Data
Platform.

Its purpose is to ingest Raw datasets into the Medallion Architecture while
preserving every source record exactly as it was generated.

Bronze serves as the immutable system-of-record for all downstream
transformations.

The Bronze Layer does **not** perform business transformations.

---

# 2. Objectives

The Bronze Layer has the following objectives.

- Preserve Raw datasets
- Create immutable copies
- Capture audit information
- Capture ingestion metadata
- Capture execution lineage
- Validate dataset structure
- Support replayability
- Support traceability

---

# 3. Bronze Philosophy

Bronze follows the principle:

> **Copy + Audit**

Every Raw dataset is copied into Bronze without changing business content.

Only technical metadata is added.

No business logic is executed inside Bronze.

---

# 4. Responsibilities

The Bronze Layer SHALL perform the following tasks.

## Read Raw Data

Read datasets produced by the Raw Data Generation layer.

---

## Validate Input

Perform structural validation.

Examples

- Dataset exists
- Dataset not empty
- Required columns exist
- Primary key exists
- Schema is valid

---

## Preserve Source Data

Bronze preserves

- Column names
- Column order
- Data types
- Business values
- Record counts

---

## Add Audit Columns

Append standardized audit columns to every Bronze dataset.

---

## Generate Metadata

Generate

- Dataset metadata
- Column metadata
- Statistics
- Lineage

---

## Write Bronze Dataset

Persist Bronze dataset to

```
data/bronze/
```

---

## Log Processing

Capture

- Execution time
- Row counts
- Errors
- Validation results

---

# 5. Bronze Does NOT Perform

The following operations are prohibited inside Bronze.

## No Business Rules

Examples

- Clinical rules
- Payer rules
- Coverage rules

---

## No Standardization

Examples

- Rename columns
- Normalize values
- Convert business codes

---

## No Joins

Bronze datasets remain independent.

---

## No Derived Columns

No calculated healthcare metrics.

---

## No Aggregations

No summaries.

---

## No Deduplication

Duplicate records remain unchanged.

---

## No Conformance

Conformed dimensions belong to Silver.

---

## No Analytics

Analytics belong to Gold.

---

# 6. Bronze Processing Lifecycle

```
                Raw Dataset
                     │
                     ▼
            Input Validation
                     │
                     ▼
         Copy Source Dataset
                     │
                     ▼
         Add Audit Columns
                     │
                     ▼
       Generate Metadata
                     │
                     ▼
       Generate Lineage
                     │
                     ▼
      Write Bronze Dataset
                     │
                     ▼
      Validation Report
                     │
                     ▼
       Pipeline Logging
```

---

# 7. Bronze Input

Bronze consumes datasets from

```
data/raw/
```

Datasets include

- Members
- Organizations
- Providers
- Facilities
- Enrollment
- Member Months
- Encounters
- Claim Headers
- Claim Lines
- Pharmacy
- Laboratory
- SDOH

---

# 8. Bronze Output

Bronze writes datasets to

```
data/bronze/
```

Each dataset retains its original business columns while adding standardized
audit columns.

---

# 9. Standard Audit Columns

Every Bronze dataset SHALL contain the following columns.

| Column | Description |
|----------|-------------|
| bronze_run_id | Pipeline execution identifier |
| bronze_load_timestamp | Timestamp of Bronze ingestion |
| bronze_source_dataset | Source Raw dataset |
| bronze_pipeline_version | Platform version |
| bronze_record_hash | SHA256 hash of complete source record |
| bronze_record_status | Active record indicator |

These columns are mandatory for every Bronze dataset.

---

# 10. Validation

Every Bronze dataset shall pass the following validation checks.

## Dataset Validation

- Dataset exists
- Dataset not empty

---

## Schema Validation

- Required columns
- Primary key
- Duplicate keys
- Column count

---

## Audit Validation

Verify

- Audit columns exist
- Run ID populated
- Timestamp populated
- Record hash populated

---

## Metadata Validation

Verify

- Metadata generated
- Statistics generated
- Lineage generated

---

# 11. Metadata Outputs

Every Bronze dataset generates

Dataset Metadata

```
data/metadata/
```

Column Metadata

```
data/metadata/
```

Statistics

```
data/metadata/
```

Lineage

```
data/metadata/
```

---

# 12. Logging

Bronze processing logs include

- Start time
- End time
- Dataset
- Input rows
- Output rows
- Duration
- Validation results
- Errors

Logs are written to

```
logs/modules/
```

and

```
logs/pipeline/
```

---

# 13. Folder Structure

Configuration

```
config/bronze/
```

Implementation

```
src/ingestion/
```

Output

```
data/bronze/
```

Metadata

```
data/metadata/
```

Logs

```
logs/
```

---

# 14. Dataset Build Order

Bronze datasets shall be implemented in the following order.

1. whay 

This order shall remain unchanged unless the architecture is formally revised.

---

# 15. Bronze Infrastructure

Before dataset builders are implemented, the following shared infrastructure
shall be completed.

- Bronze Configuration
- Bronze Orchestrator
- Bronze Validation
- Bronze Metadata
- Bronze Logging

All Bronze dataset builders must use this shared infrastructure.

---

# 16. Relationship to Other Layers

```
Reference Data
        │
        ▼
Raw Data Generation
        │
        ▼
Bronze
        │
        ▼
Silver
        │
        ▼
Gold
        │
        ▼
Feature Store
        │
        ▼
Semantic Layer
```

Bronze is responsible only for ingestion and auditing.

Business transformations begin in the Silver Layer.

---

# 17. Development Rules

Rule 1

Bronze is immutable.

---

Rule 2

Never modify business values.

---

Rule 3

Never perform joins.

---

Rule 4

Never perform aggregations.

---

Rule 5

Never derive business columns.

---

Rule 6

Only technical audit columns may be added.

---

Rule 7

All datasets must generate metadata.

---

Rule 8

All datasets must generate lineage.

---

Rule 9

All datasets must pass validation before being written.

---

Rule 10

All Bronze dataset builders must follow identical implementation patterns to
ensure consistency across the platform.

---

# 18. Completion Criteria

The Bronze Layer is considered complete when:

- All Raw datasets have Bronze equivalents.
- All Bronze datasets contain standardized audit columns.
- All datasets generate metadata.
- All datasets generate lineage.
- All datasets pass validation.
- The Bronze orchestrator executes successfully.
- All outputs are reproducible.
- Documentation is complete.

###############################################################################
# END OF DOCUMENT
###############################################################################