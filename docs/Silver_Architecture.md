###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     docs/Silver_Architecture.md
#
# Purpose:
#     Defines the architecture, responsibilities, implementation standards,
#     processing lifecycle, and design principles of the Silver Layer within
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
###############################################################################

# Silver Layer Architecture

---

# 1. Purpose

The Silver Layer is the Enterprise Integration Layer of MedFabric.

Its purpose is to transform the Bronze datasets into standardized,
conformed, analytics-ready healthcare data assets.

Silver is where business transformations begin.

Unlike Bronze, Silver is allowed to transform data while maintaining
complete lineage back to the original Raw records.

---

# 2. Objectives

The Silver Layer has the following objectives.

- Standardize enterprise healthcare data
- Conform business entities
- Build enterprise dimensions
- Build enterprise facts
- Improve data quality
- Apply business rules
- Maintain lineage
- Support Gold analytics

---

# 3. Silver Philosophy

Silver follows the principle

> **Conform + Standardize**

Silver transforms data into a consistent enterprise model while preserving
traceability to Bronze.

Silver is the first layer where business logic is allowed.

---

# 4. Responsibilities

The Silver Layer SHALL perform the following tasks.

## Read Bronze Data

Read Bronze datasets produced by the Bronze Layer.

---

## Standardize Data

Examples

- Standardize text
- Normalize date formats
- Normalize null handling
- Standardize code values
- Standardize identifiers

---

## Apply Business Rules

Examples

- Enrollment rules
- Provider rules
- Clinical rules
- Member rules
- Healthcare business rules

---

## Build Enterprise Dimensions

Create conformed dimensions used across the platform.

---

## Build Enterprise Facts

Create standardized fact tables.

---

## Improve Data Quality

Examples

- Remove duplicates
- Resolve invalid values
- Enforce business constraints
- Validate relationships

---

## Generate Metadata

Generate

- Dataset Metadata
- Column Metadata
- Statistics
- Lineage

---

## Write Silver Datasets

Persist standardized datasets into

```
data/silver/
```

---

## Log Processing

Capture

- Execution
- Validation
- Statistics
- Errors

---

# 5. Silver Does NOT Perform

The following activities are outside the Silver Layer.

## No Business Analytics

Examples

- PMPM
- Utilization dashboards
- Cost summaries
- Member 360
- Provider scorecards

These belong to Gold.

---

## No Predictive Modeling

Examples

- Risk models
- Machine learning
- AI models

These belong to the Analytics Platform.

---

## No Executive Reporting

Reporting belongs to Gold.

---

# 6. Silver Processing Lifecycle

```
             Bronze Dataset
                    │
                    ▼
          Technical Validation
                    │
                    ▼
        Data Standardization
                    │
                    ▼
       Business Rule Processing
                    │
                    ▼
     Dimension / Fact Construction
                    │
                    ▼
         Metadata Generation
                    │
                    ▼
         Lineage Generation
                    │
                    ▼
        Write Silver Dataset
                    │
                    ▼
        Validation Report
                    │
                    ▼
         Pipeline Logging
```

---

# 7. Silver Input

Silver consumes datasets from

```
data/bronze/
```

Input datasets include

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

# 8. Silver Output

Silver writes standardized enterprise datasets to

```
data/silver/
```

---

# 9. Enterprise Dimensions

The following dimensions shall be created.

## Dim Member

Enterprise Member Dimension.

Primary business entity representing an individual member.

---

## Dim Provider

Enterprise Provider Dimension.

Represents healthcare professionals.

---

## Dim Organization

Enterprise Organization Dimension.

Represents provider organizations.

---

## Dim Facility

Enterprise Facility Dimension.

Represents hospitals, clinics, laboratories, pharmacies, and healthcare facilities.

---

## Dim Date

Enterprise calendar dimension.

Used by every fact table.

---

## Dim Clinical Terminology

Enterprise terminology dimension.

Includes standardized healthcare code systems.

Examples

- ICD-10
- CPT
- HCPCS
- LOINC
- Revenue Codes
- Place of Service

---

# 10. Enterprise Facts

The following fact tables shall be created.

## Fact Enrollment

Enrollment history.

---

## Fact Member Month

Monthly enrollment activity.

---

## Fact Encounter

Healthcare encounters.

---

## Fact Claim Header

Medical claim header information.

---

## Fact Claim Line

Medical claim line information.

---

## Fact Pharmacy

Pharmacy claims.

---

## Fact Laboratory

Laboratory results.

---

## Fact SDOH

Social Determinants of Health observations.

---

# 11. Validation

Every Silver dataset shall pass

## Structural Validation

- Dataset exists
- Required columns
- Primary key
- Data types

---

## Business Validation

- Business rules
- Relationship validation
- Referential integrity

---

## Quality Validation

- Duplicate detection
- Null validation
- Domain validation
- Value validation

---

# 12. Metadata Outputs

Every Silver dataset generates

Dataset Metadata

Column Metadata

Statistics

Lineage

All metadata is written to

```
data/metadata/
```

---

# 13. Logging

Silver processing logs include

- Start time
- End time
- Input rows
- Output rows
- Validation results
- Processing duration
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

# 14. Folder Structure

Configuration

```
config/silver/
```

Implementation

```
src/silver/
```

Output

```
data/silver/
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

# 15. Dataset Build Order

Silver datasets shall be implemented in the following order.

## Dimensions

1. Dim Member

2. Dim Provider

3. Dim Organization

4. Dim Facility

5. Dim Date

6. Dim Clinical Terminology

---

## Facts

7. Fact Enrollment

8. Fact Member Month

9. Fact Encounter

10. Fact Claim Header

11. Fact Claim Line

12. Fact Pharmacy

13. Fact Laboratory

14. Fact SDOH

This order is frozen unless the architecture is formally revised.

---

# 16. Silver Infrastructure

Before implementing individual dimensions and facts, the following shared
infrastructure shall be completed.

- Silver Configuration
- Silver Orchestrator
- Silver Validation
- Silver Metadata

All Silver dataset builders must use this shared infrastructure.

---

# 17. Relationship to Other Layers

```
Reference Data
        │
        ▼
Raw
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

Silver is the enterprise integration layer between ingestion and business analytics.

---

# 18. Development Rules

Rule 1

Silver consumes only Bronze datasets.

---

Rule 2

Business transformations begin in Silver.

---

Rule 3

Enterprise dimensions are built in Silver.

---

Rule 4

Enterprise fact tables are built in Silver.

---

Rule 5

Every dataset must pass business validation.

---

Rule 6

Every dataset must generate metadata.

---

Rule 7

Every dataset must generate lineage.

---

Rule 8

All Silver dataset builders must follow identical implementation patterns.

---

# 19. Completion Criteria

The Silver Layer is considered complete when

- All dimensions are built.
- All fact tables are built.
- Business rules are applied consistently.
- Validation succeeds.
- Metadata is generated.
- Lineage is generated.
- The Silver orchestrator executes successfully.
- All outputs are reproducible.
- Documentation is complete.

###############################################################################
# END OF DOCUMENT
###############################################################################