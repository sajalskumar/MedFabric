###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     docs/Gold_Architecture.md
#
# Purpose:
#     Defines the architecture, responsibilities, implementation standards,
#     processing lifecycle, and design principles of the Gold Layer within
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

# Gold Layer Architecture

---

# 1. Purpose

The Gold Layer is the Business Analytics Layer of MedFabric.

Its purpose is to transform standardized enterprise healthcare data from the
Silver Layer into business-ready analytical data marts.

Gold datasets are optimized for reporting, dashboards, population health,
provider analytics, operational intelligence, executive reporting and future
machine learning feature generation.

---

# 2. Objectives

The Gold Layer has the following objectives.

- Build business-ready analytical datasets
- Aggregate enterprise healthcare data
- Calculate business KPIs
- Create summary marts
- Support executive dashboards
- Support self-service analytics
- Support Feature Store development
- Maintain complete lineage back to Silver

---

# 3. Gold Philosophy

Gold follows the principle

> **Summarize + Analyze**

Unlike Silver, Gold is designed for business consumption.

Business calculations, aggregations, KPIs and performance metrics are expected
to be created in this layer.

---

# 4. Responsibilities

The Gold Layer SHALL perform the following tasks.

## Read Silver Data

Consume enterprise dimensions and facts from the Silver Layer.

---

## Aggregate Data

Examples

- Monthly summaries
- Annual summaries
- Provider summaries
- Organization summaries
- Facility summaries

---

## Calculate Business Metrics

Examples

- PMPM
- Utilization Rates
- Average Cost
- Allowed Amount
- Paid Amount
- Risk Indicators
- Population Counts

---

## Create Business Marts

Produce subject-oriented analytical datasets.

---

## Generate Metadata

Generate

- Dataset Metadata
- Column Metadata
- Statistics
- Lineage

---

## Write Gold Datasets

Persist datasets into

```
data/gold/
```

---

## Log Processing

Capture

- Execution
- Validation
- Statistics
- Errors

---

# 5. Gold Does NOT Perform

The following activities are outside the Gold Layer.

## No Machine Learning

Model training belongs to the Modeling Platform.

---

## No Feature Engineering

Feature engineering belongs to the Feature Store.

---

## No Raw Data Processing

Gold never reads Raw datasets.

---

## No Bronze Processing

Gold never reads Bronze datasets.

---

Gold consumes Silver datasets only.

---

# 6. Gold Processing Lifecycle

```
            Silver Dimensions
                     │
                     ▼
              Silver Facts
                     │
                     ▼
          Business Transformations
                     │
                     ▼
          Business Aggregations
                     │
                     ▼
          KPI Calculations
                     │
                     ▼
        Gold Analytical Mart
                     │
                     ▼
        Metadata Generation
                     │
                     ▼
         Lineage Generation
                     │
                     ▼
          Write Gold Dataset
                     │
                     ▼
         Validation Report
                     │
                     ▼
          Pipeline Logging
```

---

# 7. Gold Input

Gold consumes datasets from

```
data/silver/
```

Input datasets include

- Dim Member
- Dim Provider
- Dim Organization
- Dim Facility
- Dim Date
- Dim Clinical Terminology

- Fact Enrollment
- Fact Member Month
- Fact Encounter
- Fact Claim Header
- Fact Claim Line
- Fact Pharmacy
- Fact Laboratory
- Fact SDOH

---

# 8. Gold Output

Gold writes analytical datasets into

```
data/gold/
```

---

# 9. Business Analytical Marts

The following Gold datasets shall be created.

---

## Member 360

Enterprise member-centric analytical view.

Combines enrollment, utilization, pharmacy, laboratory and SDOH information
into a single member profile.

---

## Enrollment Summary

Enrollment metrics.

Examples

- Active Members
- New Enrollments
- Terminations
- Coverage Duration

---

## Utilization Summary

Healthcare utilization metrics.

Examples

- IP Visits
- OP Visits
- ED Visits
- Office Visits
- Telehealth Visits

---

## Cost Summary

Financial summaries.

Examples

- Total Cost
- Paid Amount
- Allowed Amount
- Member Liability
- Cost by Service Category

---

## PMPM Summary

Per Member Per Month analytics.

Examples

- PMPM Cost
- PMPM Utilization
- PMPM Pharmacy Cost
- PMPM Laboratory Cost

---

## Provider Performance

Provider-level analytics.

Examples

- Panel Size
- Average Cost
- Utilization
- Quality Indicators
- Claims Volume

---

## Organization Performance

Organization-level analytics.

Examples

- Member Count
- Cost
- Utilization
- Provider Count

---

## Facility Performance

Facility-level analytics.

Examples

- Encounter Volume
- Admissions
- Average Cost
- Laboratory Volume

---

## Clinical Summary

Clinical utilization summaries.

Examples

- Diagnosis Distribution
- Procedure Distribution
- Chronic Condition Counts

---

## Pharmacy Summary

Medication utilization.

Examples

- Prescription Count
- Generic Rate
- Therapeutic Class Distribution

---

## Laboratory Summary

Laboratory utilization.

Examples

- Test Volume
- Abnormal Results
- Test Categories

---

## SDOH Summary

Population SDOH analytics.

Examples

- Housing Distribution
- Food Insecurity
- Transportation Risk
- Financial Stress

---

# 10. Validation

Every Gold dataset shall pass

## Structural Validation

- Dataset exists
- Required columns
- Primary key

---

## Business Validation

- KPI validation
- Aggregation validation
- Summary validation

---

## Quality Validation

- Duplicate detection
- Null validation
- Value validation

---

# 11. Metadata Outputs

Every Gold dataset generates

- Dataset Metadata
- Column Metadata
- Statistics
- Lineage

Metadata is written to

```
data/metadata/
```

---

# 12. Logging

Gold processing logs include

- Start Time
- End Time
- Processing Duration
- Input Rows
- Output Rows
- Aggregation Statistics
- Validation Results
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
config/gold/
```

Implementation

```
src/gold/
```

Output

```
data/gold/
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

Gold datasets shall be implemented in the following order.

1. Member 360

2. Enrollment Summary

3. Utilization Summary

4. Cost Summary

5. PMPM Summary

6. Provider Performance

7. Organization Performance

8. Facility Performance

9. Clinical Summary

10. Pharmacy Summary

11. Laboratory Summary

12. SDOH Summary

This implementation order is frozen unless the architecture is formally revised.

---

# 15. Gold Infrastructure

Before implementing analytical marts, the following shared infrastructure shall
be completed.

- Gold Configuration
- Gold Orchestrator
- Gold Validation
- Gold Metadata

All Gold dataset builders shall use this shared infrastructure.

---

# 16. Relationship to Other Layers

```
Reference
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

Gold is the Business Analytics Layer.

---

# 17. Development Rules

Rule 1

Gold consumes Silver datasets only.

---

Rule 2

Gold creates business-ready analytical marts.

---

Rule 3

Business KPIs are calculated in Gold.

---

Rule 4

Business aggregations are performed in Gold.

---

Rule 5

Every dataset shall generate metadata.

---

Rule 6

Every dataset shall generate lineage.

---

Rule 7

Every dataset shall pass business validation.

---

Rule 8

All Gold dataset builders shall follow identical implementation patterns.

---

# 18. Completion Criteria

The Gold Layer is considered complete when

- All analytical marts are implemented.
- Business KPIs are validated.
- Aggregations are verified.
- Metadata is generated.
- Lineage is generated.
- Validation succeeds.
- Gold orchestrator executes successfully.
- Outputs are reproducible.
- Documentation is complete.

###############################################################################
# END OF DOCUMENT
###############################################################################