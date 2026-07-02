# MedFabric Pipeline Architecture

**Project:** MedFabric – Enterprise Healthcare Data & AI Platform

**Layer:** Enterprise Pipeline Orchestration

**Version:** Release 2.0

**Status:** Design Complete

---

# 1. Overview

The Pipeline layer is the enterprise orchestration layer of MedFabric.

Its responsibility is **not** to perform business transformations, analytics,
or reporting. Instead, it coordinates execution of every platform layer,
provides centralized execution management, records audit history, and produces
a single entry point for running the complete MedFabric platform.

The Pipeline layer is the highest-level orchestration component in the project
and is responsible for executing all lower platform layers in the correct
dependency order.

---

# 2. Purpose

The Pipeline layer provides a single executable workflow capable of building
the entire MedFabric platform from synthetic source data through executive
reporting.

The Pipeline layer performs:

- Configuration loading
- Layer dependency management
- Runtime initialization
- Sequential orchestration
- Execution monitoring
- Audit logging
- Metadata collection
- Failure handling
- Pipeline health reporting

The Pipeline layer intentionally contains **no business logic**.

---

# 3. Position Within MedFabric

```
                MedFabric Platform

                        │
                        ▼

                Enterprise Pipeline
                (Master Orchestrator)

                        │
        ─────────────────────────────────

        Layer 1
        Data Platform

                        │

        Layer 2
        Analytics Platform

                        │

        Layer 3
        Insights Platform
```

The Pipeline layer coordinates all lower layers but never performs analytical
processing itself.

---

# 4. Architectural Principles

The Pipeline layer follows several enterprise architecture principles.

## Single Responsibility

Each platform layer is responsible only for its own work.

The Pipeline layer only coordinates execution.

---

## Configuration Driven

No execution order is hardcoded.

Execution order is defined entirely inside

```
config/pipeline/medfabric_platform.yaml
```

---

## Modular

Every platform layer remains independently executable.

Examples:

```
python -m src.data_generation.build_data_generation

python -m src.semantic_layer.build_semantic_layer

python -m src.analytics_platform.build_analytics_platform

python -m src.insights.build_insights_platform
```

The master pipeline simply invokes these modules.

---

## Fail Fast

When configured to do so, pipeline execution immediately stops after the first
critical failure.

When fail-fast is disabled, remaining layers continue executing and failures
are reported at the end.

---

## Observable

Every execution step is logged.

Every layer execution produces:

- Status
- Start time
- End time
- Duration
- Row counts
- Output counts
- Error messages
- Execution metadata

---

# 5. Pipeline Responsibilities

The Pipeline layer is responsible for:

- Loading configuration
- Initializing runtime
- Creating run identifiers
- Executing layers
- Recording execution history
- Writing audit logs
- Writing metadata
- Recording execution duration
- Producing platform execution summary
- Returning overall platform status

The Pipeline layer is **not** responsible for:

- Data generation
- Data quality
- Feature engineering
- Modeling
- Analytics
- Reporting
- Dashboard generation

---

# 6. Execution Order

The Pipeline executes layers in dependency order.

```
Data Generation

        │

        ▼

Data Platform

        │

        ▼

Feature Store

        │

        ▼

Modeling

        │

        ▼

Semantic Layer

        │

        ▼

Analytics Platform

        │

        ▼

Insights Platform
```

Each layer must successfully complete before dependent layers begin unless the
configuration explicitly allows execution to continue.

---

# 7. Platform Components

The Pipeline layer consists of the following modules.

```
src/pipeline/

    common/

        runtime.py

        io.py

        validation.py

        metadata.py

        audit.py

    build_medfabric_platform.py
```

---

# 8. Runtime

The runtime module provides:

- Pipeline context
- Run ID generation
- Configuration loading
- Logger creation
- Runtime metadata
- Layer execution state

The runtime object is shared across the Pipeline layer.

---

# 9. IO Module

The IO module is responsible for:

- Reading configuration
- Writing execution summaries
- Writing pipeline metadata
- Writing audit datasets
- Output directory creation

The IO module never performs business processing.

---

# 10. Validation Module

Validation ensures:

- Configuration exists
- Required layers are configured
- Execution order is valid
- Required dependencies exist
- Output folders exist
- Layer execution succeeded

Validation is performed before and after every layer.

---

# 11. Metadata Module

Metadata captures:

- Executed layer
- Layer version
- Start time
- End time
- Runtime
- Status
- Output datasets
- Output rows
- Pipeline version
- Configuration version

Metadata is written for every execution.

---

# 12. Audit Module

Audit captures operational history.

Typical audit information includes:

- Run ID
- User
- Machine
- Layer
- Step
- Status
- Error message
- Duration
- Timestamp

Audit records support troubleshooting and reproducibility.

---

# 13. Master Orchestrator

The master orchestrator is

```
build_medfabric_platform.py
```

Responsibilities:

1. Initialize runtime

2. Load configuration

3. Validate configuration

4. Execute platform layers

5. Record execution status

6. Record metadata

7. Record audit

8. Produce execution summary

9. Return final status

---

# 14. Configuration

Pipeline configuration is stored in

```
config/pipeline/medfabric_platform.yaml
```

Configuration controls:

- Enabled layers
- Execution order
- Fail-fast behavior
- Logging
- Audit
- Metadata
- Validation
- Output locations

No orchestration logic should be hardcoded.

---

# 15. Failure Handling

Every platform layer returns a standardized result.

Possible execution states include:

- SUCCESS
- WARNING
- FAILED
- SKIPPED

Failures are captured without losing audit history.

The Pipeline layer always attempts to produce metadata and audit outputs even
when execution fails.

---

# 16. Logging

Pipeline logging records:

- Layer start
- Layer completion
- Duration
- Errors
- Warnings
- Configuration
- Execution summary

Logs are written to the centralized logging framework.

---

# 17. Outputs

The Pipeline layer produces:

```
data/pipeline/

    metadata/

    audit/

    execution_summary/

    run_history/
```

Typical outputs include:

- Pipeline execution summary
- Layer execution summary
- Audit history
- Metadata catalog
- Validation results

---

# 18. Design Goals

The Pipeline layer is designed to be:

- Simple
- Deterministic
- Configuration-driven
- Observable
- Modular
- Reusable
- Maintainable
- Enterprise-ready

---

# 19. Future Enhancements

Future releases may introduce:

- Parallel execution
- Incremental execution
- Dependency graph scheduling
- Checkpoint restart
- Distributed execution
- Cloud orchestration
- Container support
- Airflow integration
- Azure Data Factory integration
- GitHub Actions integration

The current implementation intentionally remains sequential to maximize
traceability and simplify debugging.

---

# 20. Summary

The Pipeline layer serves as the enterprise orchestration engine for MedFabric.

Rather than implementing business functionality, it coordinates execution of
the platform's architectural layers, ensuring reproducible, auditable, and
configuration-driven execution from synthetic data generation through executive
reporting.

By centralizing orchestration while keeping business logic within each layer,
the Pipeline architecture preserves modularity, simplifies maintenance, and
provides a scalable foundation for future enterprise enhancements.