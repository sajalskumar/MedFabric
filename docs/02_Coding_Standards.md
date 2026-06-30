02_Coding_Standards.md

MedFabric Coding Standards

Version: 1.0
Status: LOCKED

⸻

Purpose

This document defines the mandatory coding standards for the MedFabric platform.

Every Python module, configuration file, and pipeline component shall follow these standards without exception.

⸻

1. Self-Documenting Code

Every production Python file shall be understandable without reading any other file.

A developer should be able to open a single file and immediately understand:

* What it does
* Why it exists
* What it reads
* What it writes
* Which configuration it uses
* Which modules it depends on
* How to execute it
* What outputs it creates

⸻

2. Mandatory File Header

Every Python file shall begin with a standardized documentation header containing:

* Module name
* Purpose
* Business description
* Inputs
* Outputs
* Configuration files
* Dependencies
* Processing steps
* Error handling notes
* Example execution command

⸻

3. Required Documentation Sections

Every production module shall include the following sections:

Module Purpose

Explain the business purpose of the module.

Business Context

Explain where the module fits within the MedFabric architecture.

Inputs

List all expected datasets, configuration files, and parameters.

Outputs

List every generated dataset, file, or artifact.

Configuration

List all YAML files used by the module.

Dependencies

List required upstream datasets and Python modules.

Processing Overview

Describe the major processing steps.

Validation

Describe validation and quality checks performed.

Logging

Describe log generation and run tracking.

Exceptions

Describe expected error handling behavior.

Example Run Command

Provide the exact command required to execute the module.

⸻

4. Function Documentation

Every public function shall include:

* Purpose
* Parameters
* Returns
* Raises
* Notes

Docstrings shall follow a consistent format throughout the project.

⸻

5. Main Section

Every executable module shall contain a clearly documented main entry point.

The main section shall:

* Load configuration
* Initialize logging
* Validate inputs
* Execute processing
* Handle exceptions
* Log completion
* Return an appropriate exit code

The execution flow should be easy to follow from top to bottom.

⸻

6. Configuration

No configurable values shall be hardcoded.

Examples include:

* Paths
* Thresholds
* File names
* Model parameters
* Feature lists
* Business rules
* Logging settings

All configuration shall be externalized to YAML.

⸻

7. Logging

Every module shall:

* Create a logger
* Record pipeline run ID
* Log major processing steps
* Record execution duration
* Log dataset sizes
* Record warnings
* Record errors
* Log successful completion

⸻

8. Validation

Every module shall validate:

* Required input files
* Required columns
* Duplicate keys
* Null values
* Data types
* Business rules

Validation failures shall produce clear, actionable messages.

⸻

9. Error Handling

Unhandled exceptions are not permitted.

Modules shall:

* Catch expected exceptions
* Log meaningful error messages
* Preserve stack traces for debugging
* Exit gracefully when possible

⸻

10. Reusability

Duplicate logic is prohibited.

Common functionality must reside in shared platform modules.

Examples include:

* Reading files
* Writing files
* Validation
* Logging
* Metadata generation
* Storage operations
* Exception handling

⸻

11. Naming Standards

Names shall be descriptive and consistent.

Examples:

* build_member_dimension.py
* build_provider_attribution.py
* train_readmission_model.py
* score_er_utilization.py

Avoid abbreviations unless they are industry standard.

⸻

12. Comments

Comments should explain why a decision was made, not simply restate the code.

Business logic, healthcare rules, and non-obvious transformations should always be explained.

⸻

13. Execution

Every executable module shall include a documented run command.

Example:

python -m src.silver.dimensions.build_dim_member

This command should appear in the module header and remain valid throughout the project.

⸻

14. Inputs and Outputs

Each module shall explicitly document:

Inputs

* Source datasets
* Configuration files
* Runtime parameters

Outputs

* Generated datasets
* Metadata
* Logs
* Model artifacts (if applicable)

⸻

15. Production Quality

All code committed to MedFabric should be considered production quality.

Temporary code, debugging statements, commented-out logic, and experimental implementations should not remain in the main codebase.

⸻

Locked Decision

These coding standards are mandatory for all current and future MedFabric development.

Exceptions require explicit approval.

⸻

