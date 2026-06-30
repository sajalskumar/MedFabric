# 05_Foundation_Public_Interface.md

# MedFabric Foundation Public Interface

**Project:** MedFabric  
**Product Name:** MedFabric  
**Product Type:** Enterprise Healthcare Data & AI Platform  
**Version:** 1.0  
**Status:** LOCKED  
**Document Type:** Foundation Engineering Standard  

---

# 1. Purpose

This document defines the approved public interface for the MedFabric Foundation layer.

The Foundation layer provides reusable platform services used by all other MedFabric components.

This document exists to prevent duplicated infrastructure logic across the project.

---

# 2. Foundation Components

The approved Foundation components are:

```text
configuration_manager.py
path_manager.py
logging_manager.py
storage_manager.py
validation_manager.py
metadata_manager.py
exception_manager.py
pipeline_context.py
```

---

# 3. Locked Rule

Business modules shall not directly manage infrastructure concerns.

Business modules should avoid direct use of:

```python
yaml
logging
os
shutil
pathlib
```

Instead, business modules shall use the Foundation components defined in this document.

---

# 4. ConfigurationManager

## Purpose

Loads and provides YAML configuration.

## Public Methods

```python
load_yaml(file_name: str, use_cache: bool = True) -> dict

get_section(
    file_name: str,
    section_name: str,
    required: bool = True
) -> dict | None

load_core_configs() -> dict

clear_cache() -> None
```

## Used For

- Loading application configuration
- Loading path configuration
- Loading logging configuration
- Loading pipeline configuration
- Loading layer-specific configuration
- Loading model-specific configuration

## Rule

No business module should import `yaml` directly.

---

# 5. PathManager

## Purpose

Resolves paths and manages directories.

## Public Methods

```python
resolve_path(path_value: str) -> Path

get_path(section: str, key: str) -> Path

ensure_directory(path: Path) -> None

ensure_core_directories() -> None

file_exists(path: Path) -> bool

directory_exists(path: Path) -> bool
```

## Used For

- Resolving local project paths
- Creating required directories
- Checking file existence
- Checking directory existence
- Supporting future cloud path abstraction

## Rule

No business module should hardcode project paths.

---

# 6. LoggingManager

## Purpose

Standardizes logging for all MedFabric modules.

## Public Methods

```python
initialize() -> None

get_logger(module_name: str) -> logging.Logger

start_step(step_name: str) -> None

end_step(step_name: str) -> None

log_dataset(
    dataset_name: str,
    row_count: int,
    column_count: int
) -> None

log_validation(
    validation_name: str,
    status: str,
    message: str
) -> None

log_exception(error: Exception) -> None

close() -> None
```

## Used For

- Console logging
- Pipeline logs
- Module logs
- Error logs
- Audit logs
- Run-level observability

## Rule

Business modules should not create custom loggers directly.

---

# 7. StorageManager

## Purpose

Centralizes all file input and output.

## Public Methods

```python
read_csv(path: str | Path) -> DataFrame

write_csv(dataframe: DataFrame, path: str | Path) -> None

read_parquet(path: str | Path) -> DataFrame

write_parquet(dataframe: DataFrame, path: str | Path) -> None

read_json(path: str | Path) -> dict

write_json(data: dict, path: str | Path) -> None

read_yaml(path: str | Path) -> dict

write_yaml(data: dict, path: str | Path) -> None

save_model(model: object, path: str | Path) -> None

load_model(path: str | Path) -> object

exists(path: str | Path) -> bool

delete(path: str | Path) -> None

copy(source: str | Path, target: str | Path) -> None

move(source: str | Path, target: str | Path) -> None
```

## Used For

- Reading datasets
- Writing datasets
- Reading configuration-style files
- Writing metadata outputs
- Saving model artifacts
- Loading model artifacts
- File movement
- File deletion

## Rule

Business modules should not directly call Pandas read/write methods.

---

# 8. ValidationManager

## Purpose

Centralizes validation and data quality checks.

## Public Methods

```python
validate_required_columns(
    dataframe: DataFrame,
    required_columns: list[str]
) -> None

validate_duplicates(
    dataframe: DataFrame,
    key_columns: list[str]
) -> None

validate_nulls(
    dataframe: DataFrame,
    columns: list[str]
) -> None

validate_datatypes(
    dataframe: DataFrame,
    expected_types: dict
) -> None

validate_primary_key(
    dataframe: DataFrame,
    key_columns: list[str]
) -> None

validate_foreign_key(
    child_dataframe: DataFrame,
    parent_dataframe: DataFrame,
    child_key: str,
    parent_key: str
) -> None

validate_schema(
    dataframe: DataFrame,
    schema_definition: dict
) -> None

validate_dataset(
    dataframe: DataFrame,
    validation_rules: dict
) -> None
```

## Used For

- Required column validation
- Primary key validation
- Duplicate detection
- Null validation
- Data type validation
- Referential integrity validation
- Dataset-level validation

## Rule

Validation logic should not be duplicated across business modules.

---

# 9. MetadataManager

## Purpose

Builds standardized metadata for governance outputs.

## Public Methods

```python
build_dataset_metadata(
    dataset_name: str,
    dataframe: DataFrame,
    source_path: str,
    output_path: str
) -> dict

build_column_metadata(
    dataset_name: str,
    dataframe: DataFrame
) -> list[dict]

build_statistics(
    dataset_name: str,
    dataframe: DataFrame
) -> dict

build_lineage(
    dataset_name: str,
    source_datasets: list[str],
    output_dataset: str
) -> dict

write_metadata(
    metadata: dict,
    output_path: str | Path
) -> None
```

## Used For

- Dataset inventory
- Column dictionary
- Metadata catalog
- Data lineage
- Data statistics
- Governance outputs

## Rule

Every generated dataset should have metadata.

---

# 10. ExceptionManager

## Purpose

Provides standard MedFabric exception classes.

## Public Exception Classes

```python
MedFabricError

ConfigurationError

PathError

LoggingError

StorageError

ValidationError

MetadataError

PipelineError

ModelError

GovernanceError
```

## Used For

- Standard error handling
- Clear failure messages
- Consistent debugging
- Pipeline failure classification

## Rule

Modules should raise MedFabric standard exceptions where practical.

---

# 11. PipelineContext

## Purpose

Stores current pipeline execution information.

## Required Attributes

```python
run_id: str

pipeline_name: str

environment: str

start_time: datetime

application_version: str

configuration: dict
```

## Optional Attributes

```python
user: str | None

logger: logging.Logger | None

metadata: dict
```

## Used For

- Passing run-level information across modules
- Tracking pipeline execution
- Logging run context
- Writing pipeline history
- Supporting auditability

---

# 12. Dependency Direction

Foundation components shall be used in this order:

```text
ConfigurationManager
        ↓
PathManager
        ↓
LoggingManager
        ↓
StorageManager
        ↓
ValidationManager
        ↓
MetadataManager
        ↓
Business Modules
```

This direction reduces circular dependencies and keeps the platform stable.

---

# 13. Foundation Design Principle

Foundation owns infrastructure.

Business modules own business logic.

Examples:

```python
# Not allowed in business modules
import yaml
import logging
import os
import shutil
```

```python
# Preferred usage
from src.common.configuration_manager import ConfigurationManager
from src.common.path_manager import PathManager
from src.common.logging_manager import LoggingManager
from src.common.storage_manager import StorageManager
from src.common.validation_manager import ValidationManager
from src.common.metadata_manager import MetadataManager
```

---

# 14. Change Control

This document is LOCKED.

Changes to the Foundation public interface require explicit approval.

If a Foundation component changes, this document must be updated in the same commit as the code change.

---

# 15. Completion Checklist

Before any Foundation component is considered complete, it must satisfy:

- Public methods match this document.
- File has complete MedFabric header.
- Inputs are documented.
- Outputs are documented.
- Dependencies are documented.
- Run command is included.
- Main section exists if executable.
- Error handling is clear.
- Test command is provided.
- Expected output is documented.

---

# End of Document