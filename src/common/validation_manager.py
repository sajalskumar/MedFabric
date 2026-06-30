###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/validation_manager.py
#
# Purpose:
#     Provides centralized validation and data quality checks for the MedFabric
#     platform.
#
# Business Context:
#     MedFabric produces healthcare analytical datasets across Raw, Bronze,
#     Silver, Gold, Feature Store, Modeling, Scoring, and Governance layers.
#     This module standardizes validation so business modules do not duplicate
#     required column checks, null checks, duplicate checks, primary key checks,
#     foreign key checks, schema checks, or dataset-level validation logic.
#
# Inputs:
#     pandas DataFrames
#     Validation rule dictionaries
#
# Outputs:
#     Validation pass/fail results
#     Validation exceptions when rules fail
#
# Dependencies:
#     pandas
#     typing
#     src.common.exception_manager.ValidationError
#
# Used By:
#     src/data_generation/*
#     src/ingestion/*
#     src/silver/*
#     src/gold/*
#     src/feature_store/*
#     src/modeling/*
#     src/scoring/*
#     src/governance/*
#     src/pipeline/*
#
# Public Interface:
#     validate_required_columns()
#     validate_duplicates()
#     validate_nulls()
#     validate_datatypes()
#     validate_primary_key()
#     validate_foreign_key()
#     validate_schema()
#     validate_dataset()
#
# Example Run Command:
#     python -m src.common.validation_manager
#
# Expected Output:
#     MedFabric validation manager validation completed successfully.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.common.exception_manager import ValidationError


class ValidationManager:
    """
    Centralized validation manager for MedFabric.

    This class owns reusable validation behavior across the platform.

    Business modules should use this manager instead of writing duplicate
    validation logic.
    """

    def validate_required_columns(
        self,
        dataframe: pd.DataFrame,
        required_columns: List[str],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate that a DataFrame contains all required columns.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        required_columns:
            Required column names.

        dataset_name:
            Human-readable dataset name used in error messages.

        Raises
        ------
        ValidationError
            Raised when required columns are missing.
        """
        self._validate_dataframe(dataframe, dataset_name)
        self._validate_column_list(required_columns, "required_columns")

        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValidationError(
                f"{dataset_name} is missing required columns: {missing_columns}"
            )

    def validate_duplicates(
        self,
        dataframe: pd.DataFrame,
        key_columns: List[str],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate that key columns do not contain duplicate combinations.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        key_columns:
            Columns that should uniquely identify rows.

        dataset_name:
            Human-readable dataset name used in error messages.

        Raises
        ------
        ValidationError
            Raised when duplicate key combinations are found.
        """
        self._validate_dataframe(dataframe, dataset_name)
        self.validate_required_columns(dataframe, key_columns, dataset_name)
        self._validate_column_list(key_columns, "key_columns")

        duplicate_count = int(dataframe.duplicated(subset=key_columns).sum())

        if duplicate_count > 0:
            raise ValidationError(
                f"{dataset_name} contains {duplicate_count} duplicate rows "
                f"for key columns: {key_columns}"
            )

    def validate_nulls(
        self,
        dataframe: pd.DataFrame,
        columns: List[str],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate that selected columns do not contain null values.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        columns:
            Columns that should not contain null values.

        dataset_name:
            Human-readable dataset name used in error messages.

        Raises
        ------
        ValidationError
            Raised when null values are found.
        """
        self._validate_dataframe(dataframe, dataset_name)
        self.validate_required_columns(dataframe, columns, dataset_name)
        self._validate_column_list(columns, "columns")

        null_counts = dataframe[columns].isna().sum()
        failing_columns = {
            column: int(count)
            for column, count in null_counts.items()
            if int(count) > 0
        }

        if failing_columns:
            raise ValidationError(
                f"{dataset_name} contains null values in required columns: "
                f"{failing_columns}"
            )

    def validate_datatypes(
        self,
        dataframe: pd.DataFrame,
        expected_types: Dict[str, str],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate DataFrame column data types.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        expected_types:
            Dictionary mapping column names to expected pandas dtype strings.

        dataset_name:
            Human-readable dataset name used in error messages.

        Raises
        ------
        ValidationError
            Raised when actual column types do not match expected types.

        Notes
        -----
        This validation compares pandas dtype names as strings.
        Example expected types:
            - int64
            - float64
            - object
            - bool
            - datetime64[ns]
        """
        self._validate_dataframe(dataframe, dataset_name)

        if not isinstance(expected_types, dict):
            raise ValidationError("expected_types must be a dictionary.")

        self.validate_required_columns(
            dataframe=dataframe,
            required_columns=list(expected_types.keys()),
            dataset_name=dataset_name,
        )

        type_failures: Dict[str, Dict[str, str]] = {}

        for column_name, expected_type in expected_types.items():
            actual_type = str(dataframe[column_name].dtype)

            if actual_type != expected_type:
                type_failures[column_name] = {
                    "expected": expected_type,
                    "actual": actual_type,
                }

        if type_failures:
            raise ValidationError(
                f"{dataset_name} contains datatype mismatches: {type_failures}"
            )

    def validate_primary_key(
        self,
        dataframe: pd.DataFrame,
        key_columns: List[str],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate primary key rules.

        A primary key must:
        - Exist in the DataFrame
        - Not contain null values
        - Be unique

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        key_columns:
            Primary key columns.

        dataset_name:
            Human-readable dataset name used in error messages.

        Raises
        ------
        ValidationError
            Raised when primary key validation fails.
        """
        self.validate_required_columns(dataframe, key_columns, dataset_name)
        self.validate_nulls(dataframe, key_columns, dataset_name)
        self.validate_duplicates(dataframe, key_columns, dataset_name)

    def validate_foreign_key(
        self,
        child_dataframe: pd.DataFrame,
        parent_dataframe: pd.DataFrame,
        child_key: str,
        parent_key: str,
        child_dataset_name: str = "child_dataset",
        parent_dataset_name: str = "parent_dataset",
    ) -> None:
        """
        Validate referential integrity between child and parent datasets.

        Parameters
        ----------
        child_dataframe:
            Child DataFrame containing the foreign key.

        parent_dataframe:
            Parent DataFrame containing the referenced key.

        child_key:
            Foreign key column in the child DataFrame.

        parent_key:
            Primary/reference key column in the parent DataFrame.

        child_dataset_name:
            Name of the child dataset.

        parent_dataset_name:
            Name of the parent dataset.

        Raises
        ------
        ValidationError
            Raised when child keys are not found in the parent dataset.
        """
        self._validate_dataframe(child_dataframe, child_dataset_name)
        self._validate_dataframe(parent_dataframe, parent_dataset_name)

        self.validate_required_columns(
            child_dataframe,
            [child_key],
            child_dataset_name,
        )
        self.validate_required_columns(
            parent_dataframe,
            [parent_key],
            parent_dataset_name,
        )

        child_values = set(child_dataframe[child_key].dropna().unique())
        parent_values = set(parent_dataframe[parent_key].dropna().unique())

        missing_values = sorted(child_values - parent_values)

        if missing_values:
            preview = missing_values[:20]
            raise ValidationError(
                f"Foreign key validation failed. "
                f"{child_dataset_name}.{child_key} contains values not found in "
                f"{parent_dataset_name}.{parent_key}. "
                f"Missing count: {len(missing_values)}. "
                f"Preview: {preview}"
            )

    def validate_schema(
        self,
        dataframe: pd.DataFrame,
        schema_definition: Dict[str, Any],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate a DataFrame against a schema definition.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        schema_definition:
            Schema dictionary.

        dataset_name:
            Human-readable dataset name used in error messages.

        Supported Schema Keys
        ---------------------
        required_columns:
            List of required columns.

        primary_key:
            List of primary key columns.

        no_nulls:
            List of columns that cannot contain nulls.

        unique:
            List of columns that must be unique as a combined key.

        datatypes:
            Dictionary mapping columns to expected pandas dtype strings.

        Raises
        ------
        ValidationError
            Raised when schema validation fails.
        """
        self.validate_dataset(
            dataframe=dataframe,
            validation_rules=schema_definition,
            dataset_name=dataset_name,
        )

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
        validation_rules: Dict[str, Any],
        dataset_name: str = "dataset",
    ) -> None:
        """
        Validate a DataFrame using a rule dictionary.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        validation_rules:
            Dictionary defining validation rules.

        dataset_name:
            Human-readable dataset name used in error messages.

        Supported Rule Keys
        -------------------
        required_columns:
            List of required columns.

        primary_key:
            List of primary key columns.

        unique:
            List of columns that must be unique as a combined key.

        no_nulls:
            List of columns that cannot contain null values.

        datatypes:
            Dictionary mapping column names to expected pandas dtype strings.

        min_rows:
            Minimum required row count.

        max_rows:
            Maximum allowed row count.

        allow_empty:
            Boolean indicating whether an empty DataFrame is allowed.

        Raises
        ------
        ValidationError
            Raised when any validation rule fails.
        """
        self._validate_dataframe(dataframe, dataset_name)

        if not isinstance(validation_rules, dict):
            raise ValidationError("validation_rules must be a dictionary.")

        allow_empty = bool(validation_rules.get("allow_empty", False))

        if not allow_empty and dataframe.empty:
            raise ValidationError(f"{dataset_name} is empty.")

        if "min_rows" in validation_rules:
            self._validate_min_rows(
                dataframe=dataframe,
                min_rows=int(validation_rules["min_rows"]),
                dataset_name=dataset_name,
            )

        if "max_rows" in validation_rules:
            self._validate_max_rows(
                dataframe=dataframe,
                max_rows=int(validation_rules["max_rows"]),
                dataset_name=dataset_name,
            )

        if "required_columns" in validation_rules:
            self.validate_required_columns(
                dataframe=dataframe,
                required_columns=list(validation_rules["required_columns"]),
                dataset_name=dataset_name,
            )

        if "primary_key" in validation_rules:
            self.validate_primary_key(
                dataframe=dataframe,
                key_columns=list(validation_rules["primary_key"]),
                dataset_name=dataset_name,
            )

        if "unique" in validation_rules:
            self.validate_duplicates(
                dataframe=dataframe,
                key_columns=list(validation_rules["unique"]),
                dataset_name=dataset_name,
            )

        if "no_nulls" in validation_rules:
            self.validate_nulls(
                dataframe=dataframe,
                columns=list(validation_rules["no_nulls"]),
                dataset_name=dataset_name,
            )

        if "datatypes" in validation_rules:
            self.validate_datatypes(
                dataframe=dataframe,
                expected_types=dict(validation_rules["datatypes"]),
                dataset_name=dataset_name,
            )

    def _validate_min_rows(
        self,
        dataframe: pd.DataFrame,
        min_rows: int,
        dataset_name: str,
    ) -> None:
        """
        Validate minimum row count.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        min_rows:
            Minimum required row count.

        dataset_name:
            Human-readable dataset name used in error messages.
        """
        if len(dataframe) < min_rows:
            raise ValidationError(
                f"{dataset_name} contains {len(dataframe)} rows, "
                f"but at least {min_rows} rows are required."
            )

    def _validate_max_rows(
        self,
        dataframe: pd.DataFrame,
        max_rows: int,
        dataset_name: str,
    ) -> None:
        """
        Validate maximum row count.

        Parameters
        ----------
        dataframe:
            DataFrame to validate.

        max_rows:
            Maximum allowed row count.

        dataset_name:
            Human-readable dataset name used in error messages.
        """
        if len(dataframe) > max_rows:
            raise ValidationError(
                f"{dataset_name} contains {len(dataframe)} rows, "
                f"but no more than {max_rows} rows are allowed."
            )

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame, dataset_name: str) -> None:
        """
        Validate that an object is a pandas DataFrame.

        Parameters
        ----------
        dataframe:
            Object expected to be a DataFrame.

        dataset_name:
            Human-readable dataset name used in error messages.

        Raises
        ------
        ValidationError
            Raised when the object is not a DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValidationError(
                f"{dataset_name} must be a pandas DataFrame."
            )

    @staticmethod
    def _validate_column_list(columns: Iterable[str], parameter_name: str) -> None:
        """
        Validate that a column list is valid.

        Parameters
        ----------
        columns:
            Iterable of column names.

        parameter_name:
            Parameter name used in error messages.

        Raises
        ------
        ValidationError
            Raised when the column list is invalid.
        """
        if columns is None:
            raise ValidationError(f"{parameter_name} cannot be None.")

        column_list = list(columns)

        if not column_list:
            raise ValidationError(f"{parameter_name} cannot be empty.")

        invalid_columns = [
            column for column in column_list if not isinstance(column, str) or not column
        ]

        if invalid_columns:
            raise ValidationError(
                f"{parameter_name} contains invalid column names: {invalid_columns}"
            )


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.validation_manager

    Expected Output
    ---------------
    MedFabric validation manager validation completed successfully.
    """
    validation_manager = ValidationManager()

    test_dataframe = pd.DataFrame(
        {
            "member_id": [1, 2, 3],
            "member_name": ["A", "B", "C"],
            "risk_score": [0.10, 0.20, 0.30],
        }
    )

    validation_rules = {
        "required_columns": [
            "member_id",
            "member_name",
            "risk_score",
        ],
        "primary_key": ["member_id"],
        "no_nulls": [
            "member_id",
            "member_name",
        ],
        "datatypes": {
            "member_id": "int64",
            "member_name": "object",
            "risk_score": "float64",
        },
        "min_rows": 1,
        "allow_empty": False,
    }

    validation_manager.validate_dataset(
        dataframe=test_dataframe,
        validation_rules=validation_rules,
        dataset_name="Validation_Manager_Test_Dataset",
    )

    print("MedFabric validation manager validation completed successfully.")


if __name__ == "__main__":
    main()