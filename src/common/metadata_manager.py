###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/metadata_manager.py
#
# Purpose:
#     Provides centralized metadata generation for MedFabric datasets,
#     columns, statistics, and lineage records.
#
# Business Context:
#     MedFabric is governance-first. Every major data product should produce
#     standardized metadata so the platform can support dataset inventory,
#     column dictionary, lineage, auditability, and data quality transparency.
#
# Inputs:
#     pandas DataFrames
#     Dataset names
#     Source paths
#     Output paths
#     Source dataset names
#
# Outputs:
#     Dataset metadata dictionaries
#     Column metadata records
#     Dataset statistics dictionaries
#     Lineage dictionaries
#     Metadata files when write_metadata() is used
#
# Dependencies:
#     pandas
#     datetime
#     pathlib
#     typing
#     src.common.storage_manager.StorageManager
#     src.common.exception_manager.MetadataError
#
# Used By:
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
#     build_dataset_metadata()
#     build_column_metadata()
#     build_statistics()
#     build_lineage()
#     write_metadata()
#
# Example Run Command:
#     python -m src.common.metadata_manager
#
# Expected Output:
#     MedFabric metadata manager validation completed successfully.
###############################################################################

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.configuration_manager import ConfigurationManager
from src.common.exception_manager import MetadataError
from src.common.path_manager import PathManager
from src.common.storage_manager import StorageManager


class MetadataManager:
    """
    Centralized metadata manager for MedFabric.

    This class owns reusable metadata generation logic across the platform.

    Business modules should use this manager instead of creating inconsistent
    metadata records independently.
    """

    def __init__(self, storage_manager: StorageManager) -> None:
        """
        Initialize MetadataManager.

        Parameters
        ----------
        storage_manager:
            Active MedFabric StorageManager instance used to write metadata files.
        """
        self.storage_manager = storage_manager

    def build_dataset_metadata(
        self,
        dataset_name: str,
        dataframe: pd.DataFrame,
        source_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        layer: Optional[str] = None,
        domain: Optional[str] = None,
        primary_key: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build standardized dataset-level metadata.

        Parameters
        ----------
        dataset_name:
            Name of the dataset.

        dataframe:
            DataFrame represented by this metadata.

        source_path:
            Optional source file path.

        output_path:
            Optional output file path.

        layer:
            Optional MedFabric layer name such as raw, bronze, silver, gold,
            feature_store, modeling, scoring, or governance.

        domain:
            Optional business or technical domain.

        primary_key:
            Optional list of primary key columns.

        description:
            Optional dataset description.

        Returns
        -------
        dict
            Dataset metadata dictionary.

        Raises
        ------
        MetadataError
            Raised when inputs are invalid.
        """
        self._validate_dataset_name(dataset_name)
        self._validate_dataframe(dataframe, dataset_name)

        metadata = {
            "dataset_name": dataset_name,
            "description": description or "",
            "layer": layer or "",
            "domain": domain or "",
            "primary_key": primary_key or [],
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "columns": list(dataframe.columns),
            "source_path": str(source_path) if source_path is not None else "",
            "output_path": str(output_path) if output_path is not None else "",
            "created_at": self._current_timestamp(),
            "metadata_version": "1.0",
        }

        return metadata

    def build_column_metadata(
        self,
        dataset_name: str,
        dataframe: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Build standardized column-level metadata.

        Parameters
        ----------
        dataset_name:
            Name of the dataset.

        dataframe:
            DataFrame whose columns should be documented.

        Returns
        -------
        list[dict]
            Column metadata records.

        Raises
        ------
        MetadataError
            Raised when inputs are invalid.
        """
        self._validate_dataset_name(dataset_name)
        self._validate_dataframe(dataframe, dataset_name)

        column_metadata: List[Dict[str, Any]] = []

        for ordinal_position, column_name in enumerate(dataframe.columns, start=1):
            series = dataframe[column_name]

            column_metadata.append(
                {
                    "dataset_name": dataset_name,
                    "column_name": column_name,
                    "ordinal_position": ordinal_position,
                    "data_type": str(series.dtype),
                    "nullable": bool(series.isna().any()),
                    "null_count": int(series.isna().sum()),
                    "non_null_count": int(series.notna().sum()),
                    "distinct_count": int(series.nunique(dropna=True)),
                    "created_at": self._current_timestamp(),
                    "metadata_version": "1.0",
                }
            )

        return column_metadata

    def build_statistics(
        self,
        dataset_name: str,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Build standardized dataset statistics.

        Parameters
        ----------
        dataset_name:
            Name of the dataset.

        dataframe:
            DataFrame to summarize.

        Returns
        -------
        dict
            Dataset statistics dictionary.

        Raises
        ------
        MetadataError
            Raised when inputs are invalid.
        """
        self._validate_dataset_name(dataset_name)
        self._validate_dataframe(dataframe, dataset_name)

        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        datetime_columns = dataframe.select_dtypes(include="datetime").columns.tolist()
        object_columns = dataframe.select_dtypes(include="object").columns.tolist()

        statistics: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "numeric_column_count": int(len(numeric_columns)),
            "datetime_column_count": int(len(datetime_columns)),
            "object_column_count": int(len(object_columns)),
            "total_null_count": int(dataframe.isna().sum().sum()),
            "columns_with_nulls": [
                column
                for column in dataframe.columns
                if int(dataframe[column].isna().sum()) > 0
            ],
            "numeric_summary": {},
            "created_at": self._current_timestamp(),
            "metadata_version": "1.0",
        }

        for column_name in numeric_columns:
            series = dataframe[column_name]

            statistics["numeric_summary"][column_name] = {
                "min": self._safe_scalar(series.min()),
                "max": self._safe_scalar(series.max()),
                "mean": self._safe_scalar(series.mean()),
                "median": self._safe_scalar(series.median()),
                "standard_deviation": self._safe_scalar(series.std()),
            }

        return statistics

    def build_lineage(
        self,
        dataset_name: str,
        source_datasets: List[str],
        output_dataset: str,
        transformation_name: Optional[str] = None,
        module_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build standardized lineage metadata.

        Parameters
        ----------
        dataset_name:
            Name of the logical dataset or process being documented.

        source_datasets:
            List of source dataset names.

        output_dataset:
            Output dataset name.

        transformation_name:
            Optional transformation or business process name.

        module_name:
            Optional Python module name that produced the output.

        Returns
        -------
        dict
            Lineage metadata dictionary.

        Raises
        ------
        MetadataError
            Raised when lineage inputs are invalid.
        """
        self._validate_dataset_name(dataset_name)

        if not isinstance(source_datasets, list):
            raise MetadataError("source_datasets must be a list.")

        if not output_dataset:
            raise MetadataError("output_dataset cannot be empty.")

        lineage = {
            "dataset_name": dataset_name,
            "source_datasets": source_datasets,
            "output_dataset": output_dataset,
            "transformation_name": transformation_name or "",
            "module_name": module_name or "",
            "created_at": self._current_timestamp(),
            "metadata_version": "1.0",
        }

        return lineage

    def write_metadata(
        self,
        metadata: Dict[str, Any] | List[Dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """
        Write metadata to disk.

        The output format is determined by file extension.

        Supported extensions:
        - .json
        - .yaml
        - .yml
        - .csv

        Parameters
        ----------
        metadata:
            Metadata dictionary or list of metadata dictionaries.

        output_path:
            Output metadata path.

        Returns
        -------
        Path
            Written output path.

        Raises
        ------
        MetadataError
            Raised when metadata cannot be written.
        """
        if metadata is None:
            raise MetadataError("metadata cannot be None.")

        file_path = Path(output_path)
        extension = file_path.suffix.lower()

        try:
            if extension == ".json":
                if isinstance(metadata, list):
                    return self.storage_manager.write_json(
                        {"records": metadata},
                        output_path,
                    )
                return self.storage_manager.write_json(metadata, output_path)

            if extension in [".yaml", ".yml"]:
                if isinstance(metadata, list):
                    return self.storage_manager.write_yaml(
                        {"records": metadata},
                        output_path,
                    )
                return self.storage_manager.write_yaml(metadata, output_path)

            if extension == ".csv":
                dataframe = pd.DataFrame(metadata if isinstance(metadata, list) else [metadata])
                return self.storage_manager.write_csv(dataframe, output_path)

            raise MetadataError(
                f"Unsupported metadata output extension: {extension}"
            )

        except Exception as error:
            if isinstance(error, MetadataError):
                raise
            raise MetadataError(
                f"Unable to write metadata to: {output_path}"
            ) from error

    @staticmethod
    def _validate_dataset_name(dataset_name: str) -> None:
        """
        Validate dataset name.

        Parameters
        ----------
        dataset_name:
            Dataset name to validate.

        Raises
        ------
        MetadataError
            Raised when dataset name is invalid.
        """
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise MetadataError("dataset_name must be a non-empty string.")

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame, dataset_name: str) -> None:
        """
        Validate DataFrame input.

        Parameters
        ----------
        dataframe:
            Object expected to be a DataFrame.

        dataset_name:
            Dataset name used in error messages.

        Raises
        ------
        MetadataError
            Raised when the object is not a DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise MetadataError(
                f"{dataset_name} must be a pandas DataFrame."
            )

    @staticmethod
    def _current_timestamp() -> str:
        """
        Return current timestamp as a string.

        Returns
        -------
        str
            Current timestamp.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _safe_scalar(value: Any) -> Any:
        """
        Convert pandas/numpy scalar values into JSON-friendly values.

        Parameters
        ----------
        value:
            Scalar value.

        Returns
        -------
        Any
            JSON-friendly scalar value.
        """
        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            return value.item()

        return value


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.metadata_manager

    Expected Output
    ---------------
    MedFabric metadata manager validation completed successfully.
    """
    configuration_manager = ConfigurationManager()
    path_manager = PathManager(configuration_manager)
    storage_manager = StorageManager(path_manager)
    metadata_manager = MetadataManager(storage_manager)

    test_dataframe = pd.DataFrame(
        {
            "member_id": [1, 2, 3],
            "member_name": ["A", "B", "C"],
            "risk_score": [0.10, 0.20, 0.30],
        }
    )

    dataset_metadata = metadata_manager.build_dataset_metadata(
        dataset_name="Metadata_Manager_Test_Dataset",
        dataframe=test_dataframe,
        source_path="data/raw/test.csv",
        output_path="data/metadata/metadata_manager_test_dataset.json",
        layer="metadata",
        domain="foundation",
        primary_key=["member_id"],
        description="Test dataset used to validate MetadataManager.",
    )

    column_metadata = metadata_manager.build_column_metadata(
        dataset_name="Metadata_Manager_Test_Dataset",
        dataframe=test_dataframe,
    )

    statistics = metadata_manager.build_statistics(
        dataset_name="Metadata_Manager_Test_Dataset",
        dataframe=test_dataframe,
    )

    lineage = metadata_manager.build_lineage(
        dataset_name="Metadata_Manager_Test_Dataset",
        source_datasets=["Test_Source"],
        output_dataset="Metadata_Manager_Test_Dataset",
        transformation_name="MetadataManager standalone validation",
        module_name="src.common.metadata_manager",
    )

    output_directory = path_manager.get_data_path("metadata")
    path_manager.ensure_directory(output_directory)

    metadata_manager.write_metadata(
        metadata=dataset_metadata,
        output_path=output_directory / "metadata_manager_dataset_metadata.json",
    )

    metadata_manager.write_metadata(
        metadata=column_metadata,
        output_path=output_directory / "metadata_manager_column_metadata.csv",
    )

    metadata_manager.write_metadata(
        metadata=statistics,
        output_path=output_directory / "metadata_manager_statistics.json",
    )

    metadata_manager.write_metadata(
        metadata=lineage,
        output_path=output_directory / "metadata_manager_lineage.json",
    )

    print("MedFabric metadata manager validation completed successfully.")


if __name__ == "__main__":
    main()