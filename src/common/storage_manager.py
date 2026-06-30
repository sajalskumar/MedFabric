###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/storage_manager.py
#
# Purpose:
#     Provides centralized file input/output operations for the MedFabric
#     platform.
#
# Business Context:
#     MedFabric business modules should not directly call pandas, yaml, json,
#     pickle, joblib, shutil, or filesystem operations. This module standardizes
#     storage behavior so local development can later move toward cloud storage
#     with minimal business logic changes.
#
# Inputs:
#     CSV files
#     Parquet files
#     JSON files
#     YAML files
#     Model artifact files
#
# Outputs:
#     CSV files
#     Parquet files
#     JSON files
#     YAML files
#     Model artifact files
#
# Dependencies:
#     pandas
#     PyYAML
#     joblib
#     json
#     shutil
#     pathlib
#     typing
#     src.common.path_manager.PathManager
#     src.common.exception_manager.StorageError
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
#     read_csv()
#     write_csv()
#     read_parquet()
#     write_parquet()
#     read_json()
#     write_json()
#     read_yaml()
#     write_yaml()
#     save_model()
#     load_model()
#     exists()
#     delete()
#     copy()
#     move()
#
# Example Run Command:
#     python -m src.common.storage_manager
#
# Expected Output:
#     MedFabric storage manager validation completed successfully.
###############################################################################

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml

from src.common.configuration_manager import ConfigurationManager
from src.common.exception_manager import StorageError
from src.common.path_manager import PathManager


class StorageManager:
    """
    Centralized storage manager for MedFabric.

    This class owns local file input/output operations.

    Business modules should use this manager instead of directly calling pandas,
    yaml, json, joblib, or shutil.
    """

    def __init__(self, path_manager: PathManager) -> None:
        """
        Initialize StorageManager.

        Parameters
        ----------
        path_manager:
            Active MedFabric PathManager instance.
        """
        self.path_manager = path_manager

    def read_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Parameters
        ----------
        path:
            CSV file path.

        kwargs:
            Optional arguments passed to pandas.read_csv.

        Returns
        -------
        pandas.DataFrame
            Loaded DataFrame.
        """
        file_path = self._resolve_existing_file(path)

        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as error:
            raise StorageError(f"Unable to read CSV file: {file_path}") from error

    def write_csv(
        self,
        dataframe: pd.DataFrame,
        path: str | Path,
        index: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Write a pandas DataFrame to CSV.

        Parameters
        ----------
        dataframe:
            DataFrame to write.

        path:
            Output CSV file path.

        index:
            Whether to write the DataFrame index.

        kwargs:
            Optional arguments passed to pandas.DataFrame.to_csv.

        Returns
        -------
        Path
            Written file path.
        """
        self._validate_dataframe(dataframe)
        file_path = self._resolve_output_file(path)

        try:
            dataframe.to_csv(file_path, index=index, **kwargs)
            return file_path
        except Exception as error:
            raise StorageError(f"Unable to write CSV file: {file_path}") from error

    def read_parquet(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """
        Read a Parquet file into a pandas DataFrame.

        Parameters
        ----------
        path:
            Parquet file path.

        kwargs:
            Optional arguments passed to pandas.read_parquet.

        Returns
        -------
        pandas.DataFrame
            Loaded DataFrame.
        """
        file_path = self._resolve_existing_file(path)

        try:
            return pd.read_parquet(file_path, **kwargs)
        except Exception as error:
            raise StorageError(f"Unable to read Parquet file: {file_path}") from error

    def write_parquet(
        self,
        dataframe: pd.DataFrame,
        path: str | Path,
        index: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Write a pandas DataFrame to Parquet.

        Parameters
        ----------
        dataframe:
            DataFrame to write.

        path:
            Output Parquet file path.

        index:
            Whether to write the DataFrame index.

        kwargs:
            Optional arguments passed to pandas.DataFrame.to_parquet.

        Returns
        -------
        Path
            Written file path.
        """
        self._validate_dataframe(dataframe)
        file_path = self._resolve_output_file(path)

        try:
            dataframe.to_parquet(file_path, index=index, **kwargs)
            return file_path
        except Exception as error:
            raise StorageError(f"Unable to write Parquet file: {file_path}") from error

    def read_json(self, path: str | Path) -> Dict[str, Any]:
        """
        Read a JSON file.

        Parameters
        ----------
        path:
            JSON file path.

        Returns
        -------
        dict
            Parsed JSON content.
        """
        file_path = self._resolve_existing_file(path)

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as error:
            raise StorageError(f"Unable to read JSON file: {file_path}") from error

        if not isinstance(data, dict):
            raise StorageError(f"JSON file must contain an object: {file_path}")

        return data

    def write_json(
        self,
        data: Dict[str, Any],
        path: str | Path,
        indent: int = 2,
    ) -> Path:
        """
        Write a dictionary to JSON.

        Parameters
        ----------
        data:
            Dictionary to write.

        path:
            Output JSON file path.

        indent:
            JSON indentation level.

        Returns
        -------
        Path
            Written file path.
        """
        if not isinstance(data, dict):
            raise StorageError("write_json expects a dictionary.")

        file_path = self._resolve_output_file(path)

        try:
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=indent, default=str)
            return file_path
        except Exception as error:
            raise StorageError(f"Unable to write JSON file: {file_path}") from error

    def read_yaml(self, path: str | Path) -> Dict[str, Any]:
        """
        Read a YAML file.

        Parameters
        ----------
        path:
            YAML file path.

        Returns
        -------
        dict
            Parsed YAML content.
        """
        file_path = self._resolve_existing_file(path)

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise StorageError(f"Invalid YAML file: {file_path}") from error
        except Exception as error:
            raise StorageError(f"Unable to read YAML file: {file_path}") from error

        if data is None:
            raise StorageError(f"YAML file is empty: {file_path}")

        if not isinstance(data, dict):
            raise StorageError(f"YAML file must contain a mapping: {file_path}")

        return data

    def write_yaml(self, data: Dict[str, Any], path: str | Path) -> Path:
        """
        Write a dictionary to YAML.

        Parameters
        ----------
        data:
            Dictionary to write.

        path:
            Output YAML file path.

        Returns
        -------
        Path
            Written file path.
        """
        if not isinstance(data, dict):
            raise StorageError("write_yaml expects a dictionary.")

        file_path = self._resolve_output_file(path)

        try:
            with file_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(data, file, sort_keys=False)
            return file_path
        except Exception as error:
            raise StorageError(f"Unable to write YAML file: {file_path}") from error

    def save_model(self, model: object, path: str | Path) -> Path:
        """
        Save a model artifact using joblib.

        Parameters
        ----------
        model:
            Python model object to save.

        path:
            Output model artifact path.

        Returns
        -------
        Path
            Written model file path.
        """
        file_path = self._resolve_output_file(path)

        try:
            joblib.dump(model, file_path)
            return file_path
        except Exception as error:
            raise StorageError(f"Unable to save model artifact: {file_path}") from error

    def load_model(self, path: str | Path) -> object:
        """
        Load a model artifact using joblib.

        Parameters
        ----------
        path:
            Model artifact file path.

        Returns
        -------
        object
            Loaded model object.
        """
        file_path = self._resolve_existing_file(path)

        try:
            return joblib.load(file_path)
        except Exception as error:
            raise StorageError(f"Unable to load model artifact: {file_path}") from error

    def exists(self, path: str | Path) -> bool:
        """
        Check whether a file or directory exists.

        Parameters
        ----------
        path:
            Path to check.

        Returns
        -------
        bool
            True if path exists.
        """
        resolved_path = self.path_manager.resolve_path(path)
        return resolved_path.exists()

    def delete(self, path: str | Path) -> None:
        """
        Delete a file or directory.

        Parameters
        ----------
        path:
            File or directory path to delete.
        """
        resolved_path = self.path_manager.resolve_path(path)

        if not resolved_path.exists():
            return

        try:
            if resolved_path.is_dir():
                shutil.rmtree(resolved_path)
            else:
                resolved_path.unlink()
        except Exception as error:
            raise StorageError(f"Unable to delete path: {resolved_path}") from error

    def copy(self, source: str | Path, target: str | Path) -> Path:
        """
        Copy a file from source to target.

        Parameters
        ----------
        source:
            Source file path.

        target:
            Target file path.

        Returns
        -------
        Path
            Target file path.
        """
        source_path = self._resolve_existing_file(source)
        target_path = self._resolve_output_file(target)

        try:
            shutil.copy2(source_path, target_path)
            return target_path
        except Exception as error:
            raise StorageError(
                f"Unable to copy file from {source_path} to {target_path}"
            ) from error

    def move(self, source: str | Path, target: str | Path) -> Path:
        """
        Move a file from source to target.

        Parameters
        ----------
        source:
            Source file path.

        target:
            Target file path.

        Returns
        -------
        Path
            Target file path.
        """
        source_path = self._resolve_existing_file(source)
        target_path = self._resolve_output_file(target)

        try:
            shutil.move(str(source_path), str(target_path))
            return target_path
        except Exception as error:
            raise StorageError(
                f"Unable to move file from {source_path} to {target_path}"
            ) from error

    def _resolve_existing_file(self, path: str | Path) -> Path:
        """
        Resolve and validate an existing file path.

        Parameters
        ----------
        path:
            File path.

        Returns
        -------
        Path
            Resolved file path.
        """
        file_path = self.path_manager.resolve_path(path)

        if not file_path.exists():
            raise StorageError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise StorageError(f"Path is not a file: {file_path}")

        return file_path

    def _resolve_output_file(self, path: str | Path) -> Path:
        """
        Resolve an output file path and ensure its parent directory exists.

        Parameters
        ----------
        path:
            Output file path.

        Returns
        -------
        Path
            Resolved output file path.
        """
        file_path = self.path_manager.resolve_path(path)
        self.path_manager.ensure_directory(file_path.parent)
        return file_path

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        """
        Validate that an object is a pandas DataFrame.

        Parameters
        ----------
        dataframe:
            Object expected to be a DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise StorageError("Expected a pandas DataFrame.")


def main() -> None:
    """
    Standalone verification entry point.

    Run Command
    -----------
    python -m src.common.storage_manager

    Expected Output
    ---------------
    MedFabric storage manager validation completed successfully.
    """
    configuration_manager = ConfigurationManager()
    path_manager = PathManager(configuration_manager)
    storage_manager = StorageManager(path_manager)

    test_directory = path_manager.get_path("temporary", "root")
    path_manager.ensure_directory(test_directory)

    test_dataframe = pd.DataFrame(
        {
            "member_id": [1, 2, 3],
            "member_name": ["A", "B", "C"],
            "risk_score": [0.10, 0.20, 0.30],
        }
    )

    csv_path = test_directory / "storage_manager_test.csv"
    parquet_path = test_directory / "storage_manager_test.parquet"
    json_path = test_directory / "storage_manager_test.json"
    yaml_path = test_directory / "storage_manager_test.yaml"
    model_path = test_directory / "storage_manager_test.joblib"

    storage_manager.write_csv(test_dataframe, csv_path)
    csv_result = storage_manager.read_csv(csv_path)

    storage_manager.write_parquet(test_dataframe, parquet_path)
    parquet_result = storage_manager.read_parquet(parquet_path)

    storage_manager.write_json({"status": "success"}, json_path)
    json_result = storage_manager.read_json(json_path)

    storage_manager.write_yaml({"status": "success"}, yaml_path)
    yaml_result = storage_manager.read_yaml(yaml_path)

    storage_manager.save_model({"model_type": "test"}, model_path)
    model_result = storage_manager.load_model(model_path)

    if len(csv_result) != 3:
        raise StorageError("CSV validation failed.")

    if len(parquet_result) != 3:
        raise StorageError("Parquet validation failed.")

    if json_result.get("status") != "success":
        raise StorageError("JSON validation failed.")

    if yaml_result.get("status") != "success":
        raise StorageError("YAML validation failed.")

    if model_result.get("model_type") != "test":
        raise StorageError("Model validation failed.")

    print("MedFabric storage manager validation completed successfully.")


if __name__ == "__main__":
    main()