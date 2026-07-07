###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/io_utils.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Provides shared input/output helpers for Modeling Framework datasets and
#     Python model artifacts.
#
# Responsibilities:
#     - Read configured input datasets.
#     - Write Modeling output datasets.
#     - Persist trained model objects as pickle files.
#
###############################################################################

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.modeling.common.output_paths import ensure_directory


def read_dataset(path: Path, file_format: str) -> pd.DataFrame:
    """
    Read a dataset from disk using the configured file format.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    if file_format == "parquet":
        return pd.read_parquet(path)

    if file_format == "csv":
        return pd.read_csv(path)

    if file_format == "json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported input file format: {file_format}")


def write_dataset(
    dataframe: pd.DataFrame,
    path: Path,
    file_format: str,
) -> None:
    """
    Write a dataframe to disk using the configured output file format.
    """

    ensure_directory(path.parent)

    if file_format == "parquet":
        dataframe.to_parquet(path, index=False)
        return

    if file_format == "csv":
        dataframe.to_csv(path, index=False)
        return

    if file_format == "json":
        dataframe.to_json(path, orient="records", indent=2)
        return

    raise ValueError(f"Unsupported output file format: {file_format}")


def save_pickle_object(obj: Any, path: Path) -> None:
    """
    Save a Python object as a pickle artifact.
    """

    ensure_directory(path.parent)

    with path.open("wb") as file:
        pickle.dump(obj, file)


###############################################################################
# End of File
###############################################################################