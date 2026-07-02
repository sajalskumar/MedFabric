###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/training/preprocessing.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Builds reusable preprocessing pipelines for all MedFabric machine
#     learning models.
#
# Architectural Notes:
#     - Independent of any specific predictive model.
#     - Independent of Analytics Platform.
#     - Shared by every classifier and future regression model.
#     - Configuration-driven preprocessing.
#
# Run:
#     python -m src.modeling.training.preprocessing
#
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


###############################################################################
# Column Discovery
###############################################################################

def discover_feature_types(
    dataframe: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical feature columns.

    Numeric columns include:
        - integer
        - float
        - boolean

    All remaining columns are treated as categorical.
    """

    numeric_columns = dataframe.select_dtypes(
        include=[
            "number",
            "bool",
        ]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    return numeric_columns, categorical_columns


###############################################################################
# Missing Value Normalization
###############################################################################

def normalize_categorical_missing_values(
    dataframe: pd.DataFrame,
    categorical_columns: List[str],
) -> pd.DataFrame:
    """
    Normalize categorical missing values before sklearn preprocessing.

    Why this exists
    ---------------
    scikit-learn SimpleImputer can fail when categorical columns contain a mix
    of Python None and string values because it may try to compare None with
    strings while determining the most frequent value.

    This function converts categorical columns to object/string-safe values and
    replaces all missing values with np.nan so SimpleImputer can process them
    consistently.
    """

    normalized = dataframe.copy()

    for column in categorical_columns:
        normalized[column] = normalized[column].replace({None: np.nan})

    return normalized


###############################################################################
# Numeric Pipeline
###############################################################################

def build_numeric_pipeline(
    preprocessing_config: Dict[str, Any],
) -> Pipeline:
    """
    Build numeric preprocessing pipeline.
    """

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config.get(
                    "numeric_imputation_strategy",
                    "median",
                ),
            ),
        )
    ]

    if bool(preprocessing_config.get("scale_numeric_features", False)):
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps)


###############################################################################
# Categorical Pipeline
###############################################################################

def build_categorical_pipeline(
    preprocessing_config: Dict[str, Any],
) -> Pipeline:
    """
    Build categorical preprocessing pipeline.
    """

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config.get(
                    "categorical_imputation_strategy",
                    "most_frequent",
                ),
                missing_values=np.nan,
            ),
        )
    ]

    if bool(preprocessing_config.get("one_hot_encode_categorical_features", True)):
        steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        )

    return Pipeline(steps)


###############################################################################
# Column Transformer
###############################################################################

def build_preprocessor(
    dataframe: pd.DataFrame,
    preprocessing_config: Dict[str, Any],
) -> ColumnTransformer:
    """
    Build complete preprocessing pipeline.
    """

    numeric_columns, categorical_columns = discover_feature_types(dataframe)

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                build_numeric_pipeline(preprocessing_config),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                build_categorical_pipeline(preprocessing_config),
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def prepare_features_for_preprocessing(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare feature dataframe before passing into sklearn pipeline.

    This keeps the public preprocessing contract simple:
        - caller passes raw feature dataframe
        - this module normalizes missing categorical values
        - sklearn handles imputation and encoding
    """

    numeric_columns, categorical_columns = discover_feature_types(dataframe)

    return normalize_categorical_missing_values(
        dataframe=dataframe,
        categorical_columns=categorical_columns,
    )


###############################################################################
# Validation
###############################################################################

def main() -> None:
    """
    Validate preprocessing framework.
    """

    dataframe = pd.DataFrame(
        {
            "age": [30, 40, None],
            "gender": ["M", "F", None],
            "paid": [100.5, 200.0, None],
            "state": ["AZ", "CA", "AZ"],
        }
    )

    preprocessing_config = {
        "numeric_imputation_strategy": "median",
        "categorical_imputation_strategy": "most_frequent",
        "scale_numeric_features": False,
        "one_hot_encode_categorical_features": True,
    }

    prepared_dataframe = prepare_features_for_preprocessing(dataframe)

    preprocessor = build_preprocessor(
        prepared_dataframe,
        preprocessing_config,
    )

    transformed = preprocessor.fit_transform(prepared_dataframe)

    print("Preprocessing validation successful.")
    print(f"Input columns : {len(dataframe.columns)}")
    print(f"Output shape  : {transformed.shape}")


if __name__ == "__main__":
    main()