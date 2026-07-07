###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/common/constants.py
#
# Layer:
#     Layer 2D - Enterprise Modeling Framework
#
# Purpose:
#     Centralized constants used throughout the Modeling Framework.
#
# Responsibilities:
#     - Default configuration paths
#     - Default framework metadata
#     - Supported output formats
#     - Standard execution status values
#
# Notes:
#     This module intentionally contains only constants.
#     No executable business logic should be added here.
#
###############################################################################

from __future__ import annotations

###############################################################################
# Configuration Paths
###############################################################################

DEFAULT_CONFIG_PATH = "config/modeling/modeling.yaml"
DEFAULT_PIPELINE_CONFIG_PATH = "config/pipeline.yaml"

###############################################################################
# Framework Metadata
###############################################################################

DEFAULT_CAPABILITY_NAME = "Enterprise Modeling Framework"

DEFAULT_LAYER_NAME = (
    "Layer 2D - Enterprise Modeling Framework"
)

DEFAULT_DOMAIN_NAME = "Modeling"

###############################################################################
# Output Defaults
###############################################################################

DEFAULT_OUTPUT_FORMAT = "parquet"

SUPPORTED_OUTPUT_FORMATS = {
    "parquet",
    "csv",
    "json",
}

###############################################################################
# Standard Execution Status Values
###############################################################################

STATUS_SUCCESS = "SUCCESS"

STATUS_FAILED = "FAILED"

STATUS_WARNING = "WARNING"

STATUS_SKIPPED = "SKIPPED"

###############################################################################
# End of File
###############################################################################