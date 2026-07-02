###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/analytics_platform/common/runtime.py
#
# Layer:
#     Layer 2 - Analytics Platform
#
# Purpose:
#     Defines shared runtime objects for Analytics Platform domain builders.
#
#     This module does not replace Layer 0 common managers. It wraps the
#     existing PipelineContext and provides a consistent Analytics Platform
#     runtime shape for domain-level builders.
#
# Dependencies:
#     src.common.pipeline_context.PipelineContext
#
# Used By:
#     src/analytics_platform/*/build_*_layer.py
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.common.pipeline_context import PipelineContext


###############################################################################
# Constants
###############################################################################

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_WARNING = "WARNING"
STATUS_SKIPPED = "SKIPPED"

DEFAULT_OUTPUT_FORMAT = "parquet"

SUPPORTED_FILE_FORMATS = {"parquet", "csv", "json"}


###############################################################################
# Time Helpers
###############################################################################

def utc_now() -> datetime:
    """
    Purpose
    -------
    Return the current timezone-aware UTC timestamp.

    Parameters
    ----------
    None

    Returns
    -------
    datetime
        Current UTC timestamp.

    Raises
    ------
    None

    Notes
    -----
    Analytics Platform outputs use UTC timestamps for repeatable audit and
    metadata records.
    """
    return datetime.now(timezone.utc)


###############################################################################
# Runtime Objects
###############################################################################

@dataclass
class AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Shared runtime context for one Analytics Platform domain execution.

    Parameters
    ----------
    context:
        Layer 0 PipelineContext containing configuration, path, storage,
        validation, metadata, and logging managers.

    config:
        Domain-specific YAML configuration loaded for the current analytics
        domain.

    config_file:
        Config-root-relative file name used to load the domain configuration.

    layer_name:
        Logical layer name for the domain.

    domain_name:
        Human-readable analytics domain name.

    start_time_utc:
        Domain execution start timestamp.

    audit_records:
        In-memory audit records created during execution.

    validation_records:
        In-memory validation records created during execution.

    dataset_records:
        In-memory dataset inventory records created during execution.

    rule_records:
        In-memory rule catalog records created during execution.

    extra_metadata:
        Optional domain-specific runtime metadata.

    Returns
    -------
    AnalyticsDomainRuntime
        Dataclass instance used by analytics domain builders.

    Raises
    ------
    None

    Notes
    -----
    This runtime intentionally stores Analytics Platform domain records only.
    All reusable platform services remain in PipelineContext.
    """

    context: PipelineContext
    config: Dict[str, Any]
    config_file: str
    layer_name: str
    domain_name: str
    start_time_utc: datetime = field(default_factory=utc_now)
    audit_records: List[Dict[str, Any]] = field(default_factory=list)
    validation_records: List[Dict[str, Any]] = field(default_factory=list)
    dataset_records: List[Dict[str, Any]] = field(default_factory=list)
    rule_records: List[Dict[str, Any]] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_logger(self, logger_name: str):
        """
        Purpose
        -------
        Return a configured logger from PipelineContext.

        Parameters
        ----------
        logger_name:
            Logger name requested by the analytics domain.

        Returns
        -------
        logging.Logger
            Configured logger from the Layer 0 LoggingManager.

        Raises
        ------
        None

        Notes
        -----
        Analytics Platform does not create separate logging utilities. Logging
        remains centralized through PipelineContext.
        """
        return self.context.get_logger(logger_name)

    def get_run_id(self) -> str:
        """
        Purpose
        -------
        Return the current pipeline run ID.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Current run ID from PipelineContext.

        Raises
        ------
        None

        Notes
        -----
        This helper keeps domain code concise while preserving the single run ID
        source of truth in PipelineContext.
        """
        return self.context.run_id

    def to_record(self) -> Dict[str, Any]:
        """
        Purpose
        -------
        Convert runtime metadata into a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Runtime metadata suitable for audit or execution summary outputs.

        Raises
        ------
        None

        Notes
        -----
        This method does not include all audit, validation, dataset, or rule
        records. It only describes the runtime itself.
        """
        return {
            "run_id": self.context.run_id,
            "layer_name": self.layer_name,
            "domain_name": self.domain_name,
            "config_file": self.config_file,
            "start_time_utc": self.start_time_utc.isoformat(),
            "environment": self.context.environment,
            "application_version": self.context.application_version,
            "user": self.context.user,
            "extra_metadata": self.extra_metadata,
        }


@dataclass
class AnalyticsBuildResult:
    """
    Purpose
    -------
    Standard result returned by Analytics Platform domain builders.

    Parameters
    ----------
    name:
        Domain or component name.

    status:
        Final execution status.

    message:
        Human-readable execution message.

    row_count:
        Optional total row count produced by the builder.

    column_count:
        Optional total column count produced by the builder.

    Returns
    -------
    AnalyticsBuildResult
        Dataclass instance returned by domain build functions.

    Raises
    ------
    None

    Notes
    -----
    Domain builders should return this object or an object with compatible
    status and message attributes.
    """

    name: str
    status: str
    message: str
    row_count: int = 0
    column_count: int = 0


def normalize_config_file(config_path: str) -> str:
    """
    Purpose
    -------
    Normalize a config path for ConfigurationManager.

    Parameters
    ----------
    config_path:
        Config file path. Supports either:
        - config/analytics_platform/example.yaml
        - analytics_platform/example.yaml

    Returns
    -------
    str
        Config-root-relative path for ConfigurationManager.

    Raises
    ------
    None

    Notes
    -----
    ConfigurationManager uses config_root="config". Domain builders should pass
    config-root-relative paths into load_yaml.
    """
    normalized = str(config_path).strip()

    if normalized.startswith("config/"):
        normalized = normalized[len("config/"):]

    return normalized


def get_domain_config_value(
    config: Dict[str, Any],
    domain_section: str,
    key: str,
    default_value: Any,
) -> Any:
    """
    Purpose
    -------
    Return a value from the top-level domain config section.

    Parameters
    ----------
    config:
        Domain YAML configuration.

    domain_section:
        Top-level domain section name such as population_health,
        predictive_analytics, or value_based_care.

    key:
        Key to retrieve from the domain section.

    default_value:
        Value returned when the key is missing.

    Returns
    -------
    Any
        Configured value or default value.

    Raises
    ------
    None

    Notes
    -----
    This helper avoids repeating nested dictionary access across domain
    builders.
    """
    return config.get(domain_section, {}).get(key, default_value)


def create_domain_runtime(
    context: PipelineContext,
    config: Dict[str, Any],
    config_file: str,
    domain_section: str,
    default_layer_name: str,
    default_domain_name: str,
) -> AnalyticsDomainRuntime:
    """
    Purpose
    -------
    Create an AnalyticsDomainRuntime for a domain builder.

    Parameters
    ----------
    context:
        Layer 0 PipelineContext.

    config:
        Loaded domain YAML configuration.

    config_file:
        Config-root-relative file name.

    domain_section:
        Top-level domain config section.

    default_layer_name:
        Default layer name when configuration does not provide one.

    default_domain_name:
        Default domain name when configuration does not provide one.

    Returns
    -------
    AnalyticsDomainRuntime
        Initialized analytics domain runtime.

    Raises
    ------
    None

    Notes
    -----
    This function only creates the runtime object. Configuration loading and
    section validation should happen before calling it.
    """
    layer_name = get_domain_config_value(
        config=config,
        domain_section=domain_section,
        key="layer_name",
        default_value=default_layer_name,
    )

    domain_name = get_domain_config_value(
        config=config,
        domain_section=domain_section,
        key="domain_name",
        default_value=default_domain_name,
    )

    return AnalyticsDomainRuntime(
        context=context,
        config=config,
        config_file=config_file,
        layer_name=layer_name,
        domain_name=domain_name,
    )