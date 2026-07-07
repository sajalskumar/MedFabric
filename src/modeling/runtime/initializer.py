###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/modeling/runtime/initializer.py
#
# Capability:
#     Enterprise Modeling Framework
#
# Purpose:
#     Initializes the Modeling Framework runtime context.
#
###############################################################################

from __future__ import annotations

from pathlib import Path

from src.common.parallel_utils import resolve_parallelism_config
from src.common.pipeline_context import create_pipeline_context
from src.modeling.common.audit import add_audit_record
from src.modeling.common.configuration import (
    load_optional_yaml_config,
    load_runtime_modeling_config,
    validate_config,
)
from src.modeling.common.constants import (
    DEFAULT_CAPABILITY_NAME,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_NAME,
    DEFAULT_PIPELINE_CONFIG_PATH,
    STATUS_SUCCESS,
)
from src.modeling.common.logging_utils import configure_logging
from src.modeling.common.output_paths import normalize_path
from src.modeling.common.runtime import ModelingRuntime
from src.modeling.common.timing import utc_now


def initialize_runtime(config_path_raw: str = DEFAULT_CONFIG_PATH) -> ModelingRuntime:
    """
    Initialize Modeling runtime using global PipelineContext.
    """

    project_root = Path.cwd()
    config_path = normalize_path(project_root, config_path_raw)
    pipeline_config_path = normalize_path(project_root, DEFAULT_PIPELINE_CONFIG_PATH)

    pipeline_context = create_pipeline_context(
        pipeline_name="MedFabric Modeling Framework",
    )

    run_id = pipeline_context.run_id

    config = load_runtime_modeling_config(config_path)
    pipeline_config = load_optional_yaml_config(pipeline_config_path)

    validate_config(config)

    modeling_config = config.get("modeling", {})

    capability_name = modeling_config.get(
        "capability_name",
        modeling_config.get("layer_name", DEFAULT_CAPABILITY_NAME),
    )

    layer_name = modeling_config.get("layer_name", capability_name)
    domain_name = modeling_config.get("domain_name", DEFAULT_DOMAIN_NAME)

    logger = configure_logging(
        project_root=project_root,
        config=config,
        run_id=run_id,
    )

    start_time_utc = utc_now()
    event_timestamp_utc = start_time_utc.isoformat()

    parallelism_config = resolve_parallelism_config(
        pipeline_config=pipeline_config,
    )

    runtime = ModelingRuntime(
        run_id=run_id,
        project_root=project_root,
        config_path=config_path,
        pipeline_config_path=pipeline_config_path,
        start_time_utc=start_time_utc,
        event_timestamp_utc=event_timestamp_utc,
        config=config,
        pipeline_config=pipeline_config,
        parallelism_config=parallelism_config,
        logger=logger,
        layer_name=layer_name,
        domain_name=domain_name,
        capability_name=capability_name,
    )

    logger.info("Global pipeline run ID resolved from PipelineContext: %s", run_id)
    logger.info("Runtime event timestamp UTC: %s", event_timestamp_utc)
    logger.info("Pipeline config path: %s", pipeline_config_path)
    logger.info("Parallelism config: %s", parallelism_config)

    add_audit_record(
        runtime=runtime,
        step_name="initialize_runtime",
        status=STATUS_SUCCESS,
        message="Modeling runtime initialized successfully using PipelineContext.",
    )

    return runtime