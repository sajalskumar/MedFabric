###############################################################################
# MedFabric
# Enterprise Healthcare Data & AI Platform
#
# File:
#     src/common/parallel_utils.py
#
# Layer:
#     Shared Common Utilities
#
# Purpose:
#     Provides project-wide parallel execution helpers for MedFabric.
#
# Business Context:
#     Several MedFabric layers may need controlled parallel execution:
#
#       - Feature Store feature group builds
#       - Modeling candidate algorithm training
#       - Metadata generation
#       - Validation jobs
#       - Future reporting and insight generation tasks
#
#     Parallelism must be controlled from the project pipeline configuration,
#     not hardcoded inside individual modules.
#
# Configuration Source:
#     config/pipeline.yaml
#
# Expected Configuration:
#
#     performance:
#
#       parallel_execution: true
#
#       max_parallel_workers: 4
#
#       parallel_strategy: "thread"
#
# Architectural Rules:
#     - Individual modules must not hardcode worker counts.
#     - Individual modules must not independently decide project-wide runtime
#       behavior.
#     - Parallelism must be optional and safe to disable.
#     - Sequential execution must remain the default fallback.
#     - This utility should be reusable across all MedFabric layers.
#
# Current Scope:
#     - Thread-based parallel task execution.
#     - Sequential fallback.
#     - Ordered result collection.
#     - Safe exception capture.
#
# Future Scope:
#     - Process-based execution.
#     - Async execution.
#     - Per-layer worker caps.
#     - Retry support.
#     - Timeout support.
#
# Run:
#     python -m src.common.parallel_utils
#
###############################################################################

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


###############################################################################
# Constants
###############################################################################

DEFAULT_PARALLEL_EXECUTION = False
DEFAULT_MAX_PARALLEL_WORKERS = 1
DEFAULT_PARALLEL_STRATEGY = "thread"

SUPPORTED_PARALLEL_STRATEGIES = {
    "thread",
    "sequential",
}

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"


###############################################################################
# Result Object
###############################################################################

@dataclass
class ParallelTaskResult:
    """
    Standard result object for one parallel task.

    Attributes:
        task_name:
            Human-readable task name.

        status:
            SUCCESS or FAILED.

        result:
            Return value from the task when successful.

        error_message:
            Error message when failed.

        original_index:
            Input order index. Used to restore deterministic result order.
    """

    task_name: str
    status: str
    result: Any = None
    error_message: Optional[str] = None
    original_index: int = 0


###############################################################################
# Configuration Helpers
###############################################################################

def resolve_parallelism_config(
    pipeline_config: Optional[Dict[str, Any]] = None,
    performance_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve project-wide parallelism configuration.

    Preferred input:
        pipeline_config["performance"]

    Also supported:
        performance_config

    This allows callers to pass either the full pipeline config or the already
    extracted performance section.

    Returns:
        Normalized parallelism configuration.
    """

    if performance_config is None:
        performance_config = {}

    if pipeline_config:
        performance_config = pipeline_config.get("performance", performance_config)

    parallel_execution = bool(
        performance_config.get(
            "parallel_execution",
            DEFAULT_PARALLEL_EXECUTION,
        )
    )

    max_parallel_workers = int(
        performance_config.get(
            "max_parallel_workers",
            DEFAULT_MAX_PARALLEL_WORKERS,
        )
    )

    parallel_strategy = str(
        performance_config.get(
            "parallel_strategy",
            DEFAULT_PARALLEL_STRATEGY,
        )
    ).lower()

    if max_parallel_workers < 1:
        max_parallel_workers = 1

    if parallel_strategy not in SUPPORTED_PARALLEL_STRATEGIES:
        raise ValueError(
            f"Unsupported parallel_strategy: {parallel_strategy}. "
            f"Supported values: {sorted(SUPPORTED_PARALLEL_STRATEGIES)}"
        )

    if not parallel_execution:
        parallel_strategy = "sequential"
        max_parallel_workers = 1

    return {
        "parallel_execution": parallel_execution,
        "max_parallel_workers": max_parallel_workers,
        "parallel_strategy": parallel_strategy,
    }


def is_parallel_enabled(parallelism_config: Dict[str, Any]) -> bool:
    """
    Return whether parallel execution is enabled.
    """

    return (
        bool(parallelism_config.get("parallel_execution", False))
        and int(parallelism_config.get("max_parallel_workers", 1)) > 1
        and str(parallelism_config.get("parallel_strategy", "sequential")).lower()
        == "thread"
    )


###############################################################################
# Task Execution Helpers
###############################################################################

def execute_single_task(
    task_name: str,
    task_callable: Callable[[], Any],
    original_index: int,
) -> ParallelTaskResult:
    """
    Execute one task and capture exceptions safely.
    """

    try:
        result = task_callable()

        return ParallelTaskResult(
            task_name=task_name,
            status=STATUS_SUCCESS,
            result=result,
            error_message=None,
            original_index=original_index,
        )

    except Exception as exc:
        return ParallelTaskResult(
            task_name=task_name,
            status=STATUS_FAILED,
            result=None,
            error_message=str(exc),
            original_index=original_index,
        )


def run_tasks_sequentially(
    tasks: Iterable[Dict[str, Any]],
) -> List[ParallelTaskResult]:
    """
    Execute tasks sequentially.

    Expected task format:
        {
            "task_name": "name",
            "callable": callable_without_arguments
        }
    """

    results: List[ParallelTaskResult] = []

    for index, task in enumerate(tasks):
        task_name = str(task.get("task_name", f"task_{index}"))
        task_callable = task.get("callable")

        if not callable(task_callable):
            results.append(
                ParallelTaskResult(
                    task_name=task_name,
                    status=STATUS_FAILED,
                    result=None,
                    error_message="Task callable is missing or not callable.",
                    original_index=index,
                )
            )
            continue

        results.append(
            execute_single_task(
                task_name=task_name,
                task_callable=task_callable,
                original_index=index,
            )
        )

    return results


def run_tasks_in_threads(
    tasks: Iterable[Dict[str, Any]],
    max_workers: int,
) -> List[ParallelTaskResult]:
    """
    Execute tasks using ThreadPoolExecutor.

    Notes:
        - Results are returned in original input order.
        - Exceptions are captured inside ParallelTaskResult.
        - This helper does not raise task exceptions directly.
    """

    task_list = list(tasks)

    results: List[ParallelTaskResult] = []

    future_map: Dict[Future, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, task in enumerate(task_list):
            task_name = str(task.get("task_name", f"task_{index}"))
            task_callable = task.get("callable")

            if not callable(task_callable):
                results.append(
                    ParallelTaskResult(
                        task_name=task_name,
                        status=STATUS_FAILED,
                        result=None,
                        error_message="Task callable is missing or not callable.",
                        original_index=index,
                    )
                )
                continue

            future = executor.submit(
                execute_single_task,
                task_name,
                task_callable,
                index,
            )

            future_map[future] = task_name

        for future in as_completed(future_map):
            results.append(future.result())

    return sorted(results, key=lambda item: item.original_index)


def run_tasks(
    tasks: Iterable[Dict[str, Any]],
    parallelism_config: Optional[Dict[str, Any]] = None,
) -> List[ParallelTaskResult]:
    """
    Run tasks using project-wide parallelism configuration.

    This is the main public function other MedFabric modules should use.

    Args:
        tasks:
            Iterable of task dictionaries.

            Each task must contain:
                task_name
                callable

        parallelism_config:
            Normalized config from resolve_parallelism_config().

    Returns:
        List of ParallelTaskResult objects.
    """

    if parallelism_config is None:
        parallelism_config = resolve_parallelism_config()

    if not is_parallel_enabled(parallelism_config):
        return run_tasks_sequentially(tasks)

    max_workers = int(parallelism_config.get("max_parallel_workers", 1))

    return run_tasks_in_threads(
        tasks=tasks,
        max_workers=max_workers,
    )


###############################################################################
# Failure Helpers
###############################################################################

def get_failed_task_results(
    results: Iterable[ParallelTaskResult],
) -> List[ParallelTaskResult]:
    """
    Return failed task results.
    """

    return [result for result in results if result.status == STATUS_FAILED]


def raise_if_any_task_failed(
    results: Iterable[ParallelTaskResult],
) -> None:
    """
    Raise a combined exception if any task failed.
    """

    failed_results = get_failed_task_results(results)

    if not failed_results:
        return

    error_lines = []

    for result in failed_results:
        error_lines.append(
            f"{result.task_name}: {result.error_message}"
        )

    raise RuntimeError(
        "One or more parallel tasks failed:\n" + "\n".join(error_lines)
    )


###############################################################################
# Standalone Validation
###############################################################################

def main() -> None:
    """
    Validate parallel utility independently.
    """

    def task_one() -> str:
        return "task one complete"

    def task_two() -> str:
        return "task two complete"

    tasks = [
        {
            "task_name": "task_one",
            "callable": task_one,
        },
        {
            "task_name": "task_two",
            "callable": task_two,
        },
    ]

    parallelism_config = resolve_parallelism_config(
        performance_config={
            "parallel_execution": True,
            "max_parallel_workers": 2,
            "parallel_strategy": "thread",
        }
    )

    results = run_tasks(
        tasks=tasks,
        parallelism_config=parallelism_config,
    )

    raise_if_any_task_failed(results)

    print("Parallel utility validation successful.")

    for result in results:
        print(
            f"{result.task_name} | "
            f"{result.status} | "
            f"{result.result}"
        )


if __name__ == "__main__":
    main()