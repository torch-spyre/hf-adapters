"""Failure categories and size limits shared across the weekly-scan pipeline.

Deliberately a leaf module: it imports nothing from this repo and nothing that
pulls in a database driver. That is what lets ``model_prefilter`` — and its unit
tests — be imported on a machine with no ``clickhouse_connect`` installed, while
still sharing one definition of each category string with ``weekly_test``,
``result_sink`` and the CI shard producer (``generate_weekly_shards``).

The category strings are persisted verbatim in the ClickHouse
``failure_category`` column, so changing a value silently splits historical rows
from new ones. Add categories rather than rename them.
"""

# Models above this parameter count are rejected before a worker is spawned:
# they cannot be brought up on Spyre. Producers apply this while building the
# model list; weekly_test keeps an in-worker backstop for rows whose parameter
# count was unknown at fetch time.
MAX_NUMBER_PARAMS = 60_000_000_000

FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER = "not-implemented-adapter"
FAILURE_CATEGORY_MODEL_TOO_LARGE = "model_too_large"
FAILURE_CATEGORY_CPU_LOAD_FAILED = "cpu_load_failed"
FAILURE_CATEGORY_CPU_GENERATE_FAILED = "cpu_generate_failed"
FAILURE_CATEGORY_QUANTIZED_MODEL = "quantized_model"
FAILURE_CATEGORY_HARDWARE_EXCEPTION = "hardware_exception"
FAILURE_CATEGORY_MISFORMED_HF_FAILED = "misformed_hf_failed"
FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION = "test_execution_exception"
FAILURE_CATEGORY_VERIFICATION_FAILED = "verification_failed"
FAILURE_CATEGORY_WORKER_CRASHED = "worker_crashed"
FAILURE_CATEGORY_WORKER_TIMEOUT = "worker_timeout"
FAILURE_CATEGORY_MOE = "moe"
