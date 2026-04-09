# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import logging

import graphrag.config.defaults as defs
from graphrag.index.validate_config import _warn_on_aggressive_concurrency
from graphrag_llm.config import RateLimitConfig

from tests.unit.config.utils import (
    DEFAULT_COMPLETION_MODEL_CONFIG,
    DEFAULT_EMBEDDING_MODEL_CONFIG,
    get_default_graphrag_config,
)


def test_warn_on_aggressive_concurrency_ignores_unused_models(caplog) -> None:
    config = get_default_graphrag_config()
    config.concurrent_requests = defs.SAFE_CONCURRENT_REQUESTS_WITHOUT_RATE_LIMIT + 1
    config.workflows = ["extract_graph"]
    config.completion_models[defs.DEFAULT_COMPLETION_MODEL_ID].rate_limit = RateLimitConfig(
        requests_per_period=60
    )
    config.completion_models["backup_completion_model"] = DEFAULT_COMPLETION_MODEL_CONFIG

    with caplog.at_level(logging.WARNING):
        _warn_on_aggressive_concurrency(config, workflows=config.workflows)

    assert caplog.messages == []


def test_warn_on_aggressive_concurrency_reports_only_active_models(caplog) -> None:
    config = get_default_graphrag_config()
    config.concurrent_requests = defs.SAFE_CONCURRENT_REQUESTS_WITHOUT_RATE_LIMIT + 1
    config.workflows = ["generate_text_embeddings"]
    config.completion_models["backup_completion_model"] = DEFAULT_COMPLETION_MODEL_CONFIG
    config.embedding_models["backup_embedding_model"] = DEFAULT_EMBEDDING_MODEL_CONFIG

    with caplog.at_level(logging.WARNING):
        _warn_on_aggressive_concurrency(config, workflows=config.workflows)

    assert len(caplog.messages) == 1
    assert defs.DEFAULT_EMBEDDING_MODEL_ID in caplog.messages[0]
    assert "backup_completion_model" not in caplog.messages[0]
    assert "backup_embedding_model" not in caplog.messages[0]
