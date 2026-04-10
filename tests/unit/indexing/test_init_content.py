# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import re
from typing import Any, cast

import yaml
import graphrag.config.defaults as defs
from graphrag.config.init_content import INIT_YAML
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag_llm.config.types import RetryType


def test_init_yaml():
    data = yaml.load(INIT_YAML, Loader=yaml.FullLoader)
    config = GraphRagConfig(**data)
    GraphRagConfig.model_validate(config, strict=True)
    assert config.concurrent_requests == defs.SAFE_CONCURRENT_REQUESTS_WITHOUT_RATE_LIMIT
    completion_retry = config.completion_models[defs.DEFAULT_COMPLETION_MODEL_ID].retry
    embedding_retry = config.embedding_models[defs.DEFAULT_EMBEDDING_MODEL_ID].retry
    assert completion_retry is not None
    assert embedding_retry is not None
    assert completion_retry.type == RetryType.ExponentialBackoff
    assert completion_retry.max_retries == 8
    assert completion_retry.base_delay == 2.0
    assert completion_retry.max_delay == 30
    assert completion_retry.jitter is True
    assert embedding_retry.type == RetryType.ExponentialBackoff
    assert embedding_retry.max_retries == 8
    assert embedding_retry.base_delay == 2.0
    assert embedding_retry.max_delay == 30
    assert embedding_retry.jitter is True


def test_init_yaml_uncommented():
    lines = INIT_YAML.splitlines()
    lines = [line for line in lines if "##" not in line]

    def uncomment_line(line: str) -> str:
        leading_whitespace = cast("Any", re.search(r"^(\s*)", line)).group(1)
        return re.sub(r"^\s*# ", leading_whitespace, line, count=1)

    content = "\n".join([uncomment_line(line) for line in lines])
    data = yaml.load(content, Loader=yaml.FullLoader)
    config = GraphRagConfig(**data)
    GraphRagConfig.model_validate(config, strict=True)
