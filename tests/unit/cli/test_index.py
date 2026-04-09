# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import pytest
from graphrag.cli import index as index_cli
from graphrag.config.enums import IndexingMethod
from graphrag.index.typing.pipeline_run_result import PipelineRunResult

from tests.unit.config.utils import get_default_graphrag_config


@pytest.mark.asyncio
async def test_build_index_with_cleanup_closes_clients_on_success(monkeypatch) -> None:
    outputs = [
        PipelineRunResult(
            workflow="extract_graph",
            result={"ok": True},
            state={},
            error=None,
        )
    ]
    cleanup_calls: list[str] = []

    async def fake_build_index(**_kwargs):
        return outputs

    async def fake_close() -> None:
        cleanup_calls.append("closed")

    monkeypatch.setattr(index_cli.api, "build_index", fake_build_index)
    monkeypatch.setattr(index_cli, "close_litellm_async_clients", fake_close)

    result = await index_cli._build_index_with_cleanup(
        config=get_default_graphrag_config(),
        method=IndexingMethod.Standard,
        is_update_run=False,
        verbose=False,
        callbacks=[],
    )

    assert result == outputs
    assert cleanup_calls == ["closed"]


@pytest.mark.asyncio
async def test_build_index_with_cleanup_closes_clients_on_failure(monkeypatch) -> None:
    cleanup_calls: list[str] = []

    async def fake_build_index(**_kwargs):
        msg = "provider failure"
        raise RuntimeError(msg)

    async def fake_close() -> None:
        cleanup_calls.append("closed")

    monkeypatch.setattr(index_cli.api, "build_index", fake_build_index)
    monkeypatch.setattr(index_cli, "close_litellm_async_clients", fake_close)

    with pytest.raises(RuntimeError, match="provider failure"):
        await index_cli._build_index_with_cleanup(
            config=get_default_graphrag_config(),
            method=IndexingMethod.Standard,
            is_update_run=False,
            verbose=False,
            callbacks=[],
        )

    assert cleanup_calls == ["closed"]
