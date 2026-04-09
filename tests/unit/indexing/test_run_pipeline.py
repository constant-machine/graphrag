# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import json

import pytest
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.index.run.run_pipeline import _run_pipeline, run_pipeline
from graphrag.index.run.utils import create_run_context
from graphrag.index.typing.pipeline import Pipeline
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.index.typing.workflow import WorkflowFunctionOutput

from tests.unit.config.utils import get_default_graphrag_config


@pytest.mark.asyncio
async def test_run_pipeline_persists_failed_status() -> None:
    async def ok_workflow(_config, _context):
        return WorkflowFunctionOutput(result={"ok": True})

    async def failing_workflow(_config, _context):
        msg = "provider failure"
        raise RuntimeError(msg)

    config = get_default_graphrag_config()
    context = create_run_context()
    pipeline = Pipeline(
        [
            ("load", ok_workflow),
            ("extract_graph", failing_workflow),
        ]
    )

    results = [result async for result in _run_pipeline(pipeline, config, context)]

    assert len(results) == 2
    assert results[-1].workflow == "extract_graph"
    assert isinstance(results[-1].error, RuntimeError)

    stats = json.loads(await context.output_storage.get("stats.json"))
    assert stats["status"] == "failed"
    assert stats["failed_workflow"] == "extract_graph"
    assert stats["error_message"] == "provider failure"
    assert "load" in stats["workflows"]


@pytest.mark.asyncio
async def test_run_pipeline_persists_success_status() -> None:
    async def workflow(_config, _context):
        return WorkflowFunctionOutput(result={"ok": True})

    config = get_default_graphrag_config()
    context = create_run_context()
    pipeline = Pipeline([("extract_graph", workflow)])

    results = [result async for result in _run_pipeline(pipeline, config, context)]

    assert len(results) == 1
    assert results[0].error is None

    stats = json.loads(await context.output_storage.get("stats.json"))
    assert stats["status"] == "success"
    assert stats["failed_workflow"] is None
    assert stats["error_message"] is None


@pytest.mark.asyncio
async def test_run_pipeline_does_not_close_litellm_clients(monkeypatch) -> None:
    async def workflow(_config, _context):
        return WorkflowFunctionOutput(result={"ok": True})

    cleanup_calls: list[str] = []

    async def fake_close() -> None:
        cleanup_calls.append("closed")

    monkeypatch.setattr(
        "graphrag.utils.litellm.close_litellm_async_clients",
        fake_close,
    )

    config = get_default_graphrag_config()
    pipeline = Pipeline([("extract_graph", workflow)])

    results = [
        result
        async for result in run_pipeline(
            pipeline=pipeline,
            config=config,
            callbacks=NoopWorkflowCallbacks(),
        )
    ]

    assert results == [
        PipelineRunResult(
            workflow="extract_graph",
            result={"ok": True},
            state={},
            error=None,
        )
    ]
    assert cleanup_calls == []
