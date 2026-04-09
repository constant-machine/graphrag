# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import asyncio

import pytest
from graphrag.index.utils.async_tasks import AsyncTaskErrorGroup, gather_with_context


@pytest.mark.asyncio
async def test_gather_with_context_waits_for_all_and_reports_context() -> None:
    state = {"slow_completed": False}

    async def slow_success() -> str:
        await asyncio.sleep(0.01)
        state["slow_completed"] = True
        return "ok"

    async def fast_failure() -> str:
        await asyncio.sleep(0)
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(AsyncTaskErrorGroup, match="item-2"):
        await gather_with_context(
            [slow_success(), fast_failure()],
            contexts=["item-1", "item-2"],
            operation="test operation",
        )

    assert state["slow_completed"] is True
