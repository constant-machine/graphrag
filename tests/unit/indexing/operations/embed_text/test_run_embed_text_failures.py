# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import pytest
from graphrag.index.operations.embed_text.run_embed_text import run_embed_text
from graphrag.index.utils.async_tasks import AsyncTaskErrorGroup

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def num_tokens(self, text: str) -> int:
        return len(text)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    async def embedding_async(self, /, **kwargs):
        self.calls += 1
        if self.calls == 2:
            msg = "embedding failure"
            raise RuntimeError(msg)

        class Response:
            embeddings = [[1.0, 2.0]]

        return Response()


@pytest.mark.asyncio
async def test_run_embed_text_raises_contextual_error_for_failed_batch() -> None:
    tokenizer = FakeTokenizer()
    model = FakeEmbeddingModel()

    with pytest.raises(AsyncTaskErrorGroup, match="batch=2/3"):
        await run_embed_text(
            input=["a", "b", "c"],
            callbacks=NoopWorkflowCallbacks(),
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
            batch_max_tokens=100,
            num_threads=3,
        )
