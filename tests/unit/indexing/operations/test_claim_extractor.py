# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

from types import SimpleNamespace

import litellm.exceptions as exceptions
import pytest
from graphrag.index.operations.extract_covariates.claim_extractor import ClaimExtractor


class _FailingModel:
    async def completion_async(self, **_kwargs):
        raise exceptions.BadGatewayError("provider unavailable", "", "")


class _SuccessfulModel:
    async def completion_async(self, **_kwargs):
        return SimpleNamespace(
            content="(alice<|>bob<|>claim<|>true<|>2024<|>2025<|>desc<|>source)"
        )


@pytest.mark.asyncio
async def test_claim_extractor_reraises_provider_errors_without_local_handler() -> None:
    on_error_calls: list[tuple] = []
    extractor = ClaimExtractor(
        model=_FailingModel(),
        extraction_prompt="{input_text}{claim_description}{entity_specs}",
        max_gleanings=0,
        on_error=lambda *args: on_error_calls.append(args),
    )

    with pytest.raises(exceptions.BadGatewayError, match="provider unavailable"):
        await extractor(
            texts=["document"],
            entity_spec={},
            resolved_entities={},
            claim_description="claims",
        )

    assert on_error_calls == []


@pytest.mark.asyncio
async def test_claim_extractor_logs_parsing_failures_and_continues(monkeypatch) -> None:
    on_error_calls: list[tuple] = []
    extractor = ClaimExtractor(
        model=_SuccessfulModel(),
        extraction_prompt="{input_text}{claim_description}{entity_specs}",
        max_gleanings=0,
        on_error=lambda *args: on_error_calls.append(args),
    )

    def fail_clean_claim(_claim, _document_id, _resolved_entities):
        msg = "bad claim payload"
        raise ValueError(msg)

    monkeypatch.setattr(extractor, "_clean_claim", fail_clean_claim)

    result = await extractor(
        texts=["document"],
        entity_spec={},
        resolved_entities={},
        claim_description="claims",
    )

    assert result.output == []
    assert result.source_docs == {}
    assert len(on_error_calls) == 1
    assert isinstance(on_error_calls[0][0], ValueError)
