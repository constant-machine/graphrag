# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for LiteLLM auth wiring in completion and embedding builders."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphrag_llm.completion.lite_llm_completion import _create_base_completions
from graphrag_llm.config import AuthMethod, ModelConfig
from graphrag_llm.embedding.lite_llm_embedding import _create_base_embeddings
from litellm import ModelResponse


@dataclass(frozen=True)
class BuilderCase:
    name: str
    create_base: Any
    sync_patch: str
    async_patch: str
    request_kwargs: dict[str, Any]
    response_factory: Any


def _completion_response() -> ModelResponse:
    return ModelResponse(
        id="1",
        created=0,
        model="openai/gpt-4o",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    )


def _embedding_response() -> MagicMock:
    response = MagicMock()
    response.model_dump.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-3-large",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    return response


BUILDERS = [
    BuilderCase(
        name="completion",
        create_base=_create_base_completions,
        sync_patch="graphrag_llm.completion.lite_llm_completion.litellm.completion",
        async_patch="graphrag_llm.completion.lite_llm_completion.litellm.acompletion",
        request_kwargs={"messages": "hello"},
        response_factory=_completion_response,
    ),
    BuilderCase(
        name="embedding",
        create_base=_create_base_embeddings,
        sync_patch="graphrag_llm.embedding.lite_llm_embedding.litellm.embedding",
        async_patch="graphrag_llm.embedding.lite_llm_embedding.litellm.aembedding",
        request_kwargs={"input": ["hello"]},
        response_factory=_embedding_response,
    ),
]


class AuthenticationError(Exception):
    """Provider-like authentication error for auth refresh tests."""


class StatusCodeError(Exception):
    """Provider-like exception with an HTTP status code."""

    def __init__(self, status_code: int):
        super().__init__(f"status code {status_code}")
        self.status_code = status_code


def _token_result(token: str) -> MagicMock:
    result = MagicMock()
    result.stdout = f"{token}\n"
    return result


def _make_model_config(
    *,
    model_provider: str,
    auth_method: AuthMethod,
    api_key: str | None = None,
    token_command: str | None = None,
) -> ModelConfig:
    kwargs: dict[str, Any] = {
        "model_provider": model_provider,
        "model": "gpt-4o" if model_provider == "azure" else "text-embedding-3-large",
        "auth_method": auth_method,
    }
    if model_provider == "azure":
        kwargs["model"] = "gpt-4o"
        kwargs["azure_deployment_name"] = "test-deployment"
        kwargs["api_base"] = "https://example.openai.azure.com"
    if api_key is not None:
        kwargs["api_key"] = api_key
    if token_command is not None:
        kwargs["token_command"] = token_command
    return ModelConfig(**kwargs)


def _create_functions(builder: BuilderCase, model_config: ModelConfig):
    return builder.create_base(
        model_config=model_config,
        drop_unsupported_params=True,
        azure_cognitive_services_audience="https://cognitiveservices.azure.com/.default",
    )


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.name)
def test_litellm_builder_wires_api_key_auth(builder: BuilderCase):
    model_config = _make_model_config(
        model_provider="openai",
        auth_method=AuthMethod.ApiKey,
        api_key="test-key",
    )
    sync_fn, _ = _create_functions(builder, model_config)

    with patch(builder.sync_patch, return_value=builder.response_factory()) as mock_call:
        sync_fn(**builder.request_kwargs)

    kwargs = mock_call.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert "azure_ad_token" not in kwargs
    assert "azure_ad_token_provider" not in kwargs


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.name)
def test_litellm_builder_wires_azure_managed_identity_auth(builder: BuilderCase):
    token_provider = MagicMock(return_value="managed-identity-token")
    model_config = _make_model_config(
        model_provider="azure",
        auth_method=AuthMethod.AzureManagedIdentity,
    )

    with (
        patch("graphrag_llm._litellm_auth.DefaultAzureCredential"),
        patch(
            "graphrag_llm._litellm_auth.get_bearer_token_provider",
            return_value=token_provider,
        ),
    ):
        sync_fn, _ = _create_functions(builder, model_config)

    with patch(builder.sync_patch, return_value=builder.response_factory()) as mock_call:
        sync_fn(**builder.request_kwargs)

    kwargs = mock_call.call_args.kwargs
    assert kwargs["azure_ad_token_provider"] is token_provider
    assert "api_key" not in kwargs or kwargs["api_key"] is None
    assert "azure_ad_token" not in kwargs


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.name)
def test_litellm_builder_wires_shell_command_api_key_auth(builder: BuilderCase):
    mock_result = MagicMock()
    mock_result.stdout = "shell-token\n"
    model_config = _make_model_config(
        model_provider="openai",
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    sync_fn, _ = _create_functions(builder, model_config)

    with (
        patch("graphrag_llm._litellm_auth.subprocess.run", return_value=mock_result),
        patch(builder.sync_patch, return_value=builder.response_factory()) as mock_call,
    ):
        sync_fn(**builder.request_kwargs)

    kwargs = mock_call.call_args.kwargs
    assert kwargs["api_key"] == "shell-token"
    assert "azure_ad_token" not in kwargs
    assert "azure_ad_token_provider" not in kwargs


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.name)
def test_litellm_builder_wires_shell_command_azure_auth(builder: BuilderCase):
    mock_result = MagicMock()
    mock_result.stdout = "azure-shell-token\n"
    model_config = _make_model_config(
        model_provider="azure",
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    sync_fn, _ = _create_functions(builder, model_config)

    with (
        patch("graphrag_llm._litellm_auth.subprocess.run", return_value=mock_result),
        patch(builder.sync_patch, return_value=builder.response_factory()) as mock_call,
    ):
        sync_fn(**builder.request_kwargs)

    kwargs = mock_call.call_args.kwargs
    assert kwargs["azure_ad_token"] == "azure-shell-token"
    assert "api_key" not in kwargs or kwargs["api_key"] is None
    assert "azure_ad_token_provider" not in kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("builder", "model_provider", "expected_auth_key"),
    [
        (BUILDERS[0], "openai", "api_key"),
        (BUILDERS[0], "azure", "azure_ad_token"),
        (BUILDERS[1], "openai", "api_key"),
        (BUILDERS[1], "azure", "azure_ad_token"),
    ],
    ids=[
        "completion-openai",
        "completion-azure",
        "embedding-openai",
        "embedding-azure",
    ],
)
async def test_litellm_builder_wires_shell_command_auth_async(
    builder: BuilderCase,
    model_provider: str,
    expected_auth_key: str,
):
    model_config = _make_model_config(
        model_provider=model_provider,
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    _, async_fn = _create_functions(builder, model_config)

    with (
        patch(
            "graphrag_llm._litellm_auth.ShellCommandTokenProvider.get_token_async",
            new=AsyncMock(return_value="async-shell-token"),
        ),
        patch(builder.async_patch, new=AsyncMock(return_value=builder.response_factory())) as mock_call,
    ):
        await async_fn(**builder.request_kwargs)

    kwargs = mock_call.call_args.kwargs
    assert kwargs[expected_auth_key] == "async-shell-token"
    if expected_auth_key == "api_key":
        assert "azure_ad_token" not in kwargs
    else:
        assert "api_key" not in kwargs or kwargs["api_key"] is None
    assert "azure_ad_token_provider" not in kwargs


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.name)
@pytest.mark.parametrize(
    ("model_provider", "expected_auth_key"),
    [("openai", "api_key"), ("azure", "azure_ad_token")],
)
def test_shell_command_auth_refreshes_token_once_after_auth_error(
    builder: BuilderCase,
    model_provider: str,
    expected_auth_key: str,
):
    model_config = _make_model_config(
        model_provider=model_provider,
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    sync_fn, _ = _create_functions(builder, model_config)

    with (
        patch(
            "graphrag_llm._litellm_auth.subprocess.run",
            side_effect=[_token_result("stale-token"), _token_result("fresh-token")],
        ) as mock_run,
        patch(
            builder.sync_patch,
            side_effect=[AuthenticationError("expired"), builder.response_factory()],
        ) as mock_call,
    ):
        sync_fn(**builder.request_kwargs)

    assert mock_run.call_count == 2
    assert mock_call.call_count == 2
    assert mock_call.call_args_list[0].kwargs[expected_auth_key] == "stale-token"
    assert mock_call.call_args_list[1].kwargs[expected_auth_key] == "fresh-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.name)
async def test_shell_command_auth_refreshes_token_once_after_async_401(
    builder: BuilderCase,
):
    model_config = _make_model_config(
        model_provider="openai",
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    _, async_fn = _create_functions(builder, model_config)

    with (
        patch(
            "graphrag_llm._litellm_auth.ShellCommandTokenProvider.get_token_async",
            new=AsyncMock(side_effect=["stale-token", "fresh-token"]),
        ) as mock_get_token,
        patch(
            builder.async_patch,
            new=AsyncMock(
                side_effect=[StatusCodeError(401), builder.response_factory()]
            ),
        ) as mock_call,
    ):
        await async_fn(**builder.request_kwargs)

    assert mock_get_token.await_count == 2
    assert mock_call.await_count == 2
    assert mock_call.call_args_list[0].kwargs["api_key"] == "stale-token"
    assert mock_call.call_args_list[1].kwargs["api_key"] == "fresh-token"


def test_shell_command_auth_does_not_refresh_non_auth_error():
    builder = BUILDERS[0]
    model_config = _make_model_config(
        model_provider="openai",
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    sync_fn, _ = _create_functions(builder, model_config)

    with (
        patch(
            "graphrag_llm._litellm_auth.subprocess.run",
            return_value=_token_result("shell-token"),
        ) as mock_run,
        patch(builder.sync_patch, side_effect=RuntimeError("boom")) as mock_call,
        pytest.raises(RuntimeError, match="boom"),
    ):
        sync_fn(**builder.request_kwargs)

    mock_run.assert_called_once()
    mock_call.assert_called_once()


def test_shell_command_auth_reraises_second_auth_error():
    builder = BUILDERS[0]
    model_config = _make_model_config(
        model_provider="openai",
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    sync_fn, _ = _create_functions(builder, model_config)

    with (
        patch(
            "graphrag_llm._litellm_auth.subprocess.run",
            side_effect=[_token_result("stale-token"), _token_result("fresh-token")],
        ) as mock_run,
        patch(
            builder.sync_patch,
            side_effect=[
                AuthenticationError("expired"),
                AuthenticationError("still expired"),
            ],
        ) as mock_call,
        pytest.raises(AuthenticationError, match="still expired"),
    ):
        sync_fn(**builder.request_kwargs)

    assert mock_run.call_count == 2
    assert mock_call.call_count == 2
