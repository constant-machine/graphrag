# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the internal LiteLLM auth binding."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_llm._litellm_auth import build_litellm_auth_binding
from graphrag_llm.config import AuthMethod, ModelConfig


def test_build_litellm_auth_binding_api_key_uses_static_args():
    model_config = ModelConfig(
        model_provider="openai",
        model="gpt-4o",
        api_key="test-key",
    )

    binding = build_litellm_auth_binding(
        model_config=model_config,
        azure_cognitive_services_audience="https://cognitiveservices.azure.com/.default",
    )

    assert binding.static_args == {"api_key": "test-key"}
    assert binding.prepare_request({"model": "openai/gpt-4o"}) == {
        "model": "openai/gpt-4o"
    }


def test_build_litellm_auth_binding_azure_managed_identity_uses_provider():
    token_provider = MagicMock(return_value="managed-identity-token")

    with (
        patch("graphrag_llm._litellm_auth.DefaultAzureCredential"),
        patch(
            "graphrag_llm._litellm_auth.get_bearer_token_provider",
            return_value=token_provider,
        ) as mock_get_bearer_token_provider,
    ):
        model_config = ModelConfig(
            model_provider="azure",
            model="gpt-4o",
            api_base="https://example.openai.azure.com",
            auth_method=AuthMethod.AzureManagedIdentity,
        )

        binding = build_litellm_auth_binding(
            model_config=model_config,
            azure_cognitive_services_audience="https://cognitiveservices.azure.com/.default",
        )

    mock_get_bearer_token_provider.assert_called_once()
    assert binding.static_args == {"azure_ad_token_provider": token_provider}


def test_build_litellm_auth_binding_shell_command_injects_api_key():
    mock_result = MagicMock()
    mock_result.stdout = "shell-token\n"

    with patch("graphrag_llm._litellm_auth.subprocess.run", return_value=mock_result):
        model_config = ModelConfig(
            model_provider="openai",
            model="gpt-4o",
            auth_method=AuthMethod.ShellCommand,
            token_command="fetch-token",
        )
        binding = build_litellm_auth_binding(
            model_config=model_config,
            azure_cognitive_services_audience="https://cognitiveservices.azure.com/.default",
        )
        args = binding.prepare_request(
            {
                "model": "openai/gpt-4o",
                "azure_ad_token": "stale-token",
                "azure_ad_token_provider": object(),
            }
        )

    assert binding.static_args == {}
    assert args["api_key"] == "shell-token"
    assert "azure_ad_token" not in args
    assert "azure_ad_token_provider" not in args


@pytest.mark.asyncio
async def test_build_litellm_auth_binding_shell_command_injects_azure_token_async():
    model_config = ModelConfig(
        model_provider="azure",
        model="gpt-4o",
        azure_deployment_name="gpt-4o",
        api_base="https://example.openai.azure.com",
        auth_method=AuthMethod.ShellCommand,
        token_command="fetch-token",
    )
    binding = build_litellm_auth_binding(
        model_config=model_config,
        azure_cognitive_services_audience="https://cognitiveservices.azure.com/.default",
    )

    with patch(
        "graphrag_llm._litellm_auth.ShellCommandTokenProvider.get_token_async",
        new=AsyncMock(return_value="azure-shell-token"),
    ):
        args = await binding.prepare_request_async(
            {
                "model": "azure/gpt-4o",
                "api_key": "stale-key",
                "azure_ad_token_provider": object(),
            }
        )

    assert binding.static_args == {}
    assert args["azure_ad_token"] == "azure-shell-token"
    assert "api_key" not in args
    assert "azure_ad_token_provider" not in args
