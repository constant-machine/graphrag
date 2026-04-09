# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Internal LiteLLM authentication helpers."""

import asyncio
import subprocess  # noqa: S404
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from graphrag_llm.config.types import AuthMethod

if TYPE_CHECKING:
    from graphrag_llm.config import ModelConfig


_TOKEN_COMMAND_TIMEOUT_SECONDS = 30


class ShellCommandTokenProvider:
    """Runs a shell command to obtain a bearer token, with TTL caching."""

    def __init__(
        self,
        command: str,
        ttl: int = 3300,
        timeout: int = _TOKEN_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self._command = command
        self._ttl = ttl
        self._timeout = timeout
        self._token: str | None = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    def get_token(self) -> str:
        """Return a cached token, refreshing if the TTL has expired."""
        now = time.monotonic()
        if self._token is not None and (now - self._fetched_at) < self._ttl:
            return self._token

        with self._lock:
            now = time.monotonic()
            if self._token is None or (now - self._fetched_at) >= self._ttl:
                self._token = self._fetch()
                self._fetched_at = time.monotonic()
            return self._token

    async def get_token_async(self) -> str:
        """Return a cached token without blocking the event loop on refresh."""
        return await asyncio.to_thread(self.get_token)

    def invalidate_token(self, expected_token: str | None = None) -> None:
        """Clear the cached token if it still matches the failed request token."""
        with self._lock:
            if expected_token is None or self._token == expected_token:
                self._token = None
                self._fetched_at = 0.0

    def _fetch(self) -> str:
        result = subprocess.run(  # noqa: S602
            self._command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=self._timeout,
        )
        token = result.stdout.strip()
        if not token:
            msg = f"token_command produced no output: {self._command!r}"
            raise ValueError(msg)
        return token


def _prepare_request_noop(args: dict[str, Any]) -> dict[str, Any]:
    return args


async def _prepare_request_noop_async(  # noqa: RUF029
    args: dict[str, Any],
) -> dict[str, Any]:
    return args


def _extract_status_code(exception: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exception, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exception, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    return None


def _is_authentication_error(exception: BaseException) -> bool:
    checked: set[int] = set()
    pending: list[BaseException | None] = [exception]
    while pending:
        current = pending.pop()
        if current is None or id(current) in checked:
            continue
        checked.add(id(current))
        if current.__class__.__name__ == "AuthenticationError":
            return True
        if _extract_status_code(current) == 401:
            return True
        pending.extend((current.__cause__, current.__context__))

    return False


@dataclass(slots=True)
class LiteLLMAuthBinding:
    """Resolved authentication behavior for a LiteLLM model."""

    static_args: dict[str, Any]
    prepare_request_sync: Callable[[dict[str, Any]], dict[str, Any]] = (
        _prepare_request_noop
    )
    prepare_request_async_fn: Callable[..., Any] = _prepare_request_noop_async
    refresh_request_sync: (
        Callable[[dict[str, Any], BaseException], dict[str, Any]] | None
    ) = None
    refresh_request_async_fn: Callable[..., Any] | None = None

    def prepare_request(self, args: dict[str, Any]) -> dict[str, Any]:
        """Prepare sync request arguments."""
        return self.prepare_request_sync(args)

    async def prepare_request_async(self, args: dict[str, Any]) -> dict[str, Any]:
        """Prepare async request arguments."""
        return await self.prepare_request_async_fn(args)

    def try_refresh_request(
        self, args: dict[str, Any], exception: BaseException
    ) -> dict[str, Any] | None:
        """Refresh request authentication after an authentication failure."""
        if self.refresh_request_sync is None or not _is_authentication_error(exception):
            return None
        return self.refresh_request_sync(args, exception)

    async def try_refresh_request_async(
        self, args: dict[str, Any], exception: BaseException
    ) -> dict[str, Any] | None:
        """Refresh async request authentication after an authentication failure."""
        if (
            self.refresh_request_async_fn is None
            or not _is_authentication_error(exception)
        ):
            return None
        return await self.refresh_request_async_fn(args, exception)


def build_litellm_auth_binding(
    *,
    model_config: "ModelConfig",
    azure_cognitive_services_audience: str,
) -> LiteLLMAuthBinding:
    """Build the LiteLLM auth binding for a model configuration."""
    if model_config.auth_method == AuthMethod.ApiKey:
        return LiteLLMAuthBinding(static_args={"api_key": model_config.api_key})

    if model_config.auth_method == AuthMethod.AzureManagedIdentity:
        return LiteLLMAuthBinding(
            static_args={
                "azure_ad_token_provider": get_bearer_token_provider(
                    DefaultAzureCredential(), azure_cognitive_services_audience
                )
            }
        )

    if not model_config.token_command:
        msg = "token_command must be set when auth_method=shell_command."
        raise ValueError(msg)

    token_provider = ShellCommandTokenProvider(
        model_config.token_command, model_config.token_ttl
    )

    if model_config.model_provider == "azure":

        def _prepare_request(args: dict[str, Any]) -> dict[str, Any]:
            token = token_provider.get_token()
            args["azure_ad_token"] = token
            args.pop("api_key", None)
            args.pop("azure_ad_token_provider", None)
            return args

        async def _prepare_request_async(args: dict[str, Any]) -> dict[str, Any]:
            token = await token_provider.get_token_async()
            args["azure_ad_token"] = token
            args.pop("api_key", None)
            args.pop("azure_ad_token_provider", None)
            return args

        def _refresh_request(
            args: dict[str, Any], _exception: BaseException
        ) -> dict[str, Any]:
            token_provider.invalidate_token(args.get("azure_ad_token"))
            return _prepare_request(args)

        async def _refresh_request_async(
            args: dict[str, Any], _exception: BaseException
        ) -> dict[str, Any]:
            token_provider.invalidate_token(args.get("azure_ad_token"))
            return await _prepare_request_async(args)

        return LiteLLMAuthBinding(
            static_args={},
            prepare_request_sync=_prepare_request,
            prepare_request_async_fn=_prepare_request_async,
            refresh_request_sync=_refresh_request,
            refresh_request_async_fn=_refresh_request_async,
        )

    def _prepare_request(args: dict[str, Any]) -> dict[str, Any]:
        token = token_provider.get_token()
        args["api_key"] = token
        args.pop("azure_ad_token", None)
        args.pop("azure_ad_token_provider", None)
        return args

    async def _prepare_request_async(args: dict[str, Any]) -> dict[str, Any]:
        token = await token_provider.get_token_async()
        args["api_key"] = token
        args.pop("azure_ad_token", None)
        args.pop("azure_ad_token_provider", None)
        return args

    def _refresh_request(
        args: dict[str, Any], _exception: BaseException
    ) -> dict[str, Any]:
        token_provider.invalidate_token(args.get("api_key"))
        return _prepare_request(args)

    async def _refresh_request_async(
        args: dict[str, Any], _exception: BaseException
    ) -> dict[str, Any]:
        token_provider.invalidate_token(args.get("api_key"))
        return await _prepare_request_async(args)

    return LiteLLMAuthBinding(
        static_args={},
        prepare_request_sync=_prepare_request,
        prepare_request_async_fn=_prepare_request_async,
        refresh_request_sync=_refresh_request,
        refresh_request_async_fn=_refresh_request_async,
    )
