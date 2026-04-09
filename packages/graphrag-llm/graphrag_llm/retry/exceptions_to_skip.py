# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Retry classification helpers for provider exceptions."""

from __future__ import annotations

from collections.abc import Iterable

_default_exceptions_to_skip = [
    "BadRequestError",
    "UnsupportedParamsError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "ImageFetchError",
    "InvalidRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "APIResponseValidationError",
    "BudgetExceededError",
]

_transient_exception_names = {
    "APIConnectionError",
    "APIError",
    "BadGatewayError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "TimeoutError",
    "APITimeoutError",
    "ConnectTimeout",
    "ReadTimeout",
    "PoolTimeout",
}

_provider_exception_names = set(_default_exceptions_to_skip) | _transient_exception_names


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


def is_provider_exception(exception: BaseException) -> bool:
    """Return True when the exception appears to originate from the provider SDK."""
    if exception.__class__.__name__ in _provider_exception_names:
        return True

    module_name = exception.__class__.__module__
    if module_name.startswith("litellm") or module_name.startswith("openai"):
        return True

    return _extract_status_code(exception) is not None


def _is_permanent_provider_exception(exception: BaseException) -> bool:
    name = exception.__class__.__name__
    if name in _default_exceptions_to_skip:
        return True

    status_code = _extract_status_code(exception)
    if status_code is not None and 400 <= status_code < 500 and status_code not in {
        408,
        429,
    }:
        return True

    return False


def should_skip_retry(
    exception: BaseException,
    *,
    configured_exceptions_to_skip: Iterable[str] | None = None,
) -> bool:
    """Return True when the exception should bypass retry logic."""
    name = exception.__class__.__name__
    configured = set(configured_exceptions_to_skip or _default_exceptions_to_skip)

    if name in configured:
        return True

    status_code = _extract_status_code(exception)
    if status_code is not None:
        if status_code in {408, 429} or 500 <= status_code < 600:
            return False
        if 400 <= status_code < 500:
            return True

    if name in _transient_exception_names:
        return False

    if isinstance(exception, (ConnectionError, TimeoutError)):
        return False

    if _is_permanent_provider_exception(exception):
        return True

    return False
