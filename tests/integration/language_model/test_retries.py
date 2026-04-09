# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Test LiteLLM Retries."""

import time
from typing import Any

import httpx
import litellm.exceptions as exceptions
import pytest
from graphrag_llm.config import RetryConfig, RetryType
from graphrag_llm.retry import create_retry


class UnknownLiteLLMError(Exception):
    """Provider exception class without explicit retry classification."""


UnknownLiteLLMError.__module__ = "litellm.tests"


class UnknownLiteLLMStatusError(Exception):
    """Provider exception class with an attached HTTP status code."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


UnknownLiteLLMStatusError.__module__ = "litellm.tests"


@pytest.mark.parametrize(
    ("config", "max_retries", "expected_time"),
    [
        (
            RetryConfig(
                type=RetryType.ExponentialBackoff,
                max_retries=3,
                base_delay=2.0,
                jitter=False,
            ),
            3,
            2 + 4 + 8,  # No jitter, so exact times
        ),
        (
            RetryConfig(
                type=RetryType.Immediate,
                max_retries=3,
            ),
            3,
            0,  # Immediate retry, so no delay
        ),
    ],
)
def test_retries(config: RetryConfig, max_retries: int, expected_time: float) -> None:
    """
    Test various retry strategies with various configurations.
    """
    retry_service = create_retry(config)

    # start at -1 because the first call is not a retry
    retries = -1

    def mock_func():
        nonlocal retries
        retries += 1
        msg = "Mock error for testing retries"
        raise ValueError(msg)

    start_time = time.time()
    with pytest.raises(ValueError, match="Mock error for testing retries"):
        retry_service.retry(func=mock_func, input_args={})
    elapsed_time = time.time() - start_time

    assert retries == max_retries, f"Expected {max_retries} retries, got {retries}"
    assert elapsed_time >= expected_time, (
        f"Expected elapsed time >= {expected_time}, got {elapsed_time}"
    )


@pytest.mark.parametrize(
    ("config", "max_retries", "expected_time"),
    [
        (
            RetryConfig(
                type=RetryType.ExponentialBackoff,
                max_retries=3,
                base_delay=2.0,
                jitter=False,
            ),
            3,
            2 + 4 + 8,  # No jitter, so exact times
        ),
        (
            RetryConfig(
                type=RetryType.Immediate,
                max_retries=3,
            ),
            3,
            0,  # Immediate retry, so no delay
        ),
    ],
)
async def test_retries_async(
    config: RetryConfig, max_retries: int, expected_time: float
) -> None:
    """
    Test various retry strategies with various configurations.
    """
    retry_service = create_retry(config)

    # start at -1 because the first call is not a retry
    retries = -1

    def mock_func():
        nonlocal retries
        retries += 1
        msg = "Mock error for testing retries"
        raise ValueError(msg)

    start_time = time.time()
    with pytest.raises(ValueError, match="Mock error for testing retries"):
        await retry_service.retry_async(func=mock_func, input_args={})
    elapsed_time = time.time() - start_time

    assert retries == max_retries, f"Expected {max_retries} retries, got {retries}"
    assert elapsed_time >= expected_time, (
        f"Expected elapsed time >= {expected_time}, got {elapsed_time}"
    )


@pytest.mark.parametrize(
    "config",
    [
        (
            RetryConfig(
                type=RetryType.ExponentialBackoff,
                max_retries=3,
                base_delay=2.0,
                jitter=False,
            )
        ),
        (
            RetryConfig(
                type=RetryType.Immediate,
                max_retries=3,
            )
        ),
    ],
)
@pytest.mark.parametrize(
    ("exception", "exception_args"),
    [
        (
            "BadRequestError",
            ["Oh no!", "", ""],
        ),
        (
            "UnsupportedParamsError",
            ["Oh no!", "", ""],
        ),
        (
            "ContextWindowExceededError",
            ["Oh no!", "", ""],
        ),
        (
            "ContentPolicyViolationError",
            ["Oh no!", "", ""],
        ),
        (
            "ImageFetchError",
            ["Oh no!", "", ""],
        ),
        (
            "InvalidRequestError",
            ["Oh no!", "", ""],
        ),
        (
            "AuthenticationError",
            ["Oh no!", "", ""],
        ),
        (
            "PermissionDeniedError",
            [
                "Oh no!",
                "",
                "",
                httpx.Response(
                    status_code=403,
                    request=httpx.Request(
                        method="GET", url="https://litellm.ai"
                    ),  # mock request object
                ),
            ],
        ),
        (
            "NotFoundError",
            ["Oh no!", "", ""],
        ),
        (
            "UnprocessableEntityError",
            [
                "Oh no!",
                "",
                "",
                httpx.Response(
                    status_code=403,
                    request=httpx.Request(
                        method="GET", url="https://litellm.ai"
                    ),  # mock request object
                ),
            ],
        ),
        (
            "APIResponseValidationError",
            ["Oh no!", "", ""],
        ),
        (
            "BudgetExceededError",
            ["Oh no!", "", ""],
        ),
    ],
)
def test_exponential_backoff_skipping_exceptions(
    config: RetryConfig, exception: str, exception_args: list[Any]
) -> None:
    """
    Test skipping retries for exceptions that should not cause a retry.
    """
    retry_service = create_retry(config)

    # start at -1 because the first call is not a retry
    retries = -1
    exception_cls = exceptions.__dict__[exception]

    def mock_func():
        nonlocal retries
        retries += 1
        raise exception_cls(*exception_args)

    with pytest.raises(exception_cls, match="Oh no!"):
        retry_service.retry(func=mock_func, input_args={})

    # subtract 1 from retries because the first call is not a retry
    assert retries == 0, (
        f"Expected not to retry for '{exception}' exception. Got {retries} retries."
    )


@pytest.mark.parametrize(
    "config",
    [
        (
            RetryConfig(
                type=RetryType.ExponentialBackoff,
                max_retries=3,
                base_delay=2.0,
                jitter=False,
            )
        ),
        (
            RetryConfig(
                type=RetryType.Immediate,
                max_retries=3,
            )
        ),
    ],
)
@pytest.mark.parametrize(
    ("exception", "exception_args"),
    [
        (
            "BadGatewayError",
            ["Oh no!", "", ""],
        ),
        (
            "RateLimitError",
            ["Oh no!", "", ""],
        ),
        (
            "APIConnectionError",
            ["Oh no!", "", ""],
        ),
        (
            "ServiceUnavailableError",
            ["Oh no!", "", ""],
        ),
        (
            "APIError",
            [500, "Oh no!", "", ""],
        ),
    ],
)
def test_transient_provider_exceptions_are_retried(
    config: RetryConfig, exception: str, exception_args: list[Any]
) -> None:
    """Transient provider exceptions should consume the full retry budget."""
    retry_service = create_retry(config)

    retries = -1
    exception_cls = exceptions.__dict__[exception]

    def mock_func():
        nonlocal retries
        retries += 1
        raise exception_cls(*exception_args)

    with pytest.raises(exception_cls, match="Oh no!"):
        retry_service.retry(func=mock_func, input_args={})

    assert retries == 3, (
        f"Expected transient exception '{exception}' to retry 3 times. Got {retries} retries."
    )


@pytest.mark.parametrize(
    "config",
    [
        RetryConfig(
            type=RetryType.ExponentialBackoff,
            max_retries=3,
            base_delay=2.0,
            jitter=False,
        ),
        RetryConfig(
            type=RetryType.Immediate,
            max_retries=3,
        ),
    ],
)
def test_unknown_provider_exceptions_retry_by_default(config: RetryConfig) -> None:
    retry_service = create_retry(config)
    retries = -1

    def mock_func():
        nonlocal retries
        retries += 1
        raise UnknownLiteLLMError("Oh no!")

    with pytest.raises(UnknownLiteLLMError, match="Oh no!"):
        retry_service.retry(func=mock_func, input_args={})

    assert retries == 3


@pytest.mark.parametrize(
    ("status_code", "expected_retries"),
    [
        (400, 0),
        (408, 3),
        (429, 3),
        (500, 3),
    ],
)
def test_unknown_provider_exceptions_follow_status_code_classification(
    status_code: int, expected_retries: int
) -> None:
    retry_service = create_retry(
        RetryConfig(
            type=RetryType.Immediate,
            max_retries=3,
        )
    )
    retries = -1

    def mock_func():
        nonlocal retries
        retries += 1
        raise UnknownLiteLLMStatusError("Oh no!", status_code)

    with pytest.raises(UnknownLiteLLMStatusError, match="Oh no!"):
        retry_service.retry(func=mock_func, input_args={})

    assert retries == expected_retries
