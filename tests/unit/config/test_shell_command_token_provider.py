# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for ShellCommandTokenProvider."""

import subprocess
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphrag_llm._litellm_auth import ShellCommandTokenProvider


def test_get_token_happy_path():
    """Command returns a token; subsequent call within TTL returns the cached value."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "my-token\n"
        mock_run.return_value = mock_result

        provider = ShellCommandTokenProvider("echo my-token", ttl=3300)
        token1 = provider.get_token()
        token2 = provider.get_token()

        assert token1 == "my-token"
        assert token2 == "my-token"
        # Command only called once — second call uses cache
        mock_run.assert_called_once()


def test_get_token_refreshes_after_ttl():
    """Token is re-fetched after the TTL expires."""
    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.stdout = f"token-{call_count}\n"
        return result

    with patch("subprocess.run", side_effect=fake_run):
        provider = ShellCommandTokenProvider("echo token", ttl=1)
        token1 = provider.get_token()

        # Force TTL expiry by manipulating the internal timestamp
        provider._fetched_at = time.monotonic() - 2  # noqa: SLF001

        token2 = provider.get_token()

    assert token1 == "token-1"
    assert token2 == "token-2"
    assert call_count == 2


def test_invalidate_token_refreshes_cached_token():
    """Invalidating the cached token forces the next call to re-run the command."""
    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.stdout = f"token-{call_count}\n"
        return result

    with patch("subprocess.run", side_effect=fake_run):
        provider = ShellCommandTokenProvider("echo token", ttl=3300)
        token1 = provider.get_token()
        provider.invalidate_token(token1)
        token2 = provider.get_token()

    assert token1 == "token-1"
    assert token2 == "token-2"
    assert call_count == 2


def test_invalidate_token_ignores_stale_expected_token():
    """A stale invalidation request must not clear a newer cached token."""
    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.stdout = f"token-{call_count}\n"
        return result

    with patch("subprocess.run", side_effect=fake_run):
        provider = ShellCommandTokenProvider("echo token", ttl=3300)
        token1 = provider.get_token()
        provider.invalidate_token("older-token")
        token2 = provider.get_token()

    assert token1 == "token-1"
    assert token2 == "token-1"
    assert call_count == 1


def test_invalidate_token_without_expected_token_clears_cache():
    """Invalidating without an expected token always clears the cached token."""
    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.stdout = f"token-{call_count}\n"
        return result

    with patch("subprocess.run", side_effect=fake_run):
        provider = ShellCommandTokenProvider("echo token", ttl=3300)
        token1 = provider.get_token()
        provider.invalidate_token()
        token2 = provider.get_token()

    assert token1 == "token-1"
    assert token2 == "token-2"
    assert call_count == 2


def test_get_token_failed_command_raises():
    """A non-zero exit from the command raises CalledProcessError."""
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "bad-cmd")):
        provider = ShellCommandTokenProvider("bad-cmd")
        with pytest.raises(subprocess.CalledProcessError):
            provider.get_token()


def test_get_token_timeout_raises():
    """A hung command raises TimeoutExpired."""
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired("slow-cmd", timeout=30),
    ):
        provider = ShellCommandTokenProvider("slow-cmd")
        with pytest.raises(subprocess.TimeoutExpired):
            provider.get_token()


def test_get_token_empty_output_raises():
    """Empty stdout from the command raises ValueError."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "   \n"
        mock_run.return_value = mock_result

        provider = ShellCommandTokenProvider("echo ''")
        with pytest.raises(ValueError, match="token_command produced no output"):
            provider.get_token()


@pytest.mark.asyncio
async def test_get_token_async_uses_worker_thread():
    """Async token refresh delegates to asyncio.to_thread."""
    provider = ShellCommandTokenProvider("echo async-token")

    with patch(
        "graphrag_llm._litellm_auth.asyncio.to_thread",
        new=AsyncMock(return_value="async-token"),
    ) as mock_to_thread:
        token = await provider.get_token_async()

    assert token == "async-token"
    mock_to_thread.assert_awaited_once_with(provider.get_token)


def test_get_token_thread_safety():
    """Concurrent callers receive the same cached token without re-running the command."""
    call_count = 0
    barrier = threading.Barrier(10)

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)  # simulate latency
        result = MagicMock()
        result.stdout = "shared-token\n"
        return result

    provider = ShellCommandTokenProvider("echo shared-token", ttl=3300)

    results: list[str] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        token = provider.get_token()
        with lock:
            results.append(token)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    with patch("subprocess.run", side_effect=fake_run):
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert all(r == "shared-token" for r in results)
    # The lock ensures only one fetch happens
    assert call_count == 1
