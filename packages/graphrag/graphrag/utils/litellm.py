# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Helpers for interacting with LiteLLM process-global state."""

import importlib
import logging

logger = logging.getLogger(__name__)


async def close_litellm_async_clients() -> None:
    """Best-effort cleanup for LiteLLM-managed async clients."""
    try:
        litellm = importlib.import_module("litellm")
    except ImportError:
        return

    close_clients = getattr(litellm, "close_litellm_async_clients", None)
    if close_clients is None:
        return

    try:
        await close_clients()
    except Exception:  # noqa: BLE001
        logger.debug("failed to close LiteLLM async clients", exc_info=True)
