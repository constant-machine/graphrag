# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Helpers for awaiting batches of async tasks without early cancellation."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass
from typing import TypeVar, cast

T = TypeVar("T")


@dataclass
class AsyncTaskFailure:
    """Context captured for a failed async task."""

    context: str
    error: Exception
    stack: str


class AsyncTaskErrorGroup(RuntimeError):
    """Raised when one or more async tasks fail after all have settled."""

    def __init__(self, operation: str, errors: list[AsyncTaskFailure]) -> None:
        self.operation = operation
        self.errors = errors

        example = errors[0]
        msg = (
            f"{len(errors)} errors occurred while running {operation}, could not complete! "
            f"Example item: {example.context}. Example error: "
            f"{example.error.__class__.__name__}: {example.error}"
        )
        super().__init__(msg)


async def gather_with_context(
    awaitables: list[asyncio.Future[T] | asyncio.Task[T] | object],
    *,
    contexts: list[str],
    operation: str,
) -> list[T]:
    """Await all tasks and raise a contextual aggregated exception when any fail."""
    results = await asyncio.gather(*awaitables, return_exceptions=True)

    errors: list[AsyncTaskFailure] = []
    values: list[T] = []
    for context, result in zip(contexts, results, strict=True):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            errors.append(
                AsyncTaskFailure(
                    context=context,
                    error=result,
                    stack="".join(
                        traceback.format_exception(
                            type(result),
                            result,
                            result.__traceback__,
                        )
                    ),
                )
            )
            continue

        values.append(cast("T", result))

    if errors:
        raise AsyncTaskErrorGroup(operation, errors)

    return values
