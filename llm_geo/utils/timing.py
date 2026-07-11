"""Timing utilities for node execution."""

from __future__ import annotations

import functools
import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


def time_node(
    coroutine: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """Wrap an async function to log execution time, inputs, and outputs."""
    logger = logging.getLogger(f"{__name__}.{coroutine.__name__}")

    @functools.wraps(coroutine)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        logger.info(f"Starting node: {coroutine.__name__}")
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        all_args = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"Input: ({all_args})")
        try:
            result = await coroutine(*args, **kwargs)
            end = time.time()
            logger.info(
                f"Finished node: {coroutine.__name__} | Duration: {end - start:.4f} seconds"
            )
            logger.debug(f"Output: {result}")
            return result
        except Exception as e:
            traceback.print_exception(e)
            end = time.time()
            logger.error(
                f"Failed node: {coroutine.__name__} | Duration: {end - start:.4f} seconds | Error: {e}"
            )
            raise

    return async_wrapper
