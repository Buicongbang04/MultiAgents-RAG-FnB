import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class QueueTimeoutError(RuntimeError):
    """Raised when a request waits/runs longer than allowed."""


class QueueExecutionError(RuntimeError):
    """Raised when queued function fails after retries."""

@dataclass
class RetryConfig:
    max_attempts: int
    base_delay_seconds: float


class RequestQueue:
    """
    Async concurrency controller.

    Dùng asyncio.Semaphore để giới hạn số lượng request LLM/retrieval đồng thời.
    Mục tiêu chính: tránh OOM trên RTX 3060 12GB khi nhiều request gọi Generator.
    """

    def __init__(
        self,
        name: str,
        max_concurrency: int,
        timeout_seconds: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self.name = name
        self.max_concurrency = max(1, max_concurrency)
        self.timeout_seconds = timeout_seconds or get_settings().request_timeout_seconds
        self.retry_config = retry_config or RetryConfig(
            max_attempts=get_settings().retry_max_attempts,
            base_delay_seconds=get_settings().retry_base_delay_seconds,
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def failed(self) -> int:
        return self._failed
    
    async def run(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        timeout_seconds: Optional[float] = None,
        retry: bool = True,
        **kwargs: Any,
    ) -> T:
        """
        Run async function under concurrency limit, timeout, and retry.

        FIFO tuyệt đối cần worker queue riêng. Ở MVP, Semaphore đủ để chống OOM.
        Giai đoạn sau có thể thay bằng asyncio.Queue worker pool nếu muốn FIFO nghiêm ngặt.
        """

        timeout = timeout_seconds or self.timeout_seconds
        start = time.perf_counter()

        try:
            async with self._semaphore:
                async with self._lock:
                    self._active += 1

                try:
                    result = await asyncio.wait_for(
                        self._run_with_retry(func, *args, retry=retry, **kwargs),
                        timeout=timeout,
                    )
                    async with self._lock:
                        self._completed += 1
                    return result
                finally:
                    async with self._lock:
                        self._active -= 1

        except asyncio.TimeoutError as exc:
            async with self._lock:
                self._failed += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Queue %s timeout after %.2f ms", self.name, elapsed_ms)
            raise QueueTimeoutError(f"{self.name} request timeout after {timeout}s") from exc

        except Exception as exc:
            async with self._lock:
                self._failed += 1
            logger.exception("Queue %s failed", self.name)
            raise QueueExecutionError(f"{self.name} request failed: {exc}") from exc
        
    async def _run_with_retry(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        retry: bool = True,
        **kwargs: Any,
    ) -> T:
        attempts = self.retry_config.max_attempts if retry else 1
        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break

                delay = self.retry_config.base_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Queue %s attempt %d/%d failed: %s. Retrying in %.2fs",
                    self.name,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise QueueExecutionError(f"Failed after {attempts} attempts: {last_exc}")

    def stats(self) -> dict:
        return {
            "name": self.name,
            "max_concurrency": self.max_concurrency,
            "active": self.active,
            "completed": self.completed,
            "failed": self.failed,
            "timeout_seconds": self.timeout_seconds,
        }


class QueueManager:
    """Central queues for different workloads."""

    def __init__(self) -> None:
        settings = get_settings()
        retry_config = RetryConfig(
            max_attempts=settings.retry_max_attempts,
            base_delay_seconds=settings.retry_base_delay_seconds,
        )

        self.router = RequestQueue(
            name="router",
            max_concurrency=settings.router_max_concurrency,
            retry_config=retry_config,
        )
        self.generator = RequestQueue(
            name="generator",
            max_concurrency=settings.generator_max_concurrency,
            retry_config=retry_config,
        )
        self.embedding = RequestQueue(
            name="embedding",
            max_concurrency=settings.embedding_max_concurrency,
            retry_config=retry_config,
        )
        self.reranker = RequestQueue(
            name="reranker",
            max_concurrency=settings.reranker_max_concurrency,
            retry_config=retry_config,
        )

    def stats(self) -> dict:
        return {
            "router": self.router.stats(),
            "generator": self.generator.stats(),
            "embedding": self.embedding.stats(),
            "reranker": self.reranker.stats(),
        }


queue_manager = QueueManager()