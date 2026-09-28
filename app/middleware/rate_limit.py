import time
from collections import defaultdict, deque
from typing import Sequence
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_requests: int,
        window_seconds: int,
        exempt_path_prefixes: Sequence[str] = (),
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exempt_path_prefixes = tuple(exempt_path_prefixes)
        self.requests = defaultdict(deque)

    async def dispatch(self, request, call_next):
        if self.exempt_path_prefixes and request.url.path.startswith(self.exempt_path_prefixes):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self.requests[client_ip]

        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "message": "Too many requests. Please try again later.",
                },
            )

        bucket.append(now)
        return await call_next(request)