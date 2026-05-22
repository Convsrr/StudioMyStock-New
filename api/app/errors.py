"""Domain errors with mapping to HTTP responses."""
from __future__ import annotations


class StudioError(Exception):
    """Base class for known, user-facing errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(StudioError):
    status_code = 400
    code = "validation_error"


class NotFoundError(StudioError):
    status_code = 404
    code = "not_found"


class UnauthorizedError(StudioError):
    status_code = 401
    code = "unauthorized"


class RateLimitedError(StudioError):
    status_code = 429
    code = "rate_limited"


class PipelineError(StudioError):
    status_code = 502
    code = "pipeline_error"


class ExternalServiceError(StudioError):
    status_code = 502
    code = "external_service_error"
