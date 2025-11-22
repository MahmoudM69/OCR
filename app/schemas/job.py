from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an OCR job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreate(BaseModel):
    """Request to create a new OCR job."""

    webhook_url: str | None = Field(
        default=None,
        description="Optional URL to call when processing completes",
    )


class JobResult(BaseModel):
    """Result of OCR processing."""

    text: str
    confidence: float = 0.0
    metadata: dict = Field(default_factory=dict)


class Job(BaseModel):
    """OCR job with status and result."""

    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    filename: str | None = None
    webhook_url: str | None = None
    result: JobResult | None = None
    error: str | None = None


class JobStatusResponse(BaseModel):
    """Response for job status endpoint."""

    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    filename: str | None = None
    result: JobResult | None = None
    error: str | None = None
