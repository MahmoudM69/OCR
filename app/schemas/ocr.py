from pydantic import BaseModel, Field


class OCRSubmitResponse(BaseModel):
    """Response after submitting an image for OCR."""

    job_id: str = Field(description="Unique identifier for the job")
    message: str = Field(default="Image submitted for processing")
