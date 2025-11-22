from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about an OCR model."""

    name: str
    is_loaded: bool = False
    is_available: bool = True


class ModelsListResponse(BaseModel):
    """Response listing available models."""

    models: list[ModelInfo]
    current_model: str | None = None


class CurrentModelResponse(BaseModel):
    """Response for current model endpoint."""

    model: str | None = Field(
        description="Currently loaded model name, or null if none loaded"
    )
    is_processing: bool = Field(
        description="Whether the model is currently processing an image"
    )


class SwitchModelRequest(BaseModel):
    """Request to switch to a different model."""

    model: str = Field(description="Name of the model to switch to")


class SwitchModelResponse(BaseModel):
    """Response after switching models."""

    previous_model: str | None
    current_model: str
    message: str
