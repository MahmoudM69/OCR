from fastapi import APIRouter, Depends
from pydantic import BaseModel

import redis
from app.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    redis: str


def get_redis_client() -> redis.Redis:
    """Get Redis client for dependency injection."""
    return redis.Redis.from_url(settings.redis_url)


@router.get("/health", response_model=HealthResponse)
async def health_check(redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Check API and Redis health.

    Returns status of the API and Redis connection.
    """
    try:
        redis_client.ping()
        redis_status = "healthy"
    except redis.ConnectionError:
        redis_status = "unhealthy"

    return HealthResponse(
        status="healthy",
        redis=redis_status,
    )
