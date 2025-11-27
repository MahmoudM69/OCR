"""
Custom RQ worker entry point with model pre-loading.

This script initializes model downloads in the background
before starting the RQ worker.
"""
from redis import Redis
from rq import Worker

from app.config import settings
from app.services.job_service import get_job_service
from app.worker.handlers import handle_job_failure
from app.worker.startup import init_worker


def cleanup_stale_jobs() -> None:
    """Clean up any jobs stuck in processing from previous runs."""
    print("[STARTUP] Checking for stale processing jobs...", flush=True)
    job_service = get_job_service()
    cleaned = job_service.cleanup_stale_processing_jobs()

    if cleaned:
        print(f"[STARTUP] Cleaned up stale jobs: {cleaned}", flush=True)
    else:
        print("[STARTUP] No stale processing jobs found", flush=True)


def main():
    """Run the RQ worker with model pre-loading."""
    # Clean up stale jobs from previous crashes
    cleanup_stale_jobs()

    # Initialize model downloads in background
    init_worker()

    # Connect to Redis
    redis_conn = Redis.from_url(settings.redis_url)

    # Start the worker with exception handler
    worker = Worker(
        queues=["default"],
        connection=redis_conn,
        exception_handlers=[handle_job_failure],
    )

    worker.work()


if __name__ == "__main__":
    main()
