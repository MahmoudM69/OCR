"""
Custom RQ worker entry point with model pre-loading.

This script initializes model downloads in the background
before starting the RQ worker.
"""
import sys

from redis import Redis
from rq import Worker

from app.config import settings
from app.worker.startup import init_worker


def main():
    """Run the RQ worker with model pre-loading."""
    # Initialize model downloads in background
    init_worker()

    # Connect to Redis
    redis_conn = Redis.from_url(settings.redis_url)

    # Start the worker
    worker = Worker(
        queues=["default"],
        connection=redis_conn,
    )

    worker.work()


if __name__ == "__main__":
    main()
