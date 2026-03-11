from celery import Celery
from app.config import settings

celery_app = Celery(
    "whisper_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 min max par task
    worker_prefetch_multiplier=1,  # 1 task à la fois par worker
)
