from celery import Celery
from app.config import settings

# Client Celery léger - ne charge pas les tasks, envoie seulement
celery_app = Celery(
    "whisper_client",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

def send_transcription_task(s3_key: str, language: str | None = None, callback_url: str | None = None):
    return celery_app.send_task(
        "transcribe_audio",
        args=[s3_key],
        kwargs={"language": language, "callback_url": callback_url},
    )