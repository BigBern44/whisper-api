import os
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
import boto3
from botocore.exceptions import ClientError
from app.celery_client import celery_app, send_transcription_task
from app.schemas import TranscribeResponse, JobStatusResponse, JobStatus
from app.config import settings
from pathlib import Path


s3_client = boto3.client(
    "s3",
    endpoint_url=settings.s3_endpoint_url,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
)

s3_public = boto3.client(
    "s3",
    endpoint_url=settings.s3_public_url,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

CONTENT_TYPES = {
    # vidéo
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
    ".avi":  "video/x-msvideo",
    # audio
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
    ".m4a":  "audio/mp4",
    ".aac":  "audio/aac",
}

# Modes de sortie disponibles
OUTPUT_MODES = {"embed", "srt", "text"}
# Modes interdits pour les fichiers audio (on ne peut pas ré-intégrer dans un audio)
AUDIO_FORBIDDEN_MODES = {"embed"}

app = FastAPI(
    title="Whisper Video Subtitling API",
    description="API de sous-titrage vidéo/audio asynchrone avec Whisper",
    version="4.0.0",
)

_cors_origins: list[str] = list(settings.cors_origins) if settings.cors_origins else []
# Toujours autoriser le dev front Vite en local
for _origin in ["http://localhost:5173", "http://127.0.0.1:5173"]:
    if _origin not in _cors_origins:
        _cors_origins.append(_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "Connection", "X-Accel-Buffering"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Bus d'événements in-memory
# job_id → asyncio.Queue  (un message poussé par le webhook, lu par le SSE)
# ─────────────────────────────────────────────────────────────────────────────
_job_queues: dict[str, asyncio.Queue] = {}


def _get_queue(job_id: str) -> asyncio.Queue:
    if job_id not in _job_queues:
        _job_queues[job_id] = asyncio.Queue(maxsize=1)
    return _job_queues[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers S3
# ─────────────────────────────────────────────────────────────────────────────

def upload_file_to_s3(file_bytes: bytes, file_ext: str) -> str:
    """Stocke le fichier source (vidéo ou audio) dans MinIO."""
    prefix = "video" if file_ext in VIDEO_EXTENSIONS else "audio"
    key = f"{prefix}/pending/{uuid.uuid4()}{file_ext}"
    s3_client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=file_bytes,
        ContentType=CONTENT_TYPES.get(file_ext, "application/octet-stream"),
    )
    return key


def make_presigned_url(output_key: str) -> str:
    try:
        return s3_public.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": output_key},
            ExpiresIn=3600,
        )
    except ClientError:
        return f"{settings.s3_public_url}/{settings.s3_bucket}/{output_key}"


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Subtitling
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/subtitles", response_model=TranscribeResponse, tags=["Subtitling"])
async def submit_file(
    file: UploadFile,
    language: str | None = Query(None, description="Code langue ISO (fr, en, etc.)"),
    mode: str = Query(
        "embed",
        description=(
            "Mode de sortie : "
            "'embed' = vidéo avec sous-titres intégrés (vidéo uniquement), "
            "'srt' = fichier SRT avec timestamps, "
            "'text' = texte brut sans timestamps"
        ),
    ),
):
    """
    Soumet une vidéo ou un fichier audio pour sous-titrage asynchrone.

    - **embed** : réintègre les sous-titres dans la vidéo (soft subtitles). Vidéo uniquement.
    - **srt**   : retourne un fichier `.srt` avec timestamps. Vidéo et audio.
    - **text**  : retourne un fichier `.txt` avec le texte brut. Vidéo et audio.
    """
    if not file.filename:
        raise HTTPException(400, "Fichier requis")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Format non supporté. Formats acceptés : {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if mode not in OUTPUT_MODES:
        raise HTTPException(
            400,
            f"Mode invalide. Valeurs acceptées : {', '.join(OUTPUT_MODES)}",
        )

    is_audio = ext in AUDIO_EXTENSIONS
    if is_audio and mode in AUDIO_FORBIDDEN_MODES:
        raise HTTPException(
            400,
            "Le mode 'embed' n'est pas disponible pour les fichiers audio. "
            "Utilisez 'srt' ou 'text'.",
        )

    content = await file.read()
    s3_key  = upload_file_to_s3(content, ext)

    callback_url = f"{settings.api_internal_url}/internal/webhook"
    task = send_transcription_task(
        s3_key,
        language,
        callback_url=callback_url,
        mode=mode,
    )

    _get_queue(task.id)

    return TranscribeResponse(job_id=task.id, status=JobStatus.PENDING)


@app.get("/subtitles/{job_id}/stream", tags=["Subtitling"])
async def stream_job(job_id: str):
    """
    SSE — connexion unique, bloquée jusqu'à réception du webhook.

    Événements :
    - `connected` : connexion établie
    - `done`      : traitement terminé, `data` = presigned URL du fichier de sortie
    - `error`     : erreur, `data` = message
    """
    queue = _get_queue(job_id)

    async def generator():
        try:
            yield f"event: connected\ndata: {job_id}\n\n"

            try:
                message = await asyncio.wait_for(queue.get(), timeout=1800)
            except asyncio.TimeoutError:
                yield "event: error\ndata: timeout\n\n"
                return

            yield f"event: {message['event']}\ndata: {message['data']}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            _job_queues.pop(job_id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route — Webhook interne (appelée par Celery à la fin du traitement)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/internal/webhook", tags=["Internal"])
async def webhook(request: Request):
    """
    Reçoit le résultat de la task Celery et débloque le SSE correspondant.
    Body JSON :
      { "job_id": str, "status": "done"|"error", "output_s3_key"?: str, "error"?: str }
    """
    body   = await request.json()
    job_id = body.get("job_id")

    if not job_id:
        raise HTTPException(400, "job_id manquant")

    if body.get("status") == "done":
        download_url = make_presigned_url(body.get("output_s3_key", ""))
        message = {"event": "done", "data": download_url}
    else:
        message = {"event": "error", "data": body.get("error", "Erreur inconnue")}

    queue = _job_queues.get(job_id)
    if queue:
        await queue.put(message)

    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Route — Statut (poll classique optionnel)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/subtitles/{job_id}", response_model=JobStatusResponse, tags=["Subtitling"])
async def get_job(job_id: str):
    task       = AsyncResult(job_id, app=celery_app)
    status_map = {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.STARTED,
        "RETRY":   JobStatus.STARTED,
        "SUCCESS": JobStatus.COMPLETED,
        "FAILURE": JobStatus.FAILED,
    }
    response = JobStatusResponse(
        job_id=job_id,
        status=status_map.get(task.state, JobStatus.PENDING),
    )

    if task.state == "SUCCESS":
        output_key = task.result.get("output_s3_key", "")
        response.result = {
            "text":         task.result.get("text"),
            "mode":         task.result.get("mode"),
            "download_url": make_presigned_url(output_key),
        }
    elif task.state == "FAILURE":
        response.error = str(task.result)

    return response


@app.delete("/subtitles/{job_id}", tags=["Subtitling"])
async def cancel_job(job_id: str):
    AsyncResult(job_id, app=celery_app).revoke(terminate=True)
    _job_queues.pop(job_id, None)
    return {"job_id": job_id, "status": "cancelled"}


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    try:
        celery_app.control.ping(timeout=1)
        redis_status = "ok"
    except Exception:
        redis_status = "error"
    return {"api": "ok", "redis": redis_status}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Whisper Video Subtitling API",
        "version": "4.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }