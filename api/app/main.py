import os
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, Query
from celery.result import AsyncResult
import boto3
from app.celery_client import celery_app, send_transcription_task
from app.schemas import TranscribeResponse, JobStatusResponse, JobStatus
from app.config import settings
from pathlib import Path


# Créer le dossier d'upload
os.makedirs(settings.upload_dir, exist_ok=True)
s3_client = boto3.client(
    "s3",
    endpoint_url=settings.s3_endpoint_url,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

app = FastAPI(
    title="Whisper Transcription API",
    description="API de transcription audio asynchrone avec Whisper",
    version="1.0.0"
)

def upload_audio_to_s3(file_bytes: bytes, file_ext: str) -> str:
    key = f"audio/pending/{uuid.uuid4()}{file_ext}"
    s3_client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=file_bytes,
        ContentType="audio/mpeg"  # adapter selon ext
    )
    return key



@app.post("/transcribe", response_model=TranscribeResponse, tags=["Transcription"])
async def submit_transcription(
    file: UploadFile,
    language: str | None = Query(None, description="Code langue ISO (fr, en, etc.)")
):
    if not file.filename:
        raise HTTPException(400, "Fichier requis")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Format non supporté. Formats acceptés: {ALLOWED_EXTENSIONS}")

    # Lire le fichier en bytes directement
    content = await file.read()
    s3_key = upload_audio_to_s3(content, Path(file.filename).suffix)

    # Envoyer les bytes + l'extension à la task
    task = send_transcription_task(s3_key, language)

    return TranscribeResponse(job_id=task.id, status=JobStatus.PENDING)

@app.get("/transcribe/{job_id}", response_model=JobStatusResponse, tags=["Transcription"])
async def get_transcription_status(job_id: str):
    """
    Récupère le statut et le résultat d'une transcription.
    """
    task = AsyncResult(job_id, app=celery_app)
    
    status_map = {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.STARTED,
        "SUCCESS": JobStatus.COMPLETED,
        "FAILURE": JobStatus.FAILED,
    }
    
    status = status_map.get(task.state, JobStatus.PENDING)
    response = JobStatusResponse(job_id=job_id, status=status)
    
    if task.state == "SUCCESS":
        result = task.result
        if result.get("status") == "failed":
            response.status = JobStatus.FAILED
            response.error = result.get("error")
        else:
            response.result = result
    elif task.state == "FAILURE":
        response.error = str(task.result)
    
    return response


@app.delete("/transcribe/{job_id}", tags=["Transcription"])
async def cancel_transcription(job_id: str):
    """Annule une transcription en cours."""
    task = AsyncResult(job_id, app=celery_app)
    task.revoke(terminate=True)
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        celery_app.control.ping(timeout=1)
        redis_status = "ok"
    except Exception:
        redis_status = "error"
    
    return {"api": "ok", "redis": redis_status}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Whisper Transcription API",
        "docs": "/docs",
        "health": "/health"
    }
