import io
import json
import os
import subprocess
import tempfile
import logging
import requests
import boto3
import numpy as np
import soundfile as sf
import librosa
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from celery import shared_task
from app.config import settings
from functools import lru_cache
from botocore.exceptions import ClientError
from pathlib import Path

SAMPLE_RATE = 16_000
MAX_SAMPLES  = 480_000   # 30s × 16 000 Hz — longueur fixe pour le batching
MODEL_NAME   = "whisper"

# Extensions vidéo supportées → conteneur de sortie
VIDEO_CONTAINERS = {
    ".mp4":  ("mp4",  "mov_text"),   # subtitle codec mp4
    ".mov":  ("mov",  "mov_text"),
    ".mkv":  ("matroska", "srt"),    # subtitle codec mkv
    ".webm": ("webm", "webvtt"),
    ".avi":  ("avi",  "srt"),
}
DEFAULT_CONTAINER = ("mp4", "mov_text")


# ─────────────────────────────────────────────────────────────────────────────
# Triton client (singleton)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache()
def get_triton_client() -> httpclient.InferenceServerClient:
    return httpclient.InferenceServerClient(url=settings.triton_url)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_from_video(video_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extrait la piste audio d'une vidéo via ffmpeg et retourne un tableau numpy
    float32 mono à `target_sr` Hz.
    Requiert ffmpeg installé sur la machine.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                      # pas de flux vidéo
                "-acodec", "pcm_f32le",     # PCM float 32 bits little-endian
                "-ar", str(target_sr),      # fréquence cible
                "-ac", "1",                 # mono
                wav_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg audio extraction failed: {e.stderr.decode('utf-8', errors='replace')}"
        ) from e

    try:
        audio, _ = sf.read(wav_path, dtype="float32", always_2d=False)
    finally:
        os.unlink(wav_path)

    return audio.astype(np.float32)


def decode_audio(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Décode un fichier audio brut (non vidéo) depuis des bytes."""
    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=None, mono=True, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def split_audio_chunks(audio: np.ndarray, max_samples: int = MAX_SAMPLES) -> list[tuple[np.ndarray, int, float]]:
    """
    Découpe l'audio en chunks de max_samples avec un overlap de 1s pour éviter
    de couper des mots en plein milieu.
    Retourne une liste de (chunk_padded, real_len, offset_seconds).
    """
    overlap    = SAMPLE_RATE          # 1 seconde d'overlap
    step       = max_samples - overlap
    total      = len(audio)
    chunks     = []
    pos        = 0

    while pos < total:
        end      = min(pos + max_samples, total)
        chunk    = audio[pos:end]
        real_len = len(chunk)

        # Padde à max_samples pour que Triton reçoive toujours la même shape
        padded         = np.zeros(max_samples, dtype=np.float32)
        padded[:real_len] = chunk

        chunks.append((padded, real_len, pos / SAMPLE_RATE))
        pos += step

    return chunks


def merge_segments(chunks_segments: list[tuple[list[dict], float]]) -> tuple[str, list[dict]]:
    """
    Fusionne les segments de plusieurs chunks en ajustant les timestamps.
    Déduplique les segments qui chevauchent la zone d'overlap.
    """
    all_segments: list[dict] = []
    full_text_parts: list[str] = []

    for segments, offset in chunks_segments:
        for seg in segments:
            start = float(seg["start"]) + offset
            end   = float(seg["end"])   + offset
            text  = seg["text"].strip()

            # Déduplique : ignore si ce segment chevauche trop le dernier ajouté
            if all_segments and start < all_segments[-1]["end"] - 0.1:
                continue

            all_segments.append({"start": start, "end": end, "text": text})
            full_text_parts.append(text)

    return " ".join(full_text_parts), all_segments


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    """Formate des secondes en timestamp SRT  HH:MM:SS,mmm"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """
    Convertit une liste de segments Whisper
    [{"start": float, "end": float, "text": str}, ...]
    en chaîne SRT valide.
    """
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _ts(float(seg["start"]))
        end   = _ts(float(seg["end"]))
        text  = seg["text"].strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def embed_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    sub_codec: str = "mov_text",
) -> None:
    """
    Ajoute les sous-titres SRT comme piste soft dans le conteneur vidéo.
    Les flux audio/vidéo sont copiés sans ré-encodage.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", srt_path,
                "-map", "0",             # tous les flux de la vidéo originale
                "-map", "1:0",           # piste de sous-titres
                "-c", "copy",            # copie sans ré-encodage
                "-c:s", sub_codec,       # codec sous-titres adapté au conteneur
                "-metadata:s:s:0", "language=und",
                "-metadata:s:s:0", "title=Transcription",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg subtitle embedding failed: {e.stderr.decode('utf-8', errors='replace')}"
        ) from e


def _notify_webhook(url: str, job_id: str, status: str, output_s3_key: str = "", error: str = "") -> None:
    """Appelle le webhook de l'API avec le résultat du job. Silencieux en cas d'échec."""
    try:
        requests.post(
            url,
            json={"job_id": job_id, "status": status, "output_s3_key": output_s3_key, "error": error},
            timeout=10,
        )
    except Exception as e:
        logging.warning(f"Webhook call failed for job {job_id}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Celery task
# ─────────────────────────────────────────────────────────────────────────────

@shared_task(bind=True, name="transcribe_audio")
def transcribe_video(
    self,
    s3_key: str,
    language: str | None = None,
    task: str = "transcribe",
    output_s3_key: str | None = None,
    callback_url: str | None = None,   # URL webhook à appeler en fin de traitement
):
    """
    Pipeline complet :
      1. Télécharge la vidéo depuis S3
      2. Extrait la piste audio avec ffmpeg
      3. Transcrit via Whisper/Triton (avec segments horodatés)
      4. Génère le fichier SRT
      5. Réintègre les sous-titres dans la vidéo (soft subtitles, sans ré-encodage)
      6. Upload la vidéo sous-titrée sur S3
      7. Supprime les fichiers temporaires S3

    Retourne un dict :
      {
        "transcription": str,          # texte complet
        "segments": [...],             # segments horodatés
        "output_s3_key": str,          # clé S3 de la vidéo sous-titrée
      }
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )

    ext           = Path(s3_key).suffix.lower()
    fmt, sub_codec = VIDEO_CONTAINERS.get(ext, DEFAULT_CONTAINER)

    # Clé S3 de sortie  (ex: "videos/interview_subtitled.mp4")
    if output_s3_key is None:
        stem          = Path(s3_key).stem
        output_s3_key = str(Path(s3_key).parent / f"{stem}_subtitled{ext}")

    # Fichiers temporaires locaux
    with tempfile.TemporaryDirectory() as tmpdir:
        video_in   = os.path.join(tmpdir, f"input{ext}")
        srt_file   = os.path.join(tmpdir, "subtitles.srt")
        video_out  = os.path.join(tmpdir, f"output{ext}")

        try:
            triton = get_triton_client()

            # ── 1. Téléchargement S3 ─────────────────────────────────────────
            try:
                s3.download_file(settings.s3_bucket, s3_key, video_in)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    raise self.retry(exc=e, countdown=5, max_retries=3)
                raise

            # ── 2. Extraction audio + découpage en chunks ────────────────────
            audio_signal = extract_audio_from_video(video_in, target_sr=SAMPLE_RATE)
            chunks       = split_audio_chunks(audio_signal, max_samples=MAX_SAMPLES)
            logging.info(f"[{self.request.id}] {len(chunks)} chunk(s) à transcrire")

            # ── 3. Inférence Triton chunk par chunk ──────────────────────────
            chunks_segments: list[tuple[list[dict], float]] = []

            for i, (audio_padded, real_len, offset) in enumerate(chunks):
                audio_input = httpclient.InferInput("audio_signal", [1, MAX_SAMPLES], "FP32")
                audio_input.set_data_from_numpy(audio_padded.reshape(1, MAX_SAMPLES))

                len_input = httpclient.InferInput("audio_len", [1, 1], "INT32")
                len_input.set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

                lang_input = httpclient.InferInput("language", [1, 1], "BYTES")
                lang_input.set_data_from_numpy(
                    np.array([[language if language else "auto"]], dtype=object)
                )

                task_input = httpclient.InferInput("task", [1, 1], "BYTES")
                task_input.set_data_from_numpy(np.array([[task]], dtype=object))

                outputs = [httpclient.InferRequestedOutput("transcription")]

                try:
                    response = triton.infer(
                        model_name=MODEL_NAME,
                        inputs=[audio_input, len_input, lang_input, task_input],
                        outputs=outputs,
                    )
                except InferenceServerException as e:
                    raise RuntimeError(f"Triton inference failed on chunk {i}: {e}") from None

                chunk_result: dict = json.loads(
                    response.as_numpy("transcription")[0][0].decode("utf-8")
                )
                segs = chunk_result.get("segments", [])

                if not segs:
                    # Fallback : un seul segment pour ce chunk
                    segs = [{"start": 0.0, "end": real_len / SAMPLE_RATE, "text": chunk_result.get("text", "")}]

                chunks_segments.append((segs, offset))
                logging.info(f"[{self.request.id}] chunk {i+1}/{len(chunks)} transcrit — {len(segs)} segments")

            # ── 4. Fusion de tous les segments avec timestamps corrigés ──────
            full_text, segments = merge_segments(chunks_segments)

            # ── 6. Génération du fichier SRT ─────────────────────────────────
            srt_content = segments_to_srt(segments)
            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)

            # ── 7. Intégration des sous-titres dans la vidéo ─────────────────
            embed_subtitles(video_in, srt_file, video_out, sub_codec=sub_codec)

            # ── 8. Upload S3 de la vidéo sous-titrée ─────────────────────────
            s3.upload_file(
                video_out,
                settings.s3_bucket,
                output_s3_key,
                ExtraArgs={"ContentType": f"video/{fmt}"},
            )

            task_result = {
                "text":           full_text,
                "segments":       segments,
                "output_s3_key":  output_s3_key,
                "subtitle_codec": sub_codec,
            }

            # ── 9. Webhook → notifie l'API que le job est terminé ─────────────
            if callback_url:
                _notify_webhook(callback_url, self.request.id, status="done", output_s3_key=output_s3_key)

            return task_result

        finally:
            # ── 10. Nettoyage de la clé source sur S3 ────────────────────────
            try:
                s3.head_object(Bucket=settings.s3_bucket, Key=s3_key)
                s3.delete_object(Bucket=settings.s3_bucket, Key=s3_key)
            except ClientError:
                pass