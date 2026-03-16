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

SAMPLE_RATE  = 16_000
MAX_SAMPLES  = 80_000   # 30s × 16 000 Hz — longueur fixe pour le batching
MODEL_NAME   = "whisper"

# Extensions vidéo supportées → (conteneur ffmpeg, codec sous-titres)
VIDEO_CONTAINERS = {
    ".mp4":  ("mp4",       "mov_text"),
    ".mov":  ("mov",       "mov_text"),
    ".mkv":  ("matroska",  "srt"),
    ".webm": ("webm",      "webvtt"),
    ".avi":  ("avi",       "srt"),
}
DEFAULT_CONTAINER = ("mp4", "mov_text")

# Extensions audio (traitement direct, sans extraction vidéo)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

# Modes de sortie
MODE_EMBED = "embed"   # vidéo avec sous-titres intégrés (soft)
MODE_SRT   = "srt"     # fichier .srt avec timestamps
MODE_TEXT  = "text"    # fichier .txt texte brut sans timestamps


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
    float32 mono à `target_sr` Hz. Requiert ffmpeg installé.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_f32le",
                "-ar", str(target_sr),
                "-ac", "1",
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


def decode_audio_file(audio_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Décode un fichier audio (mp3, wav, flac, ogg, m4a, aac…) depuis le disque.
    Retourne un tableau float32 mono à `target_sr` Hz.
    """
    try:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        # Fallback : passe par librosa (gère mp3, m4a, aac via ffmpeg)
        audio, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def split_audio_chunks(audio: np.ndarray, max_samples: int = MAX_SAMPLES) -> list[tuple[np.ndarray, int, float]]:
    """
    Découpe l'audio en chunks de max_samples avec un overlap de 1s.
    Retourne une liste de (chunk_padded, real_len, offset_seconds).
    """
    overlap = SAMPLE_RATE          # 1 seconde d'overlap
    step    = max_samples - overlap
    total   = len(audio)
    chunks  = []
    pos     = 0

    while pos < total:
        end      = min(pos + max_samples, total)
        chunk    = audio[pos:end]
        real_len = len(chunk)

        padded            = np.zeros(max_samples, dtype=np.float32)
        padded[:real_len] = chunk

        chunks.append((padded, real_len, pos / SAMPLE_RATE))
        pos += step

    return chunks


def merge_segments(chunks_segments: list[tuple[list[dict], float]]) -> tuple[str, list[dict]]:
    """
    Fusionne les segments de plusieurs chunks en ajustant les timestamps.
    Déduplique les segments qui chevauchent la zone d'overlap.
    """
    all_segments:    list[dict] = []
    full_text_parts: list[str]  = []

    for segments, offset in chunks_segments:
        for seg in segments:
            start = float(seg["start"]) + offset
            end   = float(seg["end"])   + offset
            text  = seg["text"].strip()

            if all_segments and start < all_segments[-1]["end"] - 0.1:
                continue

            all_segments.append({"start": start, "end": end, "text": text})
            full_text_parts.append(text)

    return " ".join(full_text_parts), all_segments


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle formatters
# ─────────────────────────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    """Formate des secondes en timestamp SRT : HH:MM:SS,mmm"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """
    Convertit une liste de segments Whisper en chaîne SRT valide.
    [{"start": float, "end": float, "text": str}, ...]
    """
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _ts(float(seg["start"]))
        end   = _ts(float(seg["end"]))
        text  = seg["text"].strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def segments_to_plain_text(segments: list[dict]) -> str:
    """Retourne le texte brut, un segment par ligne, sans timestamps."""
    return "\n".join(seg["text"].strip() for seg in segments if seg["text"].strip())


# ─────────────────────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────────────────────

def embed_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    sub_codec: str = "mov_text",
) -> None:
    """Ajoute les sous-titres SRT comme piste soft dans le conteneur vidéo."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", srt_path,
                "-map", "0",
                "-map", "1:0",
                "-c", "copy",
                "-c:s", sub_codec,
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


# ─────────────────────────────────────────────────────────────────────────────
# Webhook
# ─────────────────────────────────────────────────────────────────────────────

def _notify_webhook(
    url: str,
    job_id: str,
    status: str,
    output_s3_key: str = "",
    error: str = "",
) -> None:
    """Appelle le webhook de l'API avec le résultat du job. Silencieux en cas d'échec."""
    try:
        requests.post(
            url,
            json={
                "job_id":        job_id,
                "status":        status,
                "output_s3_key": output_s3_key,
                "error":         error,
            },
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
    callback_url: str | None = None,
    mode: str = MODE_EMBED,
):
    """
    Pipeline de transcription Whisper.

    Modes de sortie (`mode`) :
    - ``embed`` : réintègre les sous-titres dans la vidéo (soft subtitles, sans ré-encodage).
                  Requiert un fichier vidéo en entrée.
    - ``srt``   : génère un fichier SRT avec timestamps et l'uploade sur S3.
    - ``text``  : génère un fichier texte brut (sans timestamps) et l'uploade sur S3.

    Retourne un dict :
    {
        "text":           str,      # transcription complète
        "segments":       [...],    # segments horodatés
        "mode":           str,      # mode utilisé
        "output_s3_key":  str,      # clé S3 du fichier de sortie
    }
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )

    ext      = Path(s3_key).suffix.lower()
    is_audio = ext in AUDIO_EXTENSIONS
    is_video = ext in VIDEO_CONTAINERS

    # Sécurité : embed interdit sur audio
    if is_audio and mode == MODE_EMBED:
        mode = MODE_SRT
        logging.warning(
            f"[{self.request.id}] Mode 'embed' incompatible avec un fichier audio — "
            f"bascule automatique vers 'srt'."
        )

    # Clé S3 de sortie
    if output_s3_key is None:
        stem = Path(s3_key).stem
        parent = Path(s3_key).parent
        if mode == MODE_EMBED:
            output_s3_key = str(parent / f"{stem}_subtitled{ext}")
        elif mode == MODE_SRT:
            output_s3_key = str(parent / f"{stem}.srt")
        else:  # text
            output_s3_key = str(parent / f"{stem}.txt")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, f"input{ext}")
        srt_file   = os.path.join(tmpdir, "subtitles.srt")
        txt_file   = os.path.join(tmpdir, "subtitles.txt")

        if mode == MODE_EMBED:
            video_out = os.path.join(tmpdir, f"output{ext}")
            fmt, sub_codec = VIDEO_CONTAINERS.get(ext, DEFAULT_CONTAINER)

        try:
            triton = get_triton_client()

            # ── 1. Téléchargement S3 ─────────────────────────────────────────
            try:
                s3.download_file(settings.s3_bucket, s3_key, input_file)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    raise self.retry(exc=e, countdown=5, max_retries=3)
                raise

            # ── 2. Extraction / décodage audio ───────────────────────────────
            if is_audio:
                audio_signal = decode_audio_file(input_file, target_sr=SAMPLE_RATE)
            else:
                audio_signal = extract_audio_from_video(input_file, target_sr=SAMPLE_RATE)

            chunks = split_audio_chunks(audio_signal, max_samples=MAX_SAMPLES)
            logging.info(f"[{self.request.id}] {len(chunks)} chunk(s) à transcrire (mode={mode})")

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
                    segs = [{
                        "start": 0.0,
                        "end":   real_len / SAMPLE_RATE,
                        "text":  chunk_result.get("text", ""),
                    }]

                chunks_segments.append((segs, offset))
                logging.info(
                    f"[{self.request.id}] chunk {i+1}/{len(chunks)} — "
                    f"{len(segs)} segments"
                )

            # ── 4. Fusion des segments avec timestamps corrigés ───────────────
            full_text, segments = merge_segments(chunks_segments)

            # ── 5. Génération du fichier de sortie selon le mode ─────────────
            if mode == MODE_EMBED:
                # Génère le SRT puis l'intègre dans la vidéo
                srt_content = segments_to_srt(segments)
                with open(srt_file, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                embed_subtitles(input_file, srt_file, video_out, sub_codec=sub_codec)

                upload_path = video_out
                content_type = f"video/{fmt}"

            elif mode == MODE_SRT:
                # Génère uniquement le fichier SRT
                srt_content = segments_to_srt(segments)
                with open(srt_file, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                upload_path  = srt_file
                content_type = "text/plain; charset=utf-8"

            else:  # MODE_TEXT
                # Génère le texte brut sans timestamps
                plain_text = segments_to_plain_text(segments)
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(plain_text)

                upload_path  = txt_file
                content_type = "text/plain; charset=utf-8"

            # ── 6. Upload S3 du fichier de sortie ────────────────────────────
            s3.upload_file(
                upload_path,
                settings.s3_bucket,
                output_s3_key,
                ExtraArgs={"ContentType": content_type},
            )

            task_result = {
                "text":           full_text,
                "segments":       segments,
                "mode":           mode,
                "output_s3_key":  output_s3_key,
            }

            # ── 7. Webhook → notifie l'API ────────────────────────────────────
            if callback_url:
                _notify_webhook(
                    callback_url,
                    self.request.id,
                    status="done",
                    output_s3_key=output_s3_key,
                )

            return task_result

        except Exception as exc:
            if callback_url:
                _notify_webhook(
                    callback_url,
                    self.request.id,
                    status="error",
                    error=str(exc),
                )
            raise

        finally:
            # ── 8. Nettoyage de la clé source sur S3 ─────────────────────────
            try:
                s3.head_object(Bucket=settings.s3_bucket, Key=s3_key)
                s3.delete_object(Bucket=settings.s3_bucket, Key=s3_key)
            except ClientError:
                pass