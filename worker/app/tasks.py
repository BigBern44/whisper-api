import io
import json
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

SAMPLE_RATE = 16_000
MAX_SAMPLES  = 480_000   # 30s × 16 000 Hz — longueur fixe pour le batching
MODEL_NAME   = "whisper"


@lru_cache()
def get_triton_client() -> httpclient.InferenceServerClient:
    return httpclient.InferenceServerClient(url=settings.triton_url)


def decode_audio(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
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


def pad_audio(audio: np.ndarray, max_samples: int = MAX_SAMPLES) -> tuple[np.ndarray, int]:
    """
    Tronque ou padde l'audio à max_samples.
    Retourne (audio_padded, longueur_réelle).
    """
    real_len = min(len(audio), max_samples)
    audio    = audio[:real_len]  # tronque si > 30s

    padded = np.zeros(max_samples, dtype=np.float32)
    padded[:real_len] = audio
    return padded, real_len


@shared_task(bind=True, name="transcribe_audio")
def transcribe_audio(
    self,
    s3_key: str,
    language: str | None = None,
    task: str = "transcribe",
):
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )

    try:
        client = get_triton_client()

        # ── 1. Téléchargement S3 ──────────────────────────────────────────────
        buffer = io.BytesIO()
        try:
            s3.download_fileobj(settings.s3_bucket, s3_key, buffer)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise self.retry(exc=e, countdown=5, max_retries=3)
            raise

        # ── 2. Décodage + padding à longueur fixe ────────────────────────────
        audio_signal          = decode_audio(buffer.getvalue(), target_sr=SAMPLE_RATE)
        audio_padded, real_len = pad_audio(audio_signal, max_samples=MAX_SAMPLES)

        # ── 3. Construction des inputs Triton ─────────────────────────────────
        # Avec max_batch_size > 0, le client doit inclure la dim batch (ici 1)
        # shape [batch, ...] → Triton regroupe plusieurs requêtes en [N, ...]
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

        # ── 4. Inférence ──────────────────────────────────────────────────────
        try:
            response = client.infer(
                model_name=MODEL_NAME,
                inputs=[audio_input, len_input, lang_input, task_input],
                outputs=outputs,
            )
        except InferenceServerException as e:
            raise RuntimeError(f"Triton inference failed: {e}") from None

        # ── 5. Décodage de la réponse ─────────────────────────────────────────
        result_str = response.as_numpy("transcription")[0][0].decode("utf-8")
        return json.loads(result_str)

    finally:
        try:
            s3.head_object(Bucket=settings.s3_bucket, Key=s3_key)
            s3.delete_object(Bucket=settings.s3_bucket, Key=s3_key)
        except ClientError:
            pass