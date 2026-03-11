import io
import json
import boto3
import numpy as np
import tritonclient.http as httpclient
from celery import shared_task
from app.config import settings
from functools import lru_cache
from botocore.exceptions import ClientError

@lru_cache()
def get_triton_client() -> httpclient.InferenceServerClient:
    """Retourne un client Triton"""
    return httpclient.InferenceServerClient(url=settings.triton_url)

@shared_task(bind=True, name="transcribe_audio")
def transcribe_audio(self, s3_key: str, language: str | None = None):
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )

    try:
        client = get_triton_client()

        buffer = io.BytesIO()
        try:
            s3.download_fileobj(settings.s3_bucket, s3_key, buffer)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise self.retry(exc=e, countdown=5, max_retries=3)
            raise

        audio_bytes = buffer.getvalue()

        audio_input = httpclient.InferInput("audio_bytes", [len(audio_bytes)], "BYTES")
        audio_input.set_data_from_numpy(np.frombuffer(audio_bytes, dtype=np.uint8))

        lang_value = language if language else "auto"
        lang_input = httpclient.InferInput("language", [1], "BYTES")
        lang_input.set_data_from_numpy(np.array([lang_value], dtype=object))

        outputs = [httpclient.InferRequestedOutput("transcription")]

        try:
            response = client.infer(
                model_name="whisper",
                inputs=[audio_input, lang_input],
                outputs=outputs,
            )
        except InferenceServerException as e:
            # Re-raise as a standard exception so Celery can pickle it
            raise RuntimeError(f"Triton inference failed: {e}") from None

        result_str = response.as_numpy("transcription")[0].decode("utf-8")
        return json.loads(result_str)

    finally:
        try:
            s3.head_object(Bucket=settings.s3_bucket, Key=s3_key)
            s3.delete_object(Bucket=settings.s3_bucket, Key=s3_key)
        except ClientError:
            pass