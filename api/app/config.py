from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://redis:6379/0"

    # S3 / MinIO
    s3_endpoint_url: str  = "http://minio:9000"
    s3_public_url: str    = "http://localhost:9000"   # URL accessible depuis le navigateur
    s3_bucket: str        = "whisper-audio"
    s3_access_key: str    = "minioadmin"
    s3_secret_key: str    = "minioadmin"

    # Triton
    triton_url: str = "triton:8001"

    # API — URL interne appelée par les workers pour le webhook
    api_internal_url: str = "http://api:8000"

    # CORS — liste d'origines autorisées séparées par des virgules
    # Ex: CORS_ORIGINS="http://localhost:5173,https://myapp.com"
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Permet de passer CORS_ORIGINS="url1,url2" en variable d'environnement
        env_prefix = ""


settings = Settings()