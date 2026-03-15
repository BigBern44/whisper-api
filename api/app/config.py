from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: str = "/tmp/whisper_uploads"
    s3_endpoint_url: str = "http://minio:9000"   # interne Docker (workers)
    s3_public_url:   str = "http://localhost:9000" # externe navigateur
    api_internal_url: str = "http://whisper-api:8000"   # hostname Docker interne
    s3_bucket: str = "whisper-audio"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "null"]

    class Config:
        env_file = ".env"


settings = Settings()