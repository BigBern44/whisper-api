from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: str = "/tmp/whisper_uploads"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "whisper-audio"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    class Config:
        env_file = ".env"


settings = Settings()
