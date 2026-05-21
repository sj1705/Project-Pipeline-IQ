from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "PipelineIQ"
    debug: bool = False

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/pipelineiq"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    # Embedding
    # Embedding
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    # AWS
    aws_region: str = "ap-south-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "pipelineiq-documents"

    class Config:
        env_file = ".env"


settings = Settings()