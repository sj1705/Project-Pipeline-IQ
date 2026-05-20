from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    description="Self-Optimizing RAG Orchestration System",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running", "status": "healthy"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "debug": settings.debug,
    }