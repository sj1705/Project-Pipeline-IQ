from fastapi import FastAPI

app = FastAPI(
    title="PipelineIQ",
    description="Self-Optimizing RAG Orchestration System",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "PipelineIQ is running", "status": "healthy"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
