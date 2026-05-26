FROM python:3.11-slim

WORKDIR /app

# Install dependencies (pinned versions from working venv)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy app code
COPY app/ ./app/
COPY uploads/ ./uploads/

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
