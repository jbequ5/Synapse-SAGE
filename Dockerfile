# Dockerfile — Private Synapse Intelligence Server (10/10 production ready)
FROM python:3.11-slim

WORKDIR /app

# Install system deps + Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire private Synapse code
COPY synapse/ ./synapse/

# Pre-cache embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "synapse.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
