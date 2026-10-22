FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only counter service files (completely self-contained)
COPY . /app

# Create data directory for local outbox SQLite database
RUN mkdir -p /data /app/data

ENV PYTHONPATH=/app
ENV DB_PATH=/data/counter_outbox.db

CMD ["python", "main.py"]
