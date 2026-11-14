FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless>=4.8.0"

# Copy only counter service files (completely self-contained)
COPY . /app

# Create data directory for local outbox SQLite database
RUN mkdir -p /data /app/data

ENV PYTHONPATH=/app
ENV DB_PATH=/data/counter_outbox.db

CMD ["python", "main.py"]
