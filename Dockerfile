FROM python:3.12-slim

# Env vars to speed Python a bit and avoid cache files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install only what you need for runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir early
WORKDIR /src

# Install Python deps in their own layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source last (so code changes don’t bust the pip cache layer)
COPY ./src .

CMD ["python", "main.py"]
