FROM python:3.12-slim

# Install only what you need for runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir early
WORKDIR /app

# Install uv (faster package manager / runtime)
RUN pip install --no-cache-dir uv

# Copy your app
COPY main.py .

CMD ["uv", "run", "main.py"]
