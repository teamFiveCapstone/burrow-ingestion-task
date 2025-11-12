FROM python:3.9-slim
# Install only needed OS packages in one layer and cleanup apt lists
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Copy only requirements first to leverage Docker cache
COPY requirements.txt /tmp/requirements.txt
# Install Python deps without pip cache
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Copy source
COPY ./src /src
WORKDIR /src
CMD ["python", "main.py"]