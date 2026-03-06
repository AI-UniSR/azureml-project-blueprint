ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install uv via pip
RUN pip install --no-cache-dir uv

# Install Python dependencies
COPY requirements.txt requirements.txt

RUN uv pip install --system --no-cache -r requirements.txt