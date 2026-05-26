FROM rocker/r-ver:4.4.3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpng-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment
RUN python3 -m venv /opt/pyenv
RUN /opt/pyenv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/pyenv/bin/pip install --no-cache-dir \
    mlflow==2.22.2 \
    azureml-mlflow==1.60.0.post1 \
    numpy \
    pandas \
    scikit-learn \
    joblib

ENV PATH="/opt/pyenv/bin:${PATH}"

# R packages (minimal for calibration plot)
RUN R -q -e "install.packages(c( \
    'argparse', \
    'readr', \
    'dplyr', \
    'ggplot2', \
    'reticulate' \
), repos='https://cloud.r-project.org')"

WORKDIR /app