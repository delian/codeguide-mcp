FROM mcr.microsoft.com/devcontainers/base:noble

ENV DEBIAN_FRONTEND=noninteractive
ENV PIPX_BIN_DIR=/usr/local/bin
ENV PIPX_HOME=/usr/local/pipx

RUN apt-get update && apt-get install -y \
    curl wget unzip \
    python3 python3-pip python-is-python3 python3-poetry \
    pre-commit \
    pipx \
    git-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pipx install uv
