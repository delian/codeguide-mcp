FROM python:3.13-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --link pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml
COPY --link main.py MCP-instructions.md README.md verify_server.py ./
COPY --link coding_guides_server coding_guides_server/
COPY --link config config/
COPY --link guides guides/
ENV PYTHONUNBUFFERED=1
CMD ["python", "main.py"]
