FROM python:3.13-slim

# Ownership proof for the MCP Registry: this label MUST match the "name"
# field in server.json, or `mcp-publisher publish` fails validation.
LABEL io.modelcontextprotocol.server.name="io.github.delian/codeguide-mcp"

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --link pyproject.toml uv.lock ./
# Install from the lockfile (--locked fails the build if it is stale) so the
# image gets the exact dependency versions verified locally, instead of
# re-resolving pyproject.toml floors to whatever is newest at build time.
RUN uv export --locked --no-dev --no-emit-project --format requirements.txt -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
COPY --link main.py MCP-instructions.md README.md verify_server.py ./
COPY --link coding_guides_server coding_guides_server/
COPY --link config config/
COPY --link guides guides/
# prompts/ holds the dynamic prompt definitions; without it the image
# registers zero dynamic prompts (bug_hunt, check_security, ...).
COPY --link prompts prompts/
ENV PYTHONUNBUFFERED=1
# Select dynaconf's [production] settings (WARNING log level); the default
# environment is "development", which logs at DEBUG in the shipped image.
ENV ENV_FOR_DYNACONF=production
# Documents the default HTTP port. The image serves stdio unless a PORT env var
# is present (Cloud Run injects it), so `docker run -i` keeps working locally.
EXPOSE 8080
CMD ["python", "main.py"]
