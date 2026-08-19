# Coding Guides MCP Server

A Model Context Protocol (MCP) server that provides access to coding guides and best practices for AI assistants like Claude and GitHub Copilot.

## What is this?

This MCP server exposes coding guidelines and style guides as resources that can be accessed by MCP clients. It's designed to extend or replace `AGENTS.md` files by providing a structured way to serve coding practices and guidelines to AI assistants during development.

## Features

- **Resource-based API**: Exposes coding guides through MCP resources
- **GitHub integration**: Loads guides from GitHub repositories over the web
- **Automatic caching**: Caches downloaded guides locally for offline access
- **Fallback support**: Uses local cache or directory when network is unavailable
- **Simple file-based storage**: Guides can be stored as Markdown files locally
- **Official MCP SDK**: Built on the Python `mcp` SDK (`MCPServer`, formerly FastMCP)
- **Easy integration**: Works with any MCP-compatible client (Claude Desktop, Cline, etc.)

## Available Resources

- `guides://list` - Lists all available coding guides
- `guides://{guide_name}` - Retrieves the content of a specific guide (e.g., `guides://python.md`)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/delian/codeguide-mcp.git
cd codeguide-mcp

# Install with uv (recommended)
uv install

# Or with pip
pip install -e .
```

### With Docker

```bash
docker build -t codeguide-mcp .
docker run -i codeguide-mcp
```

### In VS Code

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_codeguide--mcp-0098FF?logo=githubcopilot&logoColor=white)](https://vscode.dev/redirect/mcp/install?name=codeguide-mcp&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-i%22%2C%22--rm%22%2C%22delian%2Fcodeguide-mcp%22%5D%7D)

Or search for `codeguide-mcp` in the Extensions view MCP servers list (type `@mcp` in the Extensions search bar), or add it manually to `.vscode/mcp.json`:

```json
{
  "servers": {
    "codeguide-mcp": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "delian/codeguide-mcp"]
    }
  }
}
```

## Configuration

Configure the server by creating a `config.toml` file or setting environment variables:

### GitHub Configuration (Recommended)

To load guides from a GitHub repository:

```toml
github_repo = "owner/repository"  # e.g., "delian/codeguide-mcp"
github_path = "guides"            # Path to guides directory in repo
github_branch = "main"            # Branch to fetch from
  cache_dir = ".guides-cache"       # Local cache directory
log_level = "INFO"
```

### Local Directory Configuration

To use local guides only:

```toml
guides_dir = "guides"
log_level = "INFO"
```

### Environment Variables

- `GUIDES_GITHUB_REPO` - GitHub repository (format: `owner/repo`)
- `GUIDES_GITHUB_PATH` - Path to guides directory in repository (default: `guides`)
- `GUIDES_GITHUB_BRANCH` - Branch to fetch from (default: `main`)
- `GUIDES_CACHE_DIR` - Local cache directory (default: `.guides-cache`)
- `GUIDES_DIR` - Local directory containing guide files (default: `guides`)
- `GUIDES_LOG_LEVEL` - Logging level (default: `INFO`)

Transport (see [Remote deployment](#remote-deployment-google-cloud-run)):

- `GUIDES_TRANSPORT` - `stdio`, `streamable-http`, or `auto` (default: `auto` — HTTP when a `PORT` env var is present, stdio otherwise)
- `PORT` - Port to listen on in HTTP mode; takes precedence over `GUIDES_PORT` (Cloud Run injects this)
- `GUIDES_HOST` - Bind address in HTTP mode (default: `0.0.0.0`)
- `GUIDES_HTTP_PATH` - MCP endpoint path (default: `/mcp`)
- `GUIDES_STATELESS_HTTP` - Handle each request independently (default: `true`; required when replicas autoscale)
- `GUIDES_ALLOWED_HOSTS` - Host header allowlist enabling DNS-rebinding protection (default: empty = no Host validation)

### Behavior

1. **Network available + GitHub configured**: Fetches guides from GitHub and caches them locally
2. **Network unavailable**: Uses local cache if available
3. **No cache available**: Falls back to local `guides_dir` if configured

## Remote deployment (Google Cloud Run)

The same image serves both transports: it speaks stdio over a pipe by default,
and switches to Streamable HTTP when a `PORT` env var is present — which Cloud Run
always injects. No separate image or entrypoint is needed.

### 1. Publish the image

```bash
docker build -t delian/codeguide-mcp:0.1.0 -t delian/codeguide-mcp:latest .
docker push delian/codeguide-mcp:0.1.0
docker push delian/codeguide-mcp:latest
```

### 2. Deploy

```bash
gcloud run deploy codeguide-mcp \
  --image=docker.io/delian/codeguide-mcp:0.1.0 \
  --region=europe-west1 \
  --allow-unauthenticated \
  --port=8080 \
  --set-env-vars=GUIDES_TRANSPORT=streamable-http,GUIDES_GITHUB_REPO= \
  --memory=512Mi --cpu=1 \
  --min-instances=0 --max-instances=4 --concurrency=40
```

`GUIDES_GITHUB_REPO=` (empty) makes the service serve the guides baked into the
image. Leaving GitHub enabled adds a network round-trip per guide and runs into
the unauthenticated GitHub API limit of 60 requests/hour per egress IP, after
which the server silently falls back to those same baked-in files anyway.

The MCP endpoint is then `https://<service-url>/mcp`:

```bash
gcloud run services describe codeguide-mcp --region=europe-west1 \
  --format='value(status.url)'
```

Cloud Run answers on two hostnames for the same service — the
`SERVICE-PROJECTNUMBER.REGION.run.app` form printed by `gcloud run deploy`, and
the older `SERVICE-HASH-REGIONCODE.a.run.app` form that `status.url` reports.
Both are equivalent; either works in a client config.

### 3. Point clients at it

See [Connecting to a remote server](#connecting-to-a-remote-server) below for the
per-client configuration.

### Pulling from Docker Hub

Cloud Run deploys public Docker Hub images directly, but caches them for only an
hour and re-pulls anonymously afterwards, so a scale-up can hit Docker Hub's
anonymous pull limits and fail to start instances. For anything beyond casual
use, mirror through an Artifact Registry remote repository:

```bash
gcloud artifacts repositories create dockerhub \
  --repository-format=docker --location=europe-west1 \
  --mode=remote-repository --remote-docker-repo=DOCKER-HUB

gcloud run deploy codeguide-mcp \
  --image=europe-west1-docker.pkg.dev/PROJECT_ID/dockerhub/delian/codeguide-mcp:0.1.0 \
  ...
```

### Notes on running it publicly

- `--allow-unauthenticated` makes the endpoint world-callable. The server is
  read-only, but the `clear_cache` prompt is reachable by any caller and drops
  the in-memory caches, and traffic drives autoscaling cost — keep
  `--max-instances` capped. To restrict access, omit the flag and have clients
  send an identity token, or front the service with Cloud Armor / API Gateway.
- `GUIDES_STATELESS_HTTP` must stay `true` unless you also enable session
  affinity, since Cloud Run may route a session's requests to different instances.
- `GET /` returns 404 by design; only `/mcp` is served. Cloud Run's default
  startup probe is a TCP check on `$PORT`, so this is fine — don't configure an
  HTTP health check on `/`.
- Set `GUIDES_ALLOWED_HOSTS` to your service hostname to enable Host-header
  validation if you expose the service under a custom domain.

### Listing it in the MCP Registry as a remote

Once deployed, add the live URL to `server.json` alongside the `packages` entry
and re-publish with `mcp-publisher publish`:

```json
"remotes": [
  { "type": "streamable-http", "url": "https://codeguide-mcp-86057491046.europe-west1.run.app/mcp" }
]
```

## Adding Guides

### Using GitHub (Recommended)

If you've configured `github_repo`, simply add Markdown files to the specified directory in your GitHub repository. The server will automatically fetch and cache them.

### Using Local Directory

Add Markdown files to the `guides/` directory. Each file will be automatically available as a resource.

Example:
```bash
echo "# Python Style Guide\n\nUse PEP 8..." > guides/python.md
```

## Usage with MCP Clients

The server can be consumed two ways:

| Mode | Transport | How the client reaches it |
| --- | --- | --- |
| **Local** | stdio | Client spawns `python main.py` or `docker run -i` and talks over a pipe |
| **Remote** | Streamable HTTP | Client makes HTTPS requests to a hosted `…/mcp` URL |

Local mode needs no network and no hosting; remote mode lets a team share one
deployment and keeps the guides identical for everyone.

### Connecting to a remote server

A deployed instance exposes its MCP endpoint at `/mcp`. The snippets below use the
reference deployment:

```
https://codeguide-mcp-86057491046.europe-west1.run.app/mcp
```

It is public and needs no credentials. Substitute your own URL if you run the
service yourself — see [Remote deployment](#remote-deployment-google-cloud-run).

**VS Code** — `.vscode/mcp.json` for one workspace, or your user `mcp.json` for
every workspace:

```json
{
  "servers": {
    "codeguide-mcp": {
      "type": "http",
      "url": "https://codeguide-mcp-86057491046.europe-west1.run.app/mcp"
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add --transport http codeguide-mcp \
  https://codeguide-mcp-86057491046.europe-west1.run.app/mcp
```

**Cursor** — `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project):

```json
{
  "mcpServers": {
    "codeguide-mcp": {
      "url": "https://codeguide-mcp-86057491046.europe-west1.run.app/mcp"
    }
  }
}
```

**Claude Desktop** — add it as a custom connector in Settings, or bridge the
remote endpoint into a stdio client with
[`mcp-remote`](https://www.npmjs.com/package/mcp-remote):

```json
{
  "mcpServers": {
    "codeguide-mcp": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://codeguide-mcp-86057491046.europe-west1.run.app/mcp"]
    }
  }
}
```

**Any client** that speaks Streamable HTTP works — point it at the `/mcp` URL.
For servers behind authentication, pass a token with
`--header "Authorization: Bearer $(gcloud auth print-identity-token)"`
(Claude Code) or the client's equivalent `headers` block.

### Verifying a remote endpoint

A single `curl` confirms a deployment is live and public:

```bash
curl -s -X POST https://codeguide-mcp-86057491046.europe-west1.run.app/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
       "protocolVersion":"2025-06-18","capabilities":{},
       "clientInfo":{"name":"curl","version":"1"}}}'
```

A healthy server replies with an SSE `event: message` frame containing its
capabilities and instructions. Note that `GET /` returns 404 by design — only
`/mcp` is served.

To exercise every resource, tool, and prompt over HTTP instead:

```bash
uv run python verify_server.py --http https://codeguide-mcp-86057491046.europe-west1.run.app/mcp
```

### Local usage

#### Claude Desktop

Add to your `mcp.json`:

```json
{
  "mcpServers": {
    "coding-guides": {
      "command": "python",
      "args": ["-m", "main"]
    }
  }
}
```

or

```json
{
  "mcpServers": {
    "coding-guides": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "docker.io/delian/codeguide-mcp"]
    }
  }
}
```

### Other MCP Clients

Run the server and connect via stdio:

```bash
python main.py
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Run the server
python main.py
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or pull request.
