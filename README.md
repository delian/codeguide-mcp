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
- **FastMCP framework**: Built on the efficient FastMCP library for quick development
- **Easy integration**: Works with any MCP-compatible client (Claude Desktop, Cline, etc.)

## Available Resources

- `guides://list` - Lists all available coding guides
- `guides://{guide_name}` - Retrieves the content of a specific guide (e.g., `guides://python_style.md`)

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

## Configuration

Configure the server by creating a `config.toml` file or setting environment variables:

### GitHub Configuration (Recommended)

To load guides from a GitHub repository:

```toml
github_repo = "owner/repository"  # e.g., "delian/codeguide-mcp"
github_path = "guides"            # Path to guides directory in repo
github_branch = "main"            # Branch to fetch from
cache_dir = ".guides_cache"       # Local cache directory
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
- `GUIDES_CACHE_DIR` - Local cache directory (default: `.guides_cache`)
- `GUIDES_DIR` - Local directory containing guide files (default: `guides`)
- `GUIDES_LOG_LEVEL` - Logging level (default: `INFO`)

### Behavior

1. **Network available + GitHub configured**: Fetches guides from GitHub and caches them locally
2. **Network unavailable**: Uses local cache if available
3. **No cache available**: Falls back to local `guides_dir` if configured

## Adding Guides

### Using GitHub (Recommended)

If you've configured `github_repo`, simply add Markdown files to the specified directory in your GitHub repository. The server will automatically fetch and cache them.

### Using Local Directory

Add Markdown files to the `guides/` directory. Each file will be automatically available as a resource.

Example:
```bash
echo "# Python Style Guide\n\nUse PEP 8..." > guides/python_style.md
```

## Usage with MCP Clients

### Claude Desktop

Add to your `mcp.json`:

```json
{
  "mcpServers": {
    "coding-guides": {
      "command": "python",
      "args": ["-m", "main.py"]
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
