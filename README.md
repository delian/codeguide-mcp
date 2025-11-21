# Coding Guides MCP Server

A Model Context Protocol (MCP) server that provides access to coding guides and best practices for AI assistants like Claude and GitHub Copilot.

## What is this?

This MCP server exposes coding guidelines and style guides as resources that can be accessed by MCP clients. It's designed to extend or replace `AGENTS.md` files by providing a structured way to serve coding practices and guidelines to AI assistants during development.

## Features

- **Resource-based API**: Exposes coding guides through MCP resources
- **Simple file-based storage**: Guides are stored as Markdown files in the `guides/` directory
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

Configure the server by creating a `settings.toml` file or setting environment variables:

```toml
guides_dir = "guides"
log_level = "INFO"
```

Environment variables:
- `GUIDES_DIR` - Directory containing guide files (default: `guides`)
- `LOG_LEVEL` - Logging level (default: `INFO`)

## Adding Guides

Simply add Markdown files to the `guides/` directory. Each file will be automatically available as a resource.

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
      "args": ["-m", "coding_guides_server"]
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
