This is an MCP server that provides **mandatory** coding guidelines and best practices.
These guides define the coding standards that **must be followed** when writing, reviewing, or modifying code.
It provides a simple API for listing and retrieving the content of coding guidelines.

## When to Use

**Before writing or modifying code**, fetch the relevant guides for the project's languages, frameworks, and tools. Apply their standards as mandatory rules — not optional suggestions.

Typical triggers:
- Starting a new coding task or feature
- Reviewing or refactoring existing code
- Setting up a new project
- When the user asks about coding standards or best practices
- When there is an explicit request

## Usage Workflow

**IMPORTANT: Always start by listing available guides before selecting any specific guide.**

1. **List Guides**: Always begin by requesting `list_guides` to discover all available coding guides and their descriptions.

2. **Select Guides Based on Software Stack**: After reviewing the list, select **all** guides that are relevant to the current project's languages, frameworks, tools, and infrastructure. Guides are composable — for example, a Python REST API with Docker would need `python.md`, `rest.md`, and `dockerfile.md`.

3. **Fetch Guide Content**: For each selected guide, request its full content using `get_guide {guide_name}`.

4. **Apply Guidelines**: Follow the fetched guidelines as mandatory standards when writing, reviewing, or modifying code. If multiple guides apply, follow all of them. Treat the guides the same way as you would treat AGENTS.md file or Skills

## API Resources

These are MCP resource URIs (not HTTP URLs). Use your MCP resource fetching capability to access them.

- `guides://list`: List all available coding guides with brief descriptions.
  - **Output format**: Each line is `guide_name.md - brief description, tools`
  - **Always use this first** to discover available guides
  
- `guides://{guide_name}`: Get the full content of a specific coding guide.
  - Replace `{guide_name}` with the exact filename from the list (e.g., `python.md`)
  - **Example**: `guides://python.md`
  - **Use after** identifying relevant guides from the list

### List Output Format

The list output follows this format:
```
guide_name.md - brief description extracted from the guide
```
Each line contains:
- The guide filename (e.g., `python.md`, `docker-compose.md`)
- A dash separator
- A brief description (up to 150 characters) explaining what the guide covers

## Error Handling

- **Guide not found**: Verify the guide name matches exactly what `guides://list` returned (including the `.md` extension).
- **Empty list**: The server may have network issues fetching guides. Retry later or proceed with general best practices.
- **Multiple guides needed**: Fetch each guide separately — one `guides://{guide_name}` request per guide.
