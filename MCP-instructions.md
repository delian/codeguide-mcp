# Codeguide MCP Server

This MCP server provides **mandatory** coding guidelines and best practices. The guides define coding standards that **must be followed** when writing, reviewing, or modifying code.

## When to Use

**Before writing or modifying code**, fetch the relevant guides for the project's languages, frameworks, and tools. Apply their standards as mandatory rules — not optional suggestions.

Typical triggers:
- Starting a new coding task or feature
- Reviewing or refactoring existing code
- Setting up a new project
- When the user asks about coding standards or best practices
- When there is an explicit request

---

## How to Use This Server (MCP Client)

The server exposes **resources** and **one tool**. Use standard MCP operations: **ListResources**, **ListResourceTemplates**, **ReadResource**, **ListTools**, and **CallTool**.

### Step 1: Discover guides (always do this first)

- **Read the resource** with URI **`guides://list`**.
  - This returns a text list of all available guides.
  - Each line has the form: `guide_name.md - brief description`
  - Use this list to choose which guides apply to the project (e.g. `python.md`, `rest.md`, `dockerfile.md`).

Alternatively, you can use **ListResources** to see that `guides://list` is available, and **ListResourceTemplates** to see that `guides://{guide_name}` is available.

### Step 2: Get full content for each guide you need

For each selected guide, get its full content using **either**:

- **ReadResource** with URI **`guides://{guide_name}`**  
  Example: read resource **`guides://python.md`** (use the exact filename from the list, including `.md`).

- **CallTool** with tool name **`get_guide`** and arguments **`{"guide_name": "guide_name.md"}`**  
  Example: call tool **`get_guide`** with **`{"guide_name": "python.md"}`**.

Both return the full markdown content of the guide. Use one or the other; no need to use both for the same guide.

### Step 3: Apply the guidelines

Follow the fetched guidelines as mandatory standards when writing, reviewing, or modifying code. If multiple guides apply, follow all of them. Treat the guides like project AGENTS.md or Skills.

---

## API Reference

### Resources

| URI | Description |
|-----|-------------|
| **`guides://list`** | Single resource. Read it to get a plain-text list of all guides. Each line: **`guides://guide_name.md - brief description`**. The first token is the resource URI you can pass to ReadResource. |
| **`guides://{guide_name}`** | Resource template. Read a specific guide by URI, e.g. **`guides://python.md`**, **`guides://rest.md`**. Use the exact URI from the list (first token of each line). |

- Use **ReadResource** with the URI to get content. The list from `guides://list` returns full URIs (e.g. `guides://python.md`) so you can use each line’s first token directly as the ReadResource URI.
- Guide names/URIs must be exact (e.g. `guides://python.md` not `Python.md`). No path segments: only a single filename in the URI, like `guides://docker-compose.md`.

### Tools

| Name | Description | Arguments |
|------|-------------|-----------|
| **`get_guide`** | Returns the full content of a specific coding guide. | **`guide_name`** (string, required): exact filename (e.g. `"python.md"`) or full URI (e.g. `"guides://python.md"`). Either form is accepted. |

- Use **CallTool** with name **`get_guide`** and arguments **`{"guide_name": "guides://python.md"}`** or **`{"guide_name": "python.md"}`**.

---

## List output format (`guides://list`)

The list is plain text, one guide per line:

```
guides://guide_name.md - brief description extracted from the guide
```

- **guides://guide_name.md**: full resource URI. Use this string as the URI for **ReadResource** to fetch the guide, or pass it (or just `guide_name.md`) to the **get_guide** tool.
- **brief description**: short summary (up to ~250 characters) of what the guide covers.

---

## Error handling

- **Guide not found**: Use the exact name from `guides://list` (including `.md`). Names are case-sensitive.
- **Invalid guide name**: Do not use path segments, backslashes, or `..`. Use only a single filename, e.g. `python.md`.
- **Empty list from `guides://list`**: The server may be offline or unable to reach the guide source. Retry later or proceed with general best practices.
- **Multiple guides**: Fetch each guide separately (one ReadResource or one CallTool per guide).
