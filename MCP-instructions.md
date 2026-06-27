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

### Step 3: Follow references (MANDATORY)

Guides are deliberately small: they do **not** duplicate shared content. Instead they **reference** other guides that own a cross-cutting concern (TDD, hexagonal architecture, secure coding, error handling, logging, configuration, etc.). You **MUST** follow those references — a guide is incomplete without them.

When a guide you fetched contains a reference marker, act on it:

- **📎 REQUIRED** — You **MUST** fetch and apply the linked guide **before writing any code**. It is a hard prerequisite; the referencing guide assumes its rules and does not restate them. Skipping it means you are violating mandatory standards.
- **📎 RECOMMENDED** — You **MUST** fetch and apply the linked guide **if the current task touches that concern** (e.g. fetch `logging.md` when the task involves logging).
- **📎 SEE ALSO** — Optional; fetch for additional depth when useful.

Reference targets appear as `guides://<name>.md` links (e.g. `guides://tdd.md`). Resolve each one with **ReadResource** or the **`get_guide`** tool, exactly as in Step 2. Follow references **transitively** — if a referenced guide has its own REQUIRED references, fetch those too. Fetch each guide only once per task and reuse it.

A guide's machine-readable prerequisites are also listed in its YAML frontmatter under `requires:` (must fetch) and `recommends:` (fetch when relevant); the `name` values map to `<name>.md`. Use these to plan which guides to pull up front.

### Step 4: Apply the guidelines

Follow the fetched guidelines — and every guide they REQUIRED/RECOMMENDED you to fetch — as mandatory standards when writing, reviewing, or modifying code. If multiple guides apply, follow all of them. Each guide also contains an auditable requirements table (IDs like `PY-TST-01`) with verification commands and pass/fail gates: satisfy every gate before presenting code. Treat the guides like project AGENTS.md or Skills.

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
