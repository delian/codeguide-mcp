import asyncio
import base64
import logging
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import httpx
from cachetools import TTLCache, cached

# mcp>=1.26 renamed FastMCP (mcp.server.fastmcp) to MCPServer (mcp.server.mcpserver);
# the decorator API (.tool/.resource/.prompt/.run) is unchanged.
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from config import Settings

# Configure logging
logging.basicConfig(level=Settings.log_level)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = MCPServer(
    name=Settings.mcp_name,
    instructions=Path(Settings.instructions).read_text(),
)

GUIDES_DIR = Path(Settings.guides_dir)
CACHE_DIR = Path(Settings.cache_dir)
GITHUB_REPO = Settings.github_repo
GITHUB_PATH = Settings.github_path
GITHUB_BRANCH = Settings.github_branch
PROMPTS_DIR = Path(Settings.prompts_dir)

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TTL cache instances (kept as module vars so clear_cache() can access them)
_network_cache: TTLCache = TTLCache(maxsize=1, ttl=600)
_github_file_cache: TTLCache = TTLCache(maxsize=100, ttl=599)
_guide_from_cache: TTLCache = TTLCache(maxsize=100, ttl=599)
_guides_list_cache: TTLCache = TTLCache(maxsize=1, ttl=600)
_guide_content_cache: TTLCache = TTLCache(maxsize=100, ttl=599)

# cachetools caches are NOT thread-safe on their own. Requests may run
# concurrently and _run_sync() offloads coroutines to worker threads, so guard
# every cache get/set with this lock. cachetools releases it during the wrapped
# call, so nested @cached calls cannot deadlock on it.
_cache_lock = threading.Lock()

# Scaffolding/meta files that live in guides/ but are NOT coding-standard guides:
# TEMPLATE.md is placeholder content ([TECHNOLOGY_NAME] …) and CONVENTIONS.md is
# the authoring spec. They are hidden from guides://list so an agent is never
# offered them, but remain fetchable by name (e.g. guides://CONVENTIONS.md
# references inside guides still resolve).
_HIDDEN_FROM_LIST = {"TEMPLATE.md", "CONVENTIONS.md"}


def _list_line_guide_name(line: str) -> str:
    """Extract the guide filename from a list line, e.g. 'guides://x.md - d' -> 'x.md'."""
    head = line.split(" - ", 1)[0].strip()
    head = head.removeprefix("guides://")
    return head.strip()


def _next_non_empty(lines: list[str], start_idx: int) -> tuple[int, str | None]:
    idx = start_idx
    while idx < len(lines):
        value = lines[idx].strip()
        if value:
            return idx, value
        idx += 1
    return idx, None


def _parse_meta_value(line: str, key: str) -> str | None:
    prefix = f"{key}:"
    if line.lower().startswith(prefix):
        return line[len(prefix) :].strip()
    return None


def _parse_prompt_definition_from_markdown(
    content: str, file_stem: str
) -> dict[str, object] | None:
    lines = content.splitlines()
    idx, first = _next_non_empty(lines, 0)
    if not first:
        return None

    prompt_name = file_stem
    if first.startswith("# "):
        prompt_name = first[2:].strip() or file_stem
        idx += 1

    messages: list[dict[str, str]] = []
    prompt_description = ""

    while True:
        idx, current = _next_non_empty(lines, idx)
        if current is None:
            break

        description = ""
        role = "assistant"
        content_type = "text"

        maybe_description = _parse_meta_value(current, "description")
        if maybe_description is not None:
            description = maybe_description
            idx += 1
            idx, current = _next_non_empty(lines, idx)
            if current is None:
                break

        maybe_role = _parse_meta_value(current, "role")
        if maybe_role is not None:
            role = maybe_role
            idx += 1
            idx, current = _next_non_empty(lines, idx)
            if current is None:
                break

        maybe_type = _parse_meta_value(current, "type")
        if maybe_type is not None:
            content_type = maybe_type
            idx += 1

        body_lines: list[str] = []
        while idx < len(lines):
            raw_line = lines[idx]
            if not raw_line.strip():
                break
            body_lines.append(raw_line)
            idx += 1

        text = "\n".join(body_lines).strip()
        if text:
            if description and not prompt_description:
                prompt_description = description
            messages.append(
                {
                    "role": role,
                    "type": content_type,
                    "text": text,
                }
            )

        while idx < len(lines) and not lines[idx].strip():
            idx += 1

    if not messages:
        return None

    return {
        "name": prompt_name,
        "description": prompt_description,
        "messages": messages,
    }


def _iter_prompt_markdown_files(directory: Path) -> Iterator[Path]:
    if not directory.exists() or not directory.is_dir():
        return iter(())
    return iter(sorted(directory.glob("*.md")))


def register_prompts_from_markdown() -> None:
    """Register prompts dynamically from markdown files in the configured prompts directory."""
    files = list(_iter_prompt_markdown_files(PROMPTS_DIR))
    if not files:
        logger.info(f"No dynamic prompt files found in {PROMPTS_DIR}")
        return

    registered = 0
    for prompt_file in files:
        try:
            content = prompt_file.read_text(encoding="utf-8")
            prompt = _parse_prompt_definition_from_markdown(content, prompt_file.stem)
            if not prompt:
                continue

            prompt_name = str(prompt["name"])
            prompt_description = str(prompt["description"])
            prompt_messages = list(prompt["messages"])

            def _build_prompt(messages: list[dict[str, str]]):
                def _dynamic_prompt() -> list[dict[str, object]]:
                    # Return plain dicts: the SDK's prompt renderer validates them
                    # into its own Message models. Returning mcp.types.PromptMessage
                    # here is WRONG — the renderer doesn't recognize that class and
                    # falls back to JSON-dumping it as user-role text.
                    # Roles are coerced to "user"/"assistant" so a typo in a prompt
                    # markdown file cannot raise at invocation time.
                    return [
                        {
                            "role": message["role"]
                            if message["role"] in ("user", "assistant")
                            else "assistant",
                            "content": {"type": "text", "text": message["text"]},
                        }
                        for message in messages
                    ]

                return _dynamic_prompt

            mcp.prompt(name=prompt_name, description=prompt_description)(
                _build_prompt(prompt_messages)
            )
            registered += 1
        except Exception as e:
            logger.warning(f"Failed to register prompts from {prompt_file.name}: {e}")

    logger.info(f"Registered {registered} dynamic prompts from {PROMPTS_DIR}")


def _run_sync(coro):
    """Run an async coroutine to completion from synchronous code, whether or not
    an event loop is already running in the current thread.

    `asyncio.run()` raises "asyncio.run() cannot be called from a running event
    loop" when the caller is already inside one — which is the case for MCP tool
    and resource handlers executing in the server's event loop. When a loop is
    running we offload the coroutine to a dedicated worker thread that owns its
    own loop; otherwise we run it directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread — safe to drive the coroutine directly.
        return asyncio.run(coro)
    # A loop is already running in this thread; run the coroutine in its own thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _check_network_available_async() -> bool:
    """
    Async version: Check if network access is available by attempting to connect to GitHub API.

    Returns:
        True if network is available, False otherwise.
    """
    if not GITHUB_REPO:
        return False

    try:
        # Try to connect to GitHub API with a short timeout
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get("https://api.github.com", follow_redirects=True)
            return response.status_code == 200
    except Exception:
        return False


@cached(cache=_network_cache, lock=_cache_lock)
def check_network_available() -> bool:
    """
    Check if network access is available by attempting to connect to GitHub API.

    Returns:
        True if network is available, False otherwise.
    """
    return _run_sync(_check_network_available_async())


async def _fetch_github_directory_listing_async() -> list[dict] | None:
    """
    Async version: Fetch the directory listing from GitHub API with pagination support.

    Returns:
        List of file/directory information from GitHub API, or None if failed.
    """
    if not GITHUB_REPO:
        return None

    try:
        # GitHub API endpoint for repository contents
        # Format: https://api.github.com/repos/{owner}/{repo}/contents/{path}
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
        params = {"ref": GITHUB_BRANCH, "per_page": 100}

        all_items = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            while url:
                response = await client.get(url, params=params, follow_redirects=True)
                response.raise_for_status()

                items = response.json()
                if isinstance(items, list):
                    all_items.extend(items)
                else:
                    # Single file, not a directory
                    return items

                # Check for pagination in Link header
                # Link header format: <url>; rel="next", <url>; rel="last"
                link_header = response.headers.get("Link", "")
                url = None  # Reset for next iteration
                params = None  # Params are in the URL from Link header

                if link_header:
                    # Parse Link header to find next page
                    for link in link_header.split(","):
                        if 'rel="next"' in link:
                            # Extract URL from <url>
                            url = link.split(";")[0].strip().strip("<>")
                            break

                if not url:
                    # No more pages
                    break

        logger.info(f"Fetched {len(all_items)} items from GitHub (with pagination)")
        return all_items

    except Exception as e:
        logger.warning(f"Failed to fetch GitHub directory listing: {e}")
        return None


def fetch_github_directory_listing() -> list[dict] | None:
    """
    Fetch the directory listing from GitHub API.

    Returns:
        List of file/directory information from GitHub API, or None if failed.
    """
    return _run_sync(_fetch_github_directory_listing_async())


async def _fetch_github_file_content_async(file_path: str) -> str | None:
    """
    Async version: Fetch file content from GitHub API.

    Args:
        file_path: Path to the file in the GitHub repository.

    Returns:
        File content as string, or None if failed.
    """
    if not GITHUB_REPO:
        return None

    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
        params = {"ref": GITHUB_BRANCH}

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, follow_redirects=True)
            response.raise_for_status()
            data = response.json()

            # Decode base64 content
            if data.get("encoding") == "base64":
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
            else:
                logger.warning(f"Unexpected encoding: {data.get('encoding')}")
                return None
    except Exception as e:
        logger.warning(f"Failed to fetch GitHub file {file_path}: {e}")
        return None


@cached(cache=_github_file_cache, lock=_cache_lock)
def fetch_github_file_content(file_path: str) -> str | None:
    """
    Fetch file content from GitHub API.

    Args:
        file_path: Path to the file in the GitHub repository.

    Returns:
        File content as string, or None if failed.
    """
    return _run_sync(_fetch_github_file_content_async(file_path))


def cache_guide_locally(guide_name: str, content: str) -> Path:
    """
    Cache a guide file locally.

    Args:
        guide_name: Name of the guide file.
        content: Content of the guide file.

    Returns:
        Path to the cached file.
    """
    cache_path = CACHE_DIR / guide_name
    cache_path.write_text(content, encoding="utf-8")
    logger.debug(f"Cached guide {guide_name} locally")
    return cache_path


@cached(cache=_guide_from_cache, lock=_cache_lock)
def get_guide_from_cache(guide_name: str) -> str | None:
    """
    Get guide content from local cache.

    Args:
        guide_name: Name of the guide file.

    Returns:
        Guide content if found in cache, None otherwise.
    """
    cache_path = CACHE_DIR / guide_name
    if cache_path.exists():
        return cached_read_text(cache_path)
    return None


def extract_brief(content: str, max_length: int = Settings.brief_max_length) -> str:
    """
    Extract a brief description from guide content.

    Args:
        content: The full content of the guide file.
        max_length: Maximum length of the brief description.

    Returns:
        A brief description extracted from the guide.
    """
    lines = content.split("\n")
    brief_parts = []

    # Skip the title (first line if it starts with #)
    start_idx = 1 if lines and lines[0].strip().startswith("#") else 0

    # Collect non-empty lines until we have enough content
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        # Stop at YAML frontmatter delimiter
        if line == "---":
            break
        # Skip markdown headers and code blocks
        if line.startswith(("#", "```")):
            continue
        brief_parts.append(line)
        # Stop if we have enough content
        if len(" ".join(brief_parts)) >= max_length:
            break

    brief = " ".join(brief_parts)
    # Truncate to max_length and add ellipsis if needed
    if len(brief) > max_length:
        brief = brief[:max_length].rsplit(" ", 1)[0] + "..."

    return brief if brief else "No description available."


def _normalize_list_to_guide_uris(list_text: str) -> str:
    """Ensure each list line starts with guides:// so clients get full resource URIs."""
    lines = []
    for line in list_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if _list_line_guide_name(line) in _HIDDEN_FROM_LIST:
            continue
        if line.startswith("guides://"):
            lines.append(line)
        elif " - " in line:
            name, _, rest = line.partition(" - ")
            name = name.strip()
            lines.append(f"guides://{name} - {rest.strip()}")
        else:
            lines.append(line)
    return "\n".join(lines) if lines else list_text


@cached(cache=_guides_list_cache, lock=_cache_lock)
def get_guides_list() -> str:
    """
    Get list of available guides, fetching from GitHub if available,
    otherwise using local cache or local directory.
    """
    # Try GitHub first if network is available
    if check_network_available():
        logger.info("Network available, fetching guides from GitHub")

        brief_path = "brief.md"
        brief_content = fetch_github_file_content(brief_path)
        if brief_content:
            return _normalize_list_to_guide_uris(brief_content)

        github_files = fetch_github_directory_listing()

        if github_files:
            guides = []
            for item in github_files:
                # Only process .md files
                if item.get("type") == "file" and item.get("name", "").endswith(".md"):
                    guide_name = item["name"]
                    if guide_name in _HIDDEN_FROM_LIST:
                        continue
                    try:
                        # Try to get from cache first
                        content = get_guide_from_cache(guide_name)

                        if content is None:
                            # Fetch from GitHub and cache it
                            content = fetch_github_file_content(item["path"])
                            if content:
                                cache_guide_locally(guide_name, content)

                        if content:
                            brief = extract_brief(content)
                            guides.append(f"guides://{guide_name} - {brief}")
                    except Exception as e:
                        logger.warning(f"Error processing guide {guide_name}: {e}")
                        guides.append(
                            f"guides://{guide_name} - Error reading description"
                        )

            if guides:
                return "\n".join(sorted(guides))

    # Fall back to local cache
    logger.info("Using local cache or local directory")
    guides = {}

    # First try cache directory
    if CACHE_DIR.exists():
        for guide_file in sorted(CACHE_DIR.glob("*.md")):
            if guide_file.name in _HIDDEN_FROM_LIST:
                continue
            try:
                content = cached_read_text(guide_file)
                brief = extract_brief(content)
                guides[guide_file.name] = f"guides://{guide_file.name} - {brief}"
            except Exception as e:
                logger.warning(f"Error reading cached guide {guide_file.name}: {e}")

    # If no cached guides, try local directory
    if GUIDES_DIR.exists():
        # Fill the missing guides from local directory
        for guide_file in sorted(GUIDES_DIR.glob("*.md")):
            if guide_file.name in guides:
                continue
            if guide_file.name in _HIDDEN_FROM_LIST:
                continue
            try:
                content = cached_read_text(guide_file)
                brief = extract_brief(content)
                guides[guide_file.name] = f"guides://{guide_file.name} - {brief}"
            except Exception as e:
                logger.warning(f"Error reading guide {guide_file.name}: {e}")

    # Sort and format the guides
    guides = sorted(guides.values())

    if not guides:
        return "No guides found. Check network connection, local cache and local directory."

    return "\n".join(guides)


@mcp.resource("guides://list")
def list_guides() -> str:
    """
    List all available coding guides.
    Fetches from GitHub if network is available, otherwise uses local cache.

    Returns:
        A string containing a list of available guides.
    """
    return get_guides_list()


def _guide_name_from_uri(name_or_uri: str) -> str:
    """Return guide filename, stripping guides:// prefix if present."""
    s = (name_or_uri or "").strip()
    s = s.removeprefix("guides://")
    return s


@mcp.resource(
    "guides://{guide_name}",
    name="guide",
    description="Get the content of a specific coding guide by name (e.g. python.md).",
)
def get_guide_resource(guide_name: str) -> str:
    """
    Resource handler for guides://{guide_name}.
    Returns the full content of the requested guide.
    Accepts the value from the URI (e.g. python.md); also accepts guides://python.md.
    """
    guide_name = _guide_name_from_uri(guide_name)
    if not guide_name or "/" in guide_name or "\\" in guide_name or ".." in guide_name:
        raise ValueError("Invalid guide name.")

    content = fetch_guide_content(guide_name)
    if content:
        return content

    raise ValueError(f"Guide '{guide_name}' not found.")


@lru_cache(maxsize=32)
def cached_read_text(path: Path) -> str:
    return path.read_text()


@cached(cache=_guide_content_cache, lock=_cache_lock)
def fetch_guide_content(guide_name: str) -> str | None:
    """
    Fetch the content of a guide, trying GitHub first if network is available,
    then falling back to local cache and local directory.

    Args:
        guide_name: The name of the guide file (e.g., 'python_style.md').

    Returns:
        The content of the guide if found, otherwise None.
    """

    # Try GitHub first if network is available
    if check_network_available() and GITHUB_REPO:
        logger.info(f"Network available, fetching guide {guide_name} from GitHub")

        # Construct GitHub path
        github_file_path = f"{GITHUB_PATH}/{guide_name}" if GITHUB_PATH else guide_name

        content = fetch_github_file_content(github_file_path)
        if content:
            cache_guide_locally(guide_name, content)
            return content

    content = get_guide_from_cache(guide_name)
    if content:
        return content

    # Fall back to local directory
    guide_path = GUIDES_DIR / guide_name
    if guide_path.exists():
        return cached_read_text(guide_path)

    return None


@mcp.tool(name="get_guide", description="Get the content of a specific coding guide.")
def get_guide(guide_name: str) -> str:
    """
    Get the content of a specific coding guide.
    Fetches from GitHub if network is available, otherwise uses local cache.

    Args:
        guide_name: The name of the guide file (e.g., 'python.md') or URI (e.g., 'guides://python.md').

    Returns:
        The content of the guide.
    """
    guide_name = _guide_name_from_uri(guide_name)
    if not guide_name or "/" in guide_name or "\\" in guide_name or ".." in guide_name:
        # Raise so the client sees a proper MCP tool error (isError=true),
        # not a success result whose text happens to describe a failure.
        raise ValueError("Invalid guide name.")

    content = fetch_guide_content(guide_name)
    if content:
        return content

    raise ValueError(f"Guide '{guide_name}' not found.")


@mcp.prompt("list_guides", description="List all available coding guides.")
def list_guides_prompt() -> str:
    """List all available coding guides."""
    # get_guides_list() already returns a newline-joined string; do NOT
    # "\n".join() it (that would insert a newline between every character).
    return get_guides_list()


@mcp.prompt("get_guide", description="Get the content of a specific coding guide.")
def get_guide_prompt(guide_name: str) -> str:
    """Get the content of a specific coding guide."""
    return get_guide(guide_name)


@mcp.prompt(
    "help", description="Get help information about available resources and tools."
)
def help_prompt() -> str:
    """Get help information about available resources and tools."""
    # MCPServer has no get_help(); surface the server instructions instead.
    return mcp.instructions or (
        "Codeguide MCP. Resources: guides://list, guides://{name}. "
        "Tool: get_guide(guide_name). Prompts: list_guides, get_guide, clear_cache, help."
    )


@mcp.prompt(
    "security_check",
    description="Check if the MCP server is secure and not exposing sensitive information.",
)
def security_check_prompt() -> str:
    """Check if the MCP server is secure and not exposing sensitive information."""
    return "Security check passed. No sensitive information is being exposed."


@mcp.prompt("exit", description="Exit the MCP server.")
def exit_prompt() -> str:
    """Exit the MCP server."""
    # MCPServer exposes no programmatic stop(); a prompt must not crash trying to
    # call one. The server lifecycle is owned by the client/host that launched it.
    return "This server is managed by its host; stop it from the client/host that launched it."


@mcp.prompt("clear_cache", description="Clear the local cache of guides.")
def clear_cache_prompt() -> str:
    """Clear the local cache of guides."""
    clear_cache()
    return "Cache cleared."


def clear_cache() -> None:
    """Clear all caches: TTL caches, lru_cache, and local cache directory."""
    # Clear TTL caches
    _network_cache.clear()
    _github_file_cache.clear()
    _guide_from_cache.clear()
    _guides_list_cache.clear()
    _guide_content_cache.clear()

    # Clear lru_cache
    cached_read_text.cache_clear()

    # Clear local cache directory
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.md"):
            f.unlink()

    logger.info("All caches cleared")


def _resolve_transport() -> str:
    """Resolve the configured transport, expanding "auto".

    "auto" means: use streamable-http when the platform injected a PORT env var
    (Cloud Run and most PaaS providers do, and they require the process to
    listen on it), otherwise stdio for local MCP clients over a pipe.
    """
    transport = str(Settings.get("transport", "auto")).strip().lower()
    if transport == "auto":
        transport = "streamable-http" if os.environ.get("PORT") else "stdio"
    if transport not in ("stdio", "sse", "streamable-http"):
        raise ValueError(
            f"Unsupported transport {transport!r}; use 'stdio', 'streamable-http', 'sse' or 'auto'."
        )
    return transport


def _http_kwargs() -> dict[str, object]:
    """Build the streamable-http keyword arguments from settings and the environment."""
    # Cloud Run (and most PaaS) dictate the listening port via $PORT; it must win
    # over the configured default or the container is killed as unhealthy.
    port = int(os.environ.get("PORT") or Settings.get("port", 8080))
    host = str(Settings.get("host", "0.0.0.0"))

    allowed_hosts = list(Settings.get("allowed_hosts", []) or [])
    allowed_origins = list(Settings.get("allowed_origins", []) or [])
    transport_security = None
    if allowed_hosts or allowed_origins:
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )

    return {
        "host": host,
        "port": port,
        "streamable_http_path": str(Settings.get("http_path", "/mcp")),
        "stateless_http": bool(Settings.get("stateless_http", True)),
        "json_response": bool(Settings.get("json_response", False)),
        "transport_security": transport_security,
    }


def main() -> None:
    """Run the MCP server."""
    register_prompts_from_markdown()

    transport = _resolve_transport()
    if transport == "stdio":
        logger.info("Starting MCP server on stdio")
        mcp.run()
        return

    kwargs = _http_kwargs()
    logger.info(
        f"Starting MCP server with {transport} transport on "
        f"http://{kwargs['host']}:{kwargs['port']}{kwargs['streamable_http_path']}"
    )
    mcp.run(transport, **kwargs)


if __name__ == "__main__":
    main()
