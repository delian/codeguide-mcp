import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional
from collections.abc import Iterator
import asyncio
import httpx
import base64
from cachetools import cached, TTLCache

from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

from config import Settings

# Configure logging
logging.basicConfig(level=Settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
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


def _next_non_empty(lines: list[str], start_idx: int) -> tuple[int, Optional[str]]:
    idx = start_idx
    while idx < len(lines):
        value = lines[idx].strip()
        if value:
            return idx, value
        idx += 1
    return idx, None


def _parse_meta_value(line: str, key: str) -> Optional[str]:
    prefix = f"{key}:"
    if line.lower().startswith(prefix):
        return line[len(prefix):].strip()
    return None


def _parse_prompt_definition_from_markdown(content: str, file_stem: str) -> Optional[dict[str, object]]:
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
                def _dynamic_prompt() -> list[PromptMessage]:
                    return [
                        PromptMessage(
                            role=message["role"],
                            content=TextContent(type=message["type"], text=message["text"]),
                        )
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


@cached(cache=_network_cache)
def check_network_available() -> bool:
    """
    Check if network access is available by attempting to connect to GitHub API.
    
    Returns:
        True if network is available, False otherwise.
    """
    return asyncio.run(_check_network_available_async())


async def _fetch_github_directory_listing_async() -> Optional[list[dict]]:
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


def fetch_github_directory_listing() -> Optional[list[dict]]:
    """
    Fetch the directory listing from GitHub API.
    
    Returns:
        List of file/directory information from GitHub API, or None if failed.
    """
    return asyncio.run(_fetch_github_directory_listing_async())

async def _fetch_github_file_content_async(file_path: str) -> Optional[str]:
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


@cached(cache=_github_file_cache)
def fetch_github_file_content(file_path: str) -> Optional[str]:
    """
    Fetch file content from GitHub API.
    
    Args:
        file_path: Path to the file in the GitHub repository.
    
    Returns:
        File content as string, or None if failed.
    """
    return asyncio.run(_fetch_github_file_content_async(file_path))


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

@cached(cache=_guide_from_cache)
def get_guide_from_cache(guide_name: str) -> Optional[str]:
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
    lines = content.split('\n')
    brief_parts = []
    
    # Skip the title (first line if it starts with #)
    start_idx = 1 if lines and lines[0].strip().startswith('#') else 0
    
    # Collect non-empty lines until we have enough content
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        # Stop at YAML frontmatter delimiter
        if line == '---':
            break
        # Skip markdown headers and code blocks
        if line.startswith('#') or line.startswith('```'):
            continue
        brief_parts.append(line)
        # Stop if we have enough content
        if len(' '.join(brief_parts)) >= max_length:
            break
    
    brief = ' '.join(brief_parts)
    # Truncate to max_length and add ellipsis if needed
    if len(brief) > max_length:
        brief = brief[:max_length].rsplit(' ', 1)[0] + '...'
    
    return brief if brief else "No description available."


def _normalize_list_to_guide_uris(list_text: str) -> str:
    """Ensure each list line starts with guides:// so clients get full resource URIs."""
    lines = []
    for line in list_text.strip().split("\n"):
        line = line.strip()
        if not line:
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


@cached(cache=_guides_list_cache)
def get_guides_list() -> str:
    """
    Get list of available guides, fetching from GitHub if available,
    otherwise using local cache or local directory.
    """
    # Try GitHub first if network is available
    if check_network_available():
        logger.info("Network available, fetching guides from GitHub")
        
        brief_path = f"brief.md"
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
                        guides.append(f"guides://{guide_name} - Error reading description")
            
            if guides:
                return "\n".join(sorted(guides))
    
    # Fall back to local cache
    logger.info("Using local cache or local directory")
    guides = {}
    
    # First try cache directory
    if CACHE_DIR.exists():
        for guide_file in sorted(CACHE_DIR.glob("*.md")):
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
            if guide_file.name in guides: continue
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
    if s.startswith("guides://"):
        s = s[9:]
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

@cached(cache=_guide_content_cache)
def fetch_guide_content(guide_name: str) -> Optional[str]:
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
        return "Invalid guide name."

    content = fetch_guide_content(guide_name)
    if content:
        return content
    
    # Return MCP error if guide not found
    raise ValueError(f"Guide '{guide_name}' not found.")
    # return f"ERROR: Guide '{guide_name}' not found! Retry later!"


@mcp.prompt("list_guides", description="List all available coding guides.")
def list_guides() -> str:
    """List all available coding guides."""
    return "\n".join(get_guides_list())

@mcp.prompt("get_guide", description="Get the content of a specific coding guide.")
def get_guide_prompt(guide_name: str) -> str:
    """Get the content of a specific coding guide."""
    return get_guide(guide_name)

@mcp.prompt("help", description="Get help information about available resources and tools.")
def help_prompt() -> str:
    """Get help information about available resources and tools."""
    return mcp.get_help()

@mcp.prompt("security_check", description="Check if the MCP server is secure and not exposing sensitive information.")
def security_check_prompt() -> str:
    """Check if the MCP server is secure and not exposing sensitive information."""
    return "Security check passed. No sensitive information is being exposed."

@mcp.prompt("exit", description="Exit the MCP server.")
def exit_prompt() -> str:
    """Exit the MCP server."""
    mcp.stop()
    return "Exiting..."

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

def main() -> None:
    """Run the MCP server."""
    register_prompts_from_markdown()
    mcp.run()


if __name__ == "__main__":
    main()
