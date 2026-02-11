import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional
import httpx
import base64
from cachetools import cached, TTLCache

from mcp.server.fastmcp import FastMCP

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

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@cached(cache=TTLCache(maxsize=1, ttl=600))
def check_network_available() -> bool:
    """
    Check if network access is available by attempting to connect to GitHub API.
    
    Returns:
        True if network is available, False otherwise.
    """
    if not GITHUB_REPO:
        return False
    
    try:
        # Try to connect to GitHub API with a short timeout
        with httpx.Client(timeout=3.0) as client:
            response = client.get("https://api.github.com", follow_redirects=True)
            return response.status_code == 200
    except Exception:
        return False


def fetch_github_directory_listing() -> Optional[list[dict]]:
    """
    Fetch the directory listing from GitHub API.
    
    Returns:
        List of file/directory information from GitHub API, or None if failed.
    """
    if not GITHUB_REPO:
        return None
    
    try:
        # GitHub API endpoint for repository contents
        # Format: https://api.github.com/repos/{owner}/{repo}/contents/{path}
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
        params = {"ref": GITHUB_BRANCH}
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params, follow_redirects=True)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch GitHub directory listing: {e}")
        return None

@cached(cache=TTLCache(maxsize=100, ttl=599))
def fetch_github_file_content(file_path: str) -> Optional[str]:
    """
    Fetch file content from GitHub API.
    
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
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params, follow_redirects=True)
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

@cached(cache=TTLCache(maxsize=100, ttl=599))
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


@cached(cache=TTLCache(maxsize=1, ttl=600))
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
            return brief_content
        
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
                            guides.append(f"{guide_name} - {brief}")
                    except Exception as e:
                        logger.warning(f"Error processing guide {guide_name}: {e}")
                        guides.append(f"{guide_name} - Error reading description")
            
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
                guides[guide_file.name] = f"{guide_file.name} - {brief}"
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
                guides[guide_file.name] = f"{guide_file.name} - {brief}"
            except Exception as e:
                logger.warning(f"Error reading guide {guide_file.name}: {e}")
    
    # Sort and format the guides
    guides = sorted(guides.values())
    
    if not guides:
        return "No guides found. Check network connection, local cache and local directory."
    
    return "\n".join(guides)


@mcp.resource("list_guides")
async def list_guides() -> str:
    """
    List all available coding guides.
    Fetches from GitHub if network is available, otherwise uses local cache.

    Returns:
        A string containing a list of available guides.
    """
    return get_guides_list()


@lru_cache(maxsize=32)
def cached_read_text(path: Path) -> str:
    return path.read_text()

@cached(cache=TTLCache(maxsize=100, ttl=599))
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


@mcp.resource("get_guide {guide_name: str}")
async def get_guide(guide_name: str) -> str:
    """
    Get the content of a specific coding guide.
    Fetches from GitHub if network is available, otherwise uses local cache.

    Args:
        guide_name: The name of the guide file (e.g., 'python_style.md').

    Returns:
        The content of the guide.
    """
    if not guide_name or "/" in guide_name or "\\" in guide_name or ".." in guide_name:
        return "Invalid guide name."

    content = fetch_guide_content(guide_name)
    if content:
        return content
    
    # Return MCP error if guide not found
    return f"ERROR: Guide '{guide_name}' not found! Retry later!"


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
