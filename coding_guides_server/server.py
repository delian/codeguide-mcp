import logging
from pathlib import Path
from functools import lru_cache

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


@lru_cache(maxsize=1)
def get_guides_list() -> str:
    if not GUIDES_DIR.exists():
        return "Guides directory not found."
    
    guides = []
    for guide_file in sorted(GUIDES_DIR.glob("*.md")):
        try:
            content = cached_read_text(guide_file)
            brief = extract_brief(content)
            guides.append(f"{guide_file.name} - {brief}")
        except Exception as e:
            logger.warning(f"Error reading guide {guide_file.name}: {e}")
            guides.append(f"{guide_file.name} - Error reading description")
    
    return "\n".join(guides)


@mcp.resource("guides://list")
async def list_guides() -> str:
    """
    List all available coding guides.

    Returns:
        A string containing a list of available guides.
    """
    return get_guides_list()


@lru_cache(maxsize=32)
def cached_read_text(path: Path) -> str:
    return path.read_text()


@mcp.resource("guides://{guide_name}")
async def get_guide(guide_name: str) -> str:
    """
    Get the content of a specific coding guide.

    Args:
        guide_name: The name of the guide file (e.g., 'python_style.md').

    Returns:
        The content of the guide.
    """
    if not guide_name or "/" in guide_name or "\\" in guide_name or ".." in guide_name:
        return "Invalid guide name."
    guide_path = GUIDES_DIR / guide_name
    if not guide_path.exists():
        return f"Guide '{guide_name}' not found."

    return cached_read_text(guide_path)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
