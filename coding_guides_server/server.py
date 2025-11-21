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


@lru_cache(maxsize=1)
def get_guides_list() -> str:
    if not GUIDES_DIR.exists():
        return "Guides directory not found."
    return "\n".join([f.name for f in GUIDES_DIR.glob("*.md")])


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
