import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from config import Settings

# Configure logging
logging.basicConfig(level=Settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("coding-guides")

GUIDES_DIR = Path(Settings.guides_dir)

@mcp.resource("guides://list")
def list_guides() -> str:
    """
    List all available coding guides.

    Returns:
        A string containing a list of available guides.
    """
    if not GUIDES_DIR.exists():
        return "Guides directory not found."

    guides = [f.name for f in GUIDES_DIR.glob("*.md")]
    return "\n".join(guides)

@mcp.resource("guides://{guide_name}")
def get_guide(guide_name: str) -> str:
    """
    Get the content of a specific coding guide.

    Args:
        guide_name: The name of the guide file (e.g., 'python_style.md').

    Returns:
        The content of the guide.
    """
    guide_path = GUIDES_DIR / guide_name
    if not guide_path.exists():
        return f"Guide '{guide_name}' not found."

    return guide_path.read_text()

def main() -> None:
    """Run the MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()
