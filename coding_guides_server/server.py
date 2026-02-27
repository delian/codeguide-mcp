import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional
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

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TTL cache instances (kept as module vars so clear_cache() can access them)
_network_cache: TTLCache = TTLCache(maxsize=1, ttl=600)
_github_file_cache: TTLCache = TTLCache(maxsize=100, ttl=599)
_guide_from_cache: TTLCache = TTLCache(maxsize=100, ttl=599)
_guides_list_cache: TTLCache = TTLCache(maxsize=1, ttl=600)
_guide_content_cache: TTLCache = TTLCache(maxsize=100, ttl=599)

@cached(cache=_network_cache)
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

@cached(cache=_github_file_cache)
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

@mcp.prompt("check_security", description="Check if the code complies with the security standards.")
def check_security_prompt() -> list[PromptMessage]:
    """Check if the code complies with the security standards."""
    return [
        PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text="You are an expert Security Engineer. Follow these steps:\n"
                     "1. Analyze the source code for security vulnerabilities.\n"
                     "2. Check dependencies with the respective commands like npm audit --fix, pip-audit, etc.\n"
                     "3. Use the 'search_documentation' tool for the error code.\n"
                     "4. Propose three potential fixes.\n"
                     "5. For each proposed fix, provide a brief explanation of how it addresses the issue and any potential trade-offs.\n"
                     "6. Make sure all fixes are not breaking the build, compilation and operation of the code and application.\n"
                     "7. If the error is related to a specific coding guide, reference the relevant guide and explain how it applies to the issue and try to fix it.\n"
            )
        ),
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Here is the code to check for security compliance:\n"
            )
        )
    ]

@mcp.prompt("check_code_compliance", description="Check if the code complies with the coding standards.")
def check_code_compliance_prompt() -> list[PromptMessage]:
    """Check if the code complies with the coding standards."""
    return [
        PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text="You are an expert Senior Engineer. Follow these steps:\n"
                     "1. Analyze the source code, architecture and infrastructure stack.\n"
                     "2. Analyze the best coding practices, styles and patterns according to this MCP codeguide mcp respective to every stack, API, architecture, documentation, tests, security, etc.\n"
                     "3. Check and consult with other best practices and patterns over internet or Context7 MCP and other resources\n"
                     "4. Use the 'search_documentation' tool for the error code.\n"
                     "5. Check deviations and propose TODO steps and improvements to make the code compliant with the best practices and patterns and coding guides.\n"
                     "6. Track any TODO steps and improvements.\n"
                     "7. Make sure all fixes are not breaking the build, compilation and operation of the code and application, all the code passes lints, could be build, unit tests are passing.\n"
                     "8. If the error is related to a specific coding guide, reference the relevant guide and explain how it applies to the issue and try to fix it.\n"
            )
        ),
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Here is the check for coding compliance:\n"
            )
        )
    ]

@mcp.prompt("check_code_quality", description="Check if the code complies with the quality standards.")
def check_code_quality_prompt() -> list[PromptMessage]:
    """Check if the code complies with the quality standards."""
    return [
        PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text="You are an expert Senior Engineer. Follow these steps:\n"
                     "1. Analyze the source code, architecture and infrastructure stack.\n"
                     "2. Analyze the best coding practices, styles and patterns according to this MCP codeguide mcp respective to every stack, API, architecture, documentation, tests, security, etc.\n"
                     "3. Check and consult with other best practices and patterns over internet or Context7 MCP and other resources\n"
                     "4. Use the 'search_documentation' tool for the error code.\n"
                     "5. Check deviations and propose TODO steps and improvements to make the code compliant with the best practices and patterns and coding guides.\n"
                     "6. Track any TODO steps and improvements.\n"
                     "7. Make sure all fixes are not breaking the build, compilation and operation of the code and application, all the code passes lints, could be build, unit tests are passing.\n"
                     "8. If the error is related to a specific coding guide, reference the relevant guide and explain how it applies to the issue and try to fix it.\n"
            )
        ),
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Here is the check for coding compliance:\n"
            )
        )
    ]

# # Check the stack used within this application and verify with the latest guides (refresh if needed) provided by the codeguide-mcp (`guides://list`), context7 mcp, and compare the recommendations and best styles with the current status of the implementation of this application and show what are the deviations and if they are serious propose steps to correct them.
# @mcp.prompt("check_code_quality", description="Check if the code complies with the quality standards.")
# def check_code_quality_prompt() -> list[PromptMessage]:
#     """Check if the code complies with the quality standards."""
#     return [
#         PromptMessage(
#             role="assistant",
#             content=TextContent(
#                 type="text",
#                 text="You are an expert Senior Engineer. Follow these steps:\n"
#                      "1. Analyze the source code, architecture and infrastructure stack.\n"
#                      "2. Analyze the best coding practices, styles and patterns according to this MCP codeguide mcp respective to every stack, API, architecture, documentation, tests, security, etc.\n"
#                      "3. Check and consult with other best practices and patterns over internet or Context7 MCP and other resources\n"
#                      "4. Use the 'search_documentation' tool for the error code.\n"
#                      "5. Check deviations and propose TODO steps and improvements to make the code compliant with the best practices and patterns and coding guides.\n"
#                      "6. Track any TODO steps and improvements.\n"
#                      "7. Make sure all fixes are not breaking the build, compilation and operation of the code and application, all the code passes lints, could be build, unit tests are passing.\n"
#                      "8. If the error is related to a specific coding guide, reference the relevant guide and explain how it applies to the issue and try to fix it.\n" 
#             )
#         ),
#         PromptMessage(
#             role="user",
#             content=TextContent(
#                 type="text",
#                 text=f"Here is the check for coding compliance:\n"
#             )
#         )
#     ]

# Check every part of the code for code that could be deduplicated and abstracted and proceed with the deduplication in order to make minimize the surface area that could produce bugs. This should be done for every part of the code - typescript, python, even dockerfiles and docker-compose. Always make sure the changes are not breaking the application build and behavior. Create and update gitea issue for the task.
@mcp.prompt("deduplicate_code", description="Check the code for duplication and propose deduplication steps.")
def deduplicate_code_prompt() -> list[PromptMessage]:
    """Check the code for duplication and propose deduplication steps."""
    return [
        PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text="You are an expert Senior Engineer. Follow these steps:\n"
                     "1. Analyze the source code, architecture and infrastructure stack.\n"
                     "2. Identify any duplicated code across the codebase, including TypeScript, Python, Dockerfiles, docker-compose, etc.\n"
                     "3. Propose specific steps to abstract and deduplicate the identified code while ensuring that the changes do not break the application build and behavior.\n"
                     "4. Make sure all fixes are not breaking the build, compilation and operation of the code and application, all the code passes lints, could be build, unit tests are passing.\n"
                     "5. For each proposed deduplication step, provide a brief explanation of how it improves the codebase and any potential trade-offs.\n"
                     "6. All the changes should produce nice, compact, readable and maintainable code.\n"
                     "7. Create or update a Git issue, commit for the deduplication task with clear instructions and references to the relevant coding guides if applicable.\n"
            )
        ),
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Here is the check for code duplication:\n"
            )
        )
    ]

# Upgrade and update the dependencies
@mcp.prompt("update_dependencies", description="Check for outdated dependencies and propose updates.")
def update_dependencies_prompt() -> list[PromptMessage]:
    """Check for outdated dependencies and propose updates."""
    return [
        PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text="You are an expert Senior Engineer. Follow these steps:\n"
                     "1. Analyze the project's dependencies across all relevant files (e.g., package.json, requirements.txt, Dockerfiles, images, docker-compose, etc.).\n"
                     "2. Identify any outdated dependencies and check for known vulnerabilities using appropriate tools (e.g., npm audit, pip-audit).\n"
                     "3. Identify major and minor version changes, API changes, package name changes, etc. that could produce breaking changes and incompatibilities.\n"
                     "4. Propose specific updates for the identified outdated dependencies while ensuring that the changes do not break the application build and behavior.\n"
                     "5. Make sure all fixes are not breaking the build, compilation and operation of the code and application, all the code passes lints, could be build, unit tests are passing.\n"
                     "6. For each proposed dependency update, provide a brief explanation of how it improves the project and any potential trade-offs.\n"
                     "7. Create or update a Git issue, commit for the dependency update task with clear instructions and references to the relevant coding guides if applicable.\n"
            )
        ),
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Here is the check for outdated dependencies:\n"
            )
        )
    ]

def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
