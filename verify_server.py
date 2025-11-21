import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from coding_guides_server.server import list_guides, get_guide


async def verify():
    print("Verifying list_guides...")
    guides = await list_guides()
    print(f"Guides list:\n{guides}")

    if "python_style.md" in guides:
        print("SUCCESS: python_style.md found in list.")
    else:
        print("FAILURE: python_style.md not found in list.")
        sys.exit(1)

    print("\nVerifying get_guide...")
    content = await get_guide("python_style.md")
    print(f"Content length: {len(content)}")

    if "Guidelines" in content:
        print("SUCCESS: Content verification passed.")
    else:
        print("FAILURE: Content verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(verify())
