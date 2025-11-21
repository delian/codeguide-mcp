import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from coding_guides_server.server import list_guides, get_guide
from coding_guides_server.server import check_network_available, GITHUB_REPO


async def verify():
    print("=" * 60)
    print("MCP Server Verification")
    print("=" * 60)
    
    # Check configuration
    print(f"\nConfiguration:")
    print(f"  GitHub Repo: {GITHUB_REPO or 'Not configured'}")
    print(f"  Network Available: {check_network_available()}")
    
    # Test list_guides
    print("\n" + "-" * 60)
    print("Testing list_guides()...")
    print("-" * 60)
    try:
        guides = await list_guides()
        print(f"\nGuides list ({len(guides.split(chr(10)))} items):")
        print(guides[:500] + "..." if len(guides) > 500 else guides)
        
        if not guides or "not found" in guides.lower():
            print("\n⚠️  WARNING: No guides found or error message returned")
        else:
            print("\n✅ SUCCESS: Guides list retrieved")
            
            # Try to find at least one .md file mentioned
            if ".md" in guides:
                print("✅ SUCCESS: At least one markdown guide found")
            else:
                print("⚠️  WARNING: No .md files found in list")
    except Exception as e:
        print(f"\n❌ FAILURE: Error listing guides: {e}")
        sys.exit(1)
    
    # Test get_guide with first available guide
    print("\n" + "-" * 60)
    print("Testing get_guide()...")
    print("-" * 60)
    
    # Try common guide names
    test_guides = ["python.md", "javascript.md", "typescript.md", "go.md"]
    guide_found = False
    
    for guide_name in test_guides:
        try:
            print(f"\nTrying to fetch: {guide_name}")
            content = await get_guide(guide_name)
            
            if content and "ERROR: Guide" not in content:
                print(f"✅ SUCCESS: Retrieved {guide_name}")
                print(f"   Content length: {len(content)} characters")
                print(f"   First 200 chars: {content[:200]}...")
                guide_found = True
                break
            else:
                print(f"   Not found or invalid")
        except Exception as e:
            print(f"   Error: {e}")
    
    if not guide_found:
        print("\n⚠️  WARNING: Could not retrieve any test guides")
        print("   This might be normal if GitHub is not configured or network is unavailable")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify())
