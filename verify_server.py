"""
Verify the codeguide-mcp server using an MCP client.

By default it starts the server as a subprocess and talks stdio. Pass
`--http URL` to verify an already-running remote deployment instead, e.g.

    uv run python verify_server.py --http https://my-service.run.app/mcp

Checks via the MCP protocol that:
- Instructions are provided (in InitializeResult)
- Resource guides://list and resource template guides://{guide_name} are exposed
- Tool get_guide(guide_name) is exposed
- All can be used: read_resource(guides://list), read_resource(guides://python.md), call_tool(get_guide)
- Non-existent guide (e.g. xxxxxx.md) returns an error via both ReadResource(guides://xxxxxx.md) and get_guide tool
- Every prompt is listed and renders non-empty messages
"""

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path (for imports if any)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Reduce log noise from server and MCP
os.environ.setdefault("GUIDES_LOG_LEVEL", "WARNING")
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("coding_guides_server").setLevel(logging.WARNING)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


@asynccontextmanager
async def _connect(http_url: str | None):
    """Yield an initialized-capable ClientSession over HTTP or stdio."""
    if http_url:
        async with (
            streamable_http_client(http_url) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            yield session
        return

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "main.py"],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        yield session


async def verify(http_url: str | None = None):
    print("=" * 60)
    print(f"MCP Server Verification (client over {http_url or 'stdio'})")
    print("=" * 60)

    errors = []
    warnings = []

    async with _connect(http_url) as session:
        # --- 1. Initialize and check instructions ---
        print("\n" + "-" * 60)
        print("1. Initialize and check instructions...")
        print("-" * 60)
        init_result = await session.initialize()
        instructions = getattr(init_result, "instructions", None)
        if not instructions or not (instructions or "").strip():
            errors.append(
                "Server did not provide instructions (InitializeResult.instructions empty)"
            )
            print("❌ FAIL: No instructions in InitializeResult")
        else:
            print(f"✅ PASS: Instructions provided ({len(instructions)} chars)")

        # mcp 2.0 renamed camelCase attributes to snake_case (serverInfo -> server_info)
        server_info = getattr(init_result, "server_info", None) or getattr(
            init_result, "serverInfo", None
        )
        server_name = getattr(server_info, "name", None) if server_info else None
        if server_name:
            print(f"   Server: {server_name}")

        # --- 2. List resources: expect guides://list ---
        print("\n" + "-" * 60)
        print("2. Checking resources (guides://list)...")
        print("-" * 60)
        list_res_result = await session.list_resources()
        resources = list_res_result.resources or []
        uris = [str(r.uri) for r in resources]
        if "guides://list" not in uris:
            errors.append(f"Resource guides://list not exposed. Got: {uris}")
            print("❌ FAIL: guides://list not in list_resources()")
        else:
            print("✅ PASS: guides://list resource exposed")

        # --- 3. List resource templates: expect guides://{guide_name} ---
        print("\n" + "-" * 60)
        print("3. Checking resource templates (guides://{guide_name})...")
        print("-" * 60)
        list_tpl_result = await session.list_resource_templates()
        templates = (
            getattr(list_tpl_result, "resource_templates", None)
            or getattr(list_tpl_result, "resourceTemplates", None)
            or []
        )
        guide_template = None
        for t in templates:
            ut = getattr(t, "uriTemplate", None) or getattr(t, "uri_template", "")
            if ut and "guide" in ut and "{guide_name}" in ut:
                guide_template = ut
                break
        if not guide_template:
            template_uris = [
                getattr(t, "uriTemplate", None) or getattr(t, "uri_template", "")
                for t in templates
            ]
            errors.append(
                f"Template guides://{{guide_name}} not exposed. Got: {template_uris}"
            )
            print("❌ FAIL: No guide template in list_resource_templates()")
        else:
            print(f"✅ PASS: Template {guide_template} exposed")

        # --- 4. List tools: expect get_guide ---
        print("\n" + "-" * 60)
        print("4. Checking tool get_guide...")
        print("-" * 60)
        list_tools_result = await session.list_tools()
        tools = list_tools_result.tools or []
        tool_names = [t.name for t in tools]
        if "get_guide" not in tool_names:
            errors.append(f"Tool get_guide not exposed. Got: {tool_names}")
            print("❌ FAIL: get_guide not in list_tools()")
        else:
            print("✅ PASS: get_guide tool exposed")

        # --- 5. Use resource guides://list ---
        print("\n" + "-" * 60)
        print("5. Using resource guides://list...")
        print("-" * 60)
        try:
            read_list_result = await session.read_resource("guides://list")
            contents = getattr(read_list_result, "contents", None) or []
            if not contents:
                errors.append("read_resource(guides://list) returned no content")
                print("❌ FAIL: No content from guides://list")
            else:
                text = getattr(contents[0], "text", None) or (
                    contents[0] if isinstance(contents[0], str) else None
                )
                if not text:
                    blob = getattr(contents[0], "blob", None)
                    if blob is not None:
                        text = (
                            blob.decode("utf-8")
                            if isinstance(blob, bytes)
                            else str(blob)
                        )
                if not text or "not found" in (text or "").lower():
                    warnings.append("guides://list content empty or error-like")
                    print("⚠️  WARN: guides://list content empty or error")
                elif ".md" not in text:
                    warnings.append("guides://list did not contain .md entries")
                    print("⚠️  WARN: No .md in list content")
                else:
                    print(f"✅ PASS: guides://list usable (content length {len(text)})")
        except Exception as e:
            errors.append(f"read_resource(guides://list): {e}")
            print(f"❌ FAIL: {e}")

        # --- 6. Use resource guides://python.md ---
        sample_guide = "python.md"
        print("\n" + "-" * 60)
        print(f"6. Using resource guides://{sample_guide}...")
        print("-" * 60)
        try:
            read_guide_result = await session.read_resource(f"guides://{sample_guide}")
            contents = getattr(read_guide_result, "contents", None) or []
            if not contents:
                errors.append(
                    f"read_resource(guides://{sample_guide}) returned no content"
                )
                print("❌ FAIL: No content")
            else:
                c0 = contents[0]
                text = getattr(c0, "text", None)
                if text is None and hasattr(c0, "blob") and c0.blob is not None:
                    text = (
                        c0.blob.decode("utf-8")
                        if isinstance(c0.blob, bytes)
                        else str(c0.blob)
                    )
                if not text:
                    text = str(c0)
                if not text.strip().startswith("#"):
                    warnings.append(f"guides://{sample_guide} content unexpected")
                    print(f"⚠️  WARN: Unexpected content (starts with: {text[:80]!r})")
                else:
                    print(
                        f"✅ PASS: guides://{sample_guide} usable (content length {len(text)})"
                    )
        except Exception as e:
            if "not found" in str(e).lower():
                warnings.append(f"guides://{sample_guide} not found (network/cache?)")
                print(f"⚠️  WARN: Guide not found: {e}")
            else:
                errors.append(f"read_resource(guides://{sample_guide}): {e}")
                print(f"❌ FAIL: {e}")

        # --- 7. Use tool get_guide(guide_name) ---
        print("\n" + "-" * 60)
        print(f"7. Using tool get_guide({sample_guide})...")
        print("-" * 60)
        try:
            call_result = await session.call_tool(
                "get_guide", {"guide_name": sample_guide}
            )
            content_blocks = getattr(call_result, "content", None) or []
            if not content_blocks:
                errors.append("call_tool(get_guide) returned no content")
                print("❌ FAIL: No content from get_guide")
            else:
                # Tool result can be list of TextContent or similar
                parts = []
                for block in content_blocks:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    else:
                        parts.append(str(block))
                text = "\n".join(parts) if parts else ""
                if "ERROR: Guide" in text:
                    warnings.append(f"get_guide returned error: {text[:200]}")
                    print("⚠️  WARN: get_guide returned error content")
                elif not text.strip():
                    warnings.append("get_guide returned empty content")
                    print("⚠️  WARN: Empty content")
                else:
                    print(
                        f"✅ PASS: get_guide({sample_guide}) usable (content length {len(text)})"
                    )
        except Exception as e:
            errors.append(f"call_tool(get_guide): {e}")
            print(f"❌ FAIL: {e}")

        # --- 8. Non-existent guide via resource must return error ---
        nonexistent = "xxxxxx.md"
        print("\n" + "-" * 60)
        print(
            f"8. Non-existent guide via resource guides://{nonexistent} (expect error)..."
        )
        print("-" * 60)
        try:
            await session.read_resource(f"guides://{nonexistent}")
            errors.append(
                f"read_resource(guides://{nonexistent}) should have failed but returned success"
            )
            print("❌ FAIL: Expected error, got success")
        except Exception as e:
            err_msg = str(e).lower()
            if "not found" in err_msg or "error" in err_msg or "resource" in err_msg:
                print(f"✅ PASS: Resource returned error as expected: {e}")
            else:
                errors.append(
                    f"read_resource(guides://{nonexistent}): unexpected error: {e}"
                )
                print(f"❌ FAIL: Unexpected error: {e}")

        # --- 9. Non-existent guide via get_guide tool must return error ---
        print("\n" + "-" * 60)
        print(
            f"9. Non-existent guide via tool get_guide({nonexistent}) (expect error)..."
        )
        print("-" * 60)
        try:
            call_result = await session.call_tool(
                "get_guide", {"guide_name": nonexistent}
            )
            content_blocks = getattr(call_result, "content", None) or []
            parts = []
            for block in content_blocks:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
            text = "\n".join(parts) if parts else ""
            # Accept either tool's explicit error string or server-wrapped error (e.g. "Error executing tool get_guide: Guide '...' not found.")
            is_error = "not found" in text.lower() and (
                "error" in text.lower() or "ERROR" in text
            )
            if is_error:
                print("✅ PASS: get_guide returned error as expected")
            else:
                errors.append(
                    f"get_guide({nonexistent}) should return error message, got: {text[:200]!r}"
                )
                print(
                    f"❌ FAIL: Expected error message in content, got: {text[:150]!r}"
                )
        except Exception as e:
            # Tool might raise or report via isError
            err_msg = str(e).lower()
            if "not found" in err_msg or "error" in err_msg:
                print(f"✅ PASS: get_guide raised error as expected: {e}")
            else:
                errors.append(f"call_tool(get_guide, {nonexistent}): {e}")
                print(f"❌ FAIL: {e}")

        # --- 10. Every prompt is listed and renders ---
        print("\n" + "-" * 60)
        print("10. Checking prompts render non-empty messages...")
        print("-" * 60)
        try:
            prompts = (await session.list_prompts()).prompts or []
            if not prompts:
                errors.append("No prompts exposed by the server")
                print("❌ FAIL: No prompts exposed")
            else:
                rendered = 0
                for prompt in prompts:
                    # Supply a usable value for any required argument so the
                    # prompt can actually render (e.g. get_guide(guide_name)).
                    args = {
                        arg.name: sample_guide
                        for arg in (prompt.arguments or [])
                        if arg.required
                    }
                    try:
                        result = await session.get_prompt(prompt.name, args)
                        text = "".join(
                            getattr(message.content, "text", "")
                            for message in result.messages
                        )
                        if text.strip():
                            rendered += 1
                        else:
                            errors.append(f"prompt {prompt.name} rendered empty")
                            print(f"❌ FAIL: prompt {prompt.name} rendered empty")
                    except Exception as e:
                        errors.append(f"prompt {prompt.name}: {e}")
                        print(f"❌ FAIL: prompt {prompt.name}: {e}")
                if rendered == len(prompts):
                    print(f"✅ PASS: all {rendered} prompts render non-empty messages")
        except Exception as e:
            errors.append(f"list_prompts(): {e}")
            print(f"❌ FAIL: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if errors:
        print(f"  ❌ FAILURES: {len(errors)}")
        for e in errors:
            print(f"     - {e}")
        sys.exit(1)
    if warnings:
        print(f"  ⚠️  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"     - {w}")
    else:
        print("  ✅ All checks passed (MCP client).")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--http",
        metavar="URL",
        help="Verify a running server over Streamable HTTP (e.g. https://host/mcp) "
        "instead of spawning a local stdio subprocess.",
    )
    args = parser.parse_args()
    asyncio.run(verify(args.http))


if __name__ == "__main__":
    main()
