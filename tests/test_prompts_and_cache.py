"""Regression tests for prompt handlers and cache thread-safety in server.py.

Covers bugs found during the 2026-06 rubber-duck of the server:
  * list_guides prompt did ``"\\n".join(get_guides_list())`` on a *string*,
    inserting a newline between every character.
  * help / exit prompts called ``mcp.get_help()`` / ``mcp.stop()`` which do not
    exist on FastMCP -> AttributeError at invocation.
  * cachetools @cached caches had no lock -> not thread-safe under concurrent
    requests / the _run_sync worker-thread offload.

Run: ``.venv/bin/python -m unittest discover -s tests`` (stdlib only).
"""

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "coding_guides_server"))

import server


class PromptHandlers(unittest.TestCase):
    def test_list_guides_prompt_is_not_char_mangled(self):
        out = server.list_guides_prompt()
        # Must equal the raw list string, NOT a per-character newline explosion.
        self.assertEqual(out, server.get_guides_list())
        first_line = out.split("\n")[0]
        self.assertGreater(
            len(first_line), 5, "lines should be guide entries, not single chars"
        )
        self.assertTrue(first_line.startswith("guides://"))

    def test_help_prompt_does_not_raise_and_returns_text(self):
        out = server.help_prompt()  # used to call non-existent mcp.get_help()
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())

    def test_exit_prompt_does_not_raise(self):
        out = server.exit_prompt()  # used to call non-existent mcp.stop()
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


class CacheThreadSafety(unittest.TestCase):
    def setUp(self):
        # Force the deterministic local-directory path (no network in tests).
        self._orig = server.check_network_available
        server.check_network_available = lambda: False
        server.clear_cache()

    def tearDown(self):
        server.check_network_available = self._orig
        server.clear_cache()

    def test_concurrent_get_guide_is_consistent(self):
        # Hammer the locked caches from many threads; all must return identical,
        # non-empty content with no exception.
        def call(_):
            return server.get_guide("python.md")

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(call, range(64)))
        self.assertTrue(all(r and "Python" in r[:80] for r in results))
        self.assertEqual(len(set(results)), 1, "all concurrent reads must agree")

    def test_concurrent_list_does_not_crash(self):
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _: server.get_guides_list(), range(64)))
        self.assertTrue(all(isinstance(r, str) and r for r in results))


class ListFiltering(unittest.TestCase):
    """guides://list must not offer scaffolding/meta files, but they stay fetchable."""

    def setUp(self):
        self._orig = server.check_network_available
        server.check_network_available = lambda: False  # local dir-scan path
        server.clear_cache()

    def tearDown(self):
        server.check_network_available = self._orig
        server.clear_cache()

    def test_template_and_conventions_hidden_from_list(self):
        names = {
            server._list_line_guide_name(l)
            for l in server.get_guides_list().splitlines()
            if l.strip()
        }
        self.assertNotIn("TEMPLATE.md", names)
        self.assertNotIn("CONVENTIONS.md", names)
        self.assertIn("python.md", names)  # real guides still present

    def test_normalize_drops_hidden_even_if_brief_contains_them(self):
        brief = (
            "guides://TEMPLATE.md - x\nguides://CONVENTIONS.md - y\nguides://tdd.md - z"
        )
        out = server._normalize_list_to_guide_uris(brief)
        kept = {server._list_line_guide_name(l) for l in out.splitlines() if l.strip()}
        self.assertEqual(kept, {"tdd.md"})

    def test_hidden_guides_remain_fetchable_by_name(self):
        # References like guides://CONVENTIONS.md inside guides must still resolve.
        self.assertTrue(server.get_guide("CONVENTIONS.md"))
        self.assertTrue(server.get_guide("TEMPLATE.md"))


if __name__ == "__main__":
    unittest.main()
