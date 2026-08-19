"""Regression tests for the async-from-sync bridge in coding_guides_server.server.

Bug: get_guide / the GitHub network wrappers called ``asyncio.run()`` directly,
which raises ``RuntimeError: asyncio.run() cannot be called from a running event
loop`` when invoked inside the MCP server's event loop (e.g. a tool handler).
It surfaced on a GitHub cache-miss (a guide not yet pushed / not yet cached).

These tests pin the fix: ``server._run_sync`` must drive a coroutine to
completion whether or not a loop is already running in the calling thread, and
the sync network wrappers must never raise the "running event loop" RuntimeError.

Run: ``.venv/bin/python -m unittest discover -s tests`` (stdlib only, no pytest).
"""

import asyncio
import sys
import unittest
from pathlib import Path

# Import the server module (repo root + package dir on the path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "coding_guides_server"))

import server


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


class RunSyncOutsideLoop(unittest.TestCase):
    def test_runs_coroutine_when_no_loop_is_running(self):
        # No event loop running in this thread -> direct asyncio.run path.
        self.assertEqual(server._run_sync(_answer()), 42)


class RunSyncInsideLoop(unittest.IsolatedAsyncioTestCase):
    async def test_runs_coroutine_from_within_a_running_loop(self):
        # This coroutine runs inside a live event loop — the condition that made
        # a bare asyncio.run() raise. The helper must return the value instead.
        self.assertEqual(server._run_sync(_answer()), 42)

    async def test_bare_asyncio_run_still_fails_here(self):
        # Guards the premise: a naive asyncio.run() DOES raise in this context,
        # so the helper is genuinely necessary (not redundant).
        with self.assertRaises(RuntimeError):
            asyncio.run(_answer())

    async def test_network_wrappers_do_not_raise_running_loop_error(self):
        # The three wrappers must be callable from inside the loop without the
        # "running event loop" RuntimeError. Network result is irrelevant here;
        # we only assert that specific failure mode is gone.
        for call in (
            server.check_network_available,
            server.fetch_github_directory_listing,
            lambda: server.fetch_github_file_content("guides/__nonexistent__.md"),
        ):
            try:
                call()
            except RuntimeError as exc:  # pragma: no cover - this is the bug
                if "running event loop" in str(exc):
                    self.fail(f"{call} raised the regressed error: {exc}")
                raise


if __name__ == "__main__":
    unittest.main()
