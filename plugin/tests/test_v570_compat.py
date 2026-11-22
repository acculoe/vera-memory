#!/usr/bin/env python3
"""V5.7.0 fresh-install checks — manifest version + zero-setup identity.

Synthetic only: throwaway temp homes, no credentials, no network. Run either way:

    python3 tests/test_v570_compat.py     # standalone script
    pytest tests/                         # collected as ordinary test functions
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]

# --- load the package exactly as an install would, without needing a gateway --
try:  # a real Hermes runtime already provides this
    from agent.memory_provider import MemoryProvider  # noqa: F401
except Exception:  # standalone run: stub the single runtime import
    _agent = types.ModuleType("agent")
    _mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal base, only needed for isinstance/signature
        pass

    _mp.MemoryProvider = MemoryProvider
    _agent.memory_provider = _mp
    sys.modules["agent"] = _agent
    sys.modules["agent.memory_provider"] = _mp

_spec = importlib.util.spec_from_file_location(
    "vera_memory", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)],
)
vera_memory = importlib.util.module_from_spec(_spec)
sys.modules["vera_memory"] = vera_memory
_spec.loader.exec_module(vera_memory)


def _declared_version() -> str:
    manifest = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8-sig")
    for line in manifest.splitlines():
        if line.strip().startswith("version:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def test_version_label_tracks_manifest():
    """The agent-facing prompt block must never advertise a stale version."""
    declared = _declared_version()
    assert declared, "plugin.yaml has no version line"
    assert vera_memory._VERSION == declared, f"{vera_memory._VERSION} != {declared}"
    block = vera_memory.VERAMemoryProvider().system_prompt_block()
    assert f"v{declared}" in block, block[:120]


def test_fresh_home_reports_available():
    """A brand-new install has no profiles/ directory yet and must still count as available."""
    with tempfile.TemporaryDirectory(prefix="vera570-home-") as tmp:
        home = Path(tmp) / "hermes"
        home.mkdir()
        os.environ["HERMES_HOME"] = str(home)
        assert not (home / "profiles").exists()
        assert vera_memory.VERAMemoryProvider().is_available() is True


def test_first_turn_store_works_without_manual_provisioning():
    """Hermes supplies agent_identity, so the agent's own principal exists on turn one."""
    with tempfile.TemporaryDirectory(prefix="vera570-home-") as tmp:
        home = Path(tmp)
        os.environ["HERMES_HOME"] = str(home)
        provider = vera_memory.VERAMemoryProvider()
        provider.initialize(session_id="v570-fresh", hermes_home=str(home), agent_identity="freshagent")
        try:
            stored = provider.handle_tool_call("vera_store", {"content": "fresh install round trip", "category": "facts"})
            assert stored.startswith("Memory stored"), stored
            found = provider.handle_tool_call("vera_search", {"query": "fresh install round trip"})
            assert "fresh install round trip" in found, found
        finally:
            provider.shutdown()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001 - a failing check must not stop the rest
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{3 - failures}/3 checks passed")
    raise SystemExit(1 if failures else 0)
