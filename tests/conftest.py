"""Test setup. The pack is imported as `looking_ahead_comfy` from the repository root.

Optional references (CI sets all of them and LOOKING_AHEAD_REQUIRE_REFERENCES=1, so nothing may be skipped there):
- LOOKING_AHEAD_UPSTREAM_DIR: a TencentARC/Looking-Ahead checkout at the vendored commit (and its Python dependencies).
- LOOKING_AHEAD_WAN22_MODEL: Wan2.2 wan/modules/model.py at the commit in vendor/UPSTREAM.md.
- LOOKING_AHEAD_COMFYUI_DIR: a ComfyUI checkout, for the node tests.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reference(name):
    value = os.environ.get(name)
    if value:
        return Path(value)
    if os.environ.get("LOOKING_AHEAD_REQUIRE_REFERENCES") == "1":
        pytest.fail(f"{name} is required")
    pytest.skip(f"{name} not set")


@pytest.fixture(scope="session")
def comfyui_dir():
    path = reference("LOOKING_AHEAD_COMFYUI_DIR")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


@pytest.fixture(scope="session")
def upstream_dir():
    return reference("LOOKING_AHEAD_UPSTREAM_DIR")


@pytest.fixture(scope="session")
def wan22_model():
    return reference("LOOKING_AHEAD_WAN22_MODEL")
