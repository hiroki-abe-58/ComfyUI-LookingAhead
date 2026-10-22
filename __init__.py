"""Looking Ahead for ComfyUI (unofficial integration of https://github.com/TencentARC/Looking-Ahead)."""

import logging

try:
    from .looking_ahead_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[LookingAhead] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
