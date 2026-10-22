"""Regenerate looking_ahead_comfy/vendor/ from a TencentARC/Looking-Ahead checkout at UPSTREAM_COMMIT.

Every change is an exact text replacement that must apply exactly once (or a method body swapped for a stub), so a new
upstream commit that moves the code fails loudly instead of being patched silently. vendor/UPSTREAM.md explains each change.

    python scripts/vendor_patch.py <Looking-Ahead checkout> [--check]

--check regenerates in memory and fails if the files in vendor/ differ (used by the tests with an upstream checkout).
"""
import argparse
import subprocess
import sys
from pathlib import Path

UPSTREAM_REPO = "https://github.com/TencentARC/Looking-Ahead"
UPSTREAM_COMMIT = "210794c758f386d3a1eaf3673b51b7cdd8374dfc"
VENDOR = Path(__file__).resolve().parents[1] / "looking_ahead_comfy" / "vendor"

FILES = {
    "utils/causal_model.py": "src/utils/causal_model.py",
    "utils/causal_rope.py": "src/utils/causal_rope.py",
    "utils/attention.py": "src/utils/attention.py",
    "wan/modules/model.py": "src/wan/modules/model.py",
    "wan/modules/attention.py": "src/wan/modules/attention.py",
    "wan/utils/fm_solvers_unipc.py": "src/wan/utils/fm_solvers_unipc.py",
    "dataset/image_preprocessing.py": "src/dataset/image_preprocessing.py",
}

FA4_GUARD = (
    "if FLASH_ATTN_4_AVAILABLE:\n    import flash_attn.cute\n",
    "if FLASH_ATTN_4_AVAILABLE:\n    try:\n        import flash_attn.cute\n    except ImportError:\n"
    "        # Installed but not importable, e.g. on Windows, where its quack dependency imports fcntl.\n"
    "        FLASH_ATTN_4_AVAILABLE = False\n",
)
def fa4_probe(dots):
    return ("from transformers.utils.import_utils import is_flash_attn_4_available\n",
            f"from {dots}compat import is_flash_attn_4_available\n")

REPLACEMENTS = {
    "utils/causal_model.py": [
        ("from diffusers.configuration_utils import ConfigMixin, register_to_config\n"
         "from diffusers.models.modeling_utils import ModelMixin\n",
         "from ..compat import ConfigMixin, ModelMixin, register_to_config\n"),
        ("from src.utils.causal_mask import CausalMaskMixin, visualize_causal_mask\n", ""),
        ("from src.utils.causal_rope import causal_rope_apply\n", "from .causal_rope import causal_rope_apply\n"),
        ("from src.utils.distributed import is_main_process\n", ""),
        ("from src.wan.modules.model import WAN_CROSSATTENTION_CLASSES\n",
         "from ..wan.modules.model import WAN_CROSSATTENTION_CLASSES\n"),
        ("from src.wan22.modules.model import (\n", "from ..wan22.modules.model import (\n"),
        ('flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")\n', ""),
        ("class CausalWanModel(ModelMixin, ConfigMixin, CausalMaskMixin):\n", "class CausalWanModel(ModelMixin, ConfigMixin):\n"),
    ],
    "utils/attention.py": [fa4_probe(".."), FA4_GUARD],
    "wan/modules/attention.py": [fa4_probe("..."), FA4_GUARD],
    "wan/modules/model.py": [
        ("from .attention import flash_attention\n", "from .attention import attention\n"),
        ("        x = flash_attention(q, k, v, k_lens=context_lens)\n", "        x = attention(q, k, v, k_lens=context_lens)\n"),
    ],
    "wan/utils/fm_solvers_unipc.py": [
        ("from diffusers.configuration_utils import ConfigMixin, register_to_config\n"
         "from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput\n",
         "from ...compat import ConfigMixin, KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput, register_to_config\n"),
    ],
}

# Methods whose body is replaced: the teacher-forcing training forward (it also reads WAN_VIS_* environment variables and
# writes mask images), which inference never calls.
STUBS = {
    "utils/causal_model.py": [
        ("    def _forward_train(self, x, t, context, clean_x=None):\n", "    def forward(self, *args, **kwargs):\n",
         "    def _forward_train(self, x, t, context, clean_x=None):\n"
         '        raise NotImplementedError("Training is not part of ComfyUI-LookingAhead; use the official repository.")\n\n'),
    ],
}


# vendor/wan22/modules/model.py: Looking-Ahead's src/wan22/modules/model.py keeps five primitives through a Wan2.2-TI2V-5B-Turbo
# lineage with a maintainer-provided permission (its NOTICE). The same five come here from the official Wan2.2 repository
# (Apache-2.0, Alibaba Wan Team) instead, with the two one-line changes that make them identical to Looking-Ahead's (bfloat16
# LayerNorm input and the rope output dtype); tests/test_vendor.py compares the syntax trees with Looking-Ahead's file.
WAN22_REPO = "https://github.com/Wan-Video/Wan2.2"
WAN22_COMMIT = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
WAN22_FILE = "wan/modules/model.py"
WAN22_SHA256 = "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432"
WAN22_NAMES = ("sinusoidal_embedding_1d", "rope_params", "rope_apply", "WanRMSNorm", "WanLayerNorm")
WAN22_REPLACEMENTS = [
    ("        return super().forward(x.float()).type_as(x)\n", "        return super().forward(x).type_as(x)\n"),
    ("    return torch.stack(output).float()\n", "    return torch.stack(output).type_as(x)\n"),
]


def build_wan22(source: Path) -> str:
    import ast
    import hashlib

    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WAN22_SHA256:
        raise SystemExit(f"{source}: not {WAN22_REPO} {WAN22_FILE} at {WAN22_COMMIT[:12]} (sha256 mismatch)")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    parts = []
    for name in WAN22_NAMES:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        parts.append("\n".join(lines[start:node.end_lineno]) + "\n")
    body = "\n\n".join(parts)
    for old, new in WAN22_REPLACEMENTS:
        if body.count(old) != 1:
            raise SystemExit(f"wan22: expected exactly one {old!r}, found {body.count(old)}")
        body = body.replace(old, new)
    return (f"# The five primitives below are from {WAN22_REPO} at {WAN22_COMMIT[:12]}, {WAN22_FILE}\n"
            "# (Apache-2.0, see LICENSE.Wan2.2.txt), modified by scripts/vendor_patch.py, see vendor/UPSTREAM.md.\n"
            "# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.\n"
            "import torch\nimport torch.nn as nn\n\n\n" + body)


def header(vendor_path, upstream_path, changed):
    note = "modified by scripts/vendor_patch.py, see vendor/UPSTREAM.md" if changed else "unchanged"
    return f"# Vendored from {UPSTREAM_REPO} at {UPSTREAM_COMMIT[:12]}, {upstream_path} ({note}).\n"


def build(upstream: Path, wan22_model: Path) -> dict:
    out = {"wan22/modules/model.py": build_wan22(wan22_model)}
    for vendor_path, upstream_path in FILES.items():
        text = (upstream / upstream_path).read_bytes().decode("utf-8").replace("\r\n", "\n")
        for old, new in REPLACEMENTS.get(vendor_path, []):
            if text.count(old) != 1:
                raise SystemExit(f"{vendor_path}: expected exactly one {old!r}, found {text.count(old)}")
            text = text.replace(old, new)
        for start, end, replacement in STUBS.get(vendor_path, []):
            if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
                raise SystemExit(f"{vendor_path}: cannot locate the {start.strip()!r} block")
            text = text[:text.index(start)] + replacement + text[text.index(end):]
        changed = vendor_path in REPLACEMENTS or vendor_path in STUBS
        out[vendor_path] = header(vendor_path, upstream_path, changed) + text
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upstream", type=Path)
    ap.add_argument("wan22_model", type=Path, help=f"{WAN22_FILE} of {WAN22_REPO} at {WAN22_COMMIT}")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream checkout is at {head}, expected {UPSTREAM_COMMIT}")
    files = build(a.upstream, a.wan22_model)
    if a.check:
        stale = [p for p, text in files.items() if (VENDOR / p).read_bytes().decode("utf-8") != text]
        if stale:
            raise SystemExit(f"vendor/ differs from the regenerated files: {stale}")
        print(f"vendor/ matches {UPSTREAM_COMMIT[:12]} plus the recorded patches ({len(files)} files)")
        return
    for p, text in files.items():
        dest = VENDOR / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(text.encode("utf-8"))
        for parent in dest.relative_to(VENDOR).parents:
            init = VENDOR / parent / "__init__.py"
            if not init.exists():
                init.write_bytes(b"")
    print(f"wrote {len(files)} files to {VENDOR}")


if __name__ == "__main__":
    sys.exit(main())
