import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "looking_ahead_comfy" / "vendor"


def recorded_hashes():
    text = (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")
    return dict(re.findall(r"^\| `([\w/.]+\.py)` \| `([0-9a-f]{64})` \|$", text, flags=re.M))


def test_vendored_files_match_recorded_hashes():
    recorded = recorded_hashes()
    present = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*.py") if p.name not in ("__init__.py", "compat.py")}
    assert set(recorded) == present
    for name, sha in recorded.items():
        data = (VENDOR / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == sha, name


def test_vendored_code_has_no_environment_process_or_file_access():
    pattern = re.compile(r"os\.environ|getenv|subprocess|\beval\(|\bexec\(|\bopen\(|torch\.load|pickle|urllib|requests")
    for path in VENDOR.rglob("*.py"):
        hits = [line for line in path.read_text(encoding="utf-8").splitlines() if pattern.search(line)]
        assert not hits, (path.name, hits)


def test_shipped_modules_avoid_what_the_registry_scan_flags():
    # The Comfy Registry scan flagged `signature.bind(...)` in compat.py as a network operation (0.1.0); keep the shipped
    # modules free of that and of process, environment and network access. tools/download_model.py is a user-run
    # download script and is not imported by the nodes.
    package = ROOT / "looking_ahead_comfy"
    pattern = re.compile(r"\.bind\(|\bsocket\b|urllib|requests|subprocess|os\.environ|getenv|(?<![.\w])eval\(|(?<![.\w])exec\(|pickle")
    for path in [*package.rglob("*.py"), ROOT / "__init__.py"]:
        hits = [line for line in path.read_text(encoding="utf-8").splitlines() if pattern.search(line)]
        assert not hits, (path.name, hits)
        for line in path.read_text(encoding="utf-8").splitlines():
            if "torch.load(" in line:
                assert "weights_only=True" in line, (path.name, line)


def test_register_to_config_records_constructor_arguments():
    from looking_ahead_comfy.vendor.compat import register_to_config

    class Thing:
        @register_to_config
        def __init__(self, a, b=2, c=None):
            self.sum = a + b

    t = Thing(1, c="x")
    assert vars(t.config) == {"a": 1, "b": 2, "c": "x"} and t.sum == 3
    assert vars(Thing(5, 6).config) == {"a": 5, "b": 6, "c": None}
    assert vars(Thing(a=1, b=3).config) == {"a": 1, "b": 3, "c": None}


def test_regenerated_from_pinned_sources(upstream_dir, wan22_model):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "vendor_patch.py"), str(upstream_dir), str(wan22_model), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _definitions(path, drop=()):
    out = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                node.body = [b for b in node.body if not (isinstance(b, ast.FunctionDef) and b.name in drop)]
            out[node.name] = ast.dump(node)
    return out


def test_wan22_primitives_equal_looking_aheads(upstream_dir):
    ours = _definitions(VENDOR / "wan22" / "modules" / "model.py")
    theirs = _definitions(upstream_dir / "src" / "wan22" / "modules" / "model.py", drop=("reset_parameters",))
    assert ours == theirs
