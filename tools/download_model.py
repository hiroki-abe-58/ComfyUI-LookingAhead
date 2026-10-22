"""One-time setup: download a Looking Ahead checkpoint (and, if asked, the Wan text encoder and VAE in ComfyUI's format) into
<ComfyUI>/models at pinned revisions, and verify every file's SHA-256. The ComfyUI nodes never download anything.

    python tools/download_model.py --models-dir <ComfyUI>/models                 # FCP 1.3B, about 8.0 GB
    python tools/download_model.py --models-dir <ComfyUI>/models --tf            # also the TF baseline, about 5.7 GB
    python tools/download_model.py --models-dir <ComfyUI>/models --wan           # also umT5-XXL fp8 (6.7 GB) + Wan 2.1 VAE (0.25 GB)
    python tools/download_model.py --models-dir <ComfyUI>/models --wan-fp16      # umT5-XXL fp16 (11.4 GB) instead of fp8
    python tools/download_model.py --models-dir <ComfyUI>/models --5b --wan      # FCP 5B (34.2 GB) + umT5-XXL fp8 + Wan 2.2 VAE (1.4 GB)

With --5b, --tf adds the 5B TF baseline (20.0 GB). The 5B checkpoints are fp32 with their training-only branches; the loader
keeps a 10 GB bfloat16 copy of the backbone.
Run it with the Python of your ComfyUI environment (huggingface_hub is installed with transformers).
Licenses: Looking Ahead weights Apache-2.0 with a NOTICE (both are saved next to the checkpoint); Wan files Apache-2.0.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

LOOKING_AHEAD = ("TencentARC/Looking-Ahead", "91b4b2e8ce6d39bb85a242f621e410e0e4e7033b")
WAN = ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d")
WAN22 = ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "ee6f4a40737a995bf5818954cfce6d59443b0f04")
CHECKPOINTS = {
    "fcp_wan21_t2v_1.3b": "415946daf673e4ab8ed8a121cfd329391cdc0aaab601a9252be598af5c11d146",
    "tf_wan21_t2v_1.3b": "b5adbbf8513ee869364a8902d55c1c13e0a86e5e245d26d50c849a96bff27d30",
    "fcp_wan22_ti2v_5b": "0c47d3fefe0889c4095f4a7faea4191e0d0c7bbf3e1e02d50ed86662f976def5",
    "tf_wan22_ti2v_5b": "9fca5ce740a0296877a62dcb6c577223715bc2372e960dc34e3fcc2bb4cc0453",
}
CHECKPOINT_TEXTS = {"LICENSE": None, "NOTICE": "f636ead7a2641d3619cdd3871a441ef309c4c48f277bbbf2324ff3e9f836fa10"}
WAN_FILES = {
    "fp8": ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders",
            "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"),
    "fp16": ("split_files/text_encoders/umt5_xxl_fp16.safetensors", "text_encoders",
             "7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af"),
    "vae": ("split_files/vae/wan_2.1_vae.safetensors", "vae", "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
}
WAN22_VAE = ("split_files/vae/wan2.2_vae.safetensors", "vae", "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, revision: str, name: str, dest: Path, expected: str | None) -> None:
    from huggingface_hub import hf_hub_download

    if dest.exists() and (expected is None or sha256(dest) == expected):
        print(f"ok (already present): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        got = Path(hf_hub_download(repo, name, revision=revision, local_dir=tmp))
        if expected is not None and sha256(got) != expected:
            raise SystemExit(f"SHA-256 mismatch for {repo}/{name}; nothing installed")
        shutil.move(str(got), dest)
    print(f"ok: {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, required=True, help="ComfyUI/models")
    ap.add_argument("--5b", dest="five_b", action="store_true", help="the 5B (Wan 2.2 TI2V) model instead of 1.3B")
    ap.add_argument("--tf", action="store_true", help="also download the teacher-forcing (TF) baseline")
    ap.add_argument("--wan", action="store_true", help="also download umT5-XXL fp8 and the model's VAE (ComfyUI format)")
    ap.add_argument("--wan-fp16", action="store_true", help="like --wan, with the fp16 text encoder")
    a = ap.parse_args()
    if not a.models_dir.is_dir():
        raise SystemExit(f"{a.models_dir} is not a directory (expected ComfyUI/models)")
    size = "wan22_ti2v_5b" if a.five_b else "wan21_t2v_1.3b"
    for config in [f"fcp_{size}"] + ([f"tf_{size}"] if a.tf else []):
        target = a.models_dir / "looking_ahead" / config
        fetch(*LOOKING_AHEAD, f"{config}/model_ema.pt", target / "model_ema.pt", CHECKPOINTS[config])
        for text, expected in CHECKPOINT_TEXTS.items():
            fetch(*LOOKING_AHEAD, f"{config}/{text}", target / text, expected)
    if a.wan or a.wan_fp16:
        name, folder, expected = WAN_FILES["fp16" if a.wan_fp16 else "fp8"]
        fetch(*WAN, name, a.models_dir / folder / Path(name).name, expected)
        name, folder, expected = WAN22_VAE if a.five_b else WAN_FILES["vae"]
        fetch(*(WAN22 if a.five_b else WAN), name, a.models_dir / folder / Path(name).name, expected)


if __name__ == "__main__":
    main()
