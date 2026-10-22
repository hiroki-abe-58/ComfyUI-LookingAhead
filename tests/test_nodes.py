import asyncio

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="module")
def nodes(comfyui_dir):
    from looking_ahead_comfy import nodes as module

    return module


def test_extension_lists_the_three_nodes(nodes):
    extension = asyncio.run(nodes.comfy_entrypoint())
    listed = asyncio.run(extension.get_node_list())
    assert [n.define_schema().node_id for n in listed] == ["LookingAheadModelLoader", "LookingAheadImageToVideo", "LookingAheadUnload"]
    import folder_paths

    assert "looking_ahead" in folder_paths.folder_names_and_paths


def test_checkpoints_are_listed_with_forward_slashes(nodes, tmp_path, monkeypatch):
    import folder_paths

    from looking_ahead_comfy import paths

    (tmp_path / "fcp_wan21_t2v_1.3b").mkdir()
    (tmp_path / "fcp_wan21_t2v_1.3b" / "model_ema.pt").write_bytes(b"")
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "looking_ahead", ([str(tmp_path)], {".pt", ".pth"}))
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    assert paths.checkpoints() == ["fcp_wan21_t2v_1.3b/model_ema.pt"]
    assert paths.resolve("fcp_wan21_t2v_1.3b/model_ema.pt") == str(tmp_path / "fcp_wan21_t2v_1.3b" / "model_ema.pt")
    with pytest.raises(FileNotFoundError, match="huggingface.co/TencentARC/Looking-Ahead"):
        paths.resolve("missing.pt")


def test_sampler_schema_defaults_are_the_official_settings(nodes):
    inputs = {i.id: i for i in nodes.LookingAheadImageToVideo.define_schema().inputs}
    assert (inputs["steps"].default, inputs["cfg"].default, inputs["shift"].default, inputs["seconds"].default) == (50, 3.0, 5.0, 5)
    assert (inputs["width"].default, inputs["height"].default) == (768, 432)
    outputs = [o.display_name for o in nodes.LookingAheadImageToVideo.define_schema().outputs]
    assert outputs == ["video", "images", "latent", "report"]


def test_embeddings_are_padded_to_512_and_validated(nodes):
    embeds = torch.randn(1, 68, 4096)
    padded = nodes._embeddings([[embeds, {}]], "positive")
    assert padded.shape == (1, 512, 4096)
    assert torch.equal(padded[:, :68], embeds) and not padded[:, 68:].any()
    with pytest.raises(ValueError, match="umT5"):
        nodes._embeddings([[torch.randn(1, 77, 768), {}]], "positive")
    with pytest.raises(ValueError, match="512"):
        nodes._embeddings([[torch.randn(1, 600, 4096), {}]], "positive")
    with pytest.raises(ValueError):
        nodes._embeddings([], "negative")


@pytest.mark.parametrize("size", [(1672, 941), (432, 768), (768, 432), (1000, 1000)])
def test_image_preprocessing_equals_the_official_pil_path(nodes, size):
    from looking_ahead_comfy.vendor.dataset.image_preprocessing import resize_center_crop
    import torchvision.transforms.functional as TF

    rgb = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    official, report = resize_center_crop(Image.fromarray(rgb), 432, 768)
    want = TF.to_tensor(official).permute(1, 2, 0).unsqueeze(0)  # what the official run normalizes, as a ComfyUI IMAGE
    image = torch.from_numpy(rgb).float().div(255).unsqueeze(0)  # what LoadImage gives for the same file
    got, got_report = nodes._prepare_image(image, 768, 432)
    assert got.shape == (1, 432, 768, 3)
    assert torch.equal(got, want)
    assert got_report == report
