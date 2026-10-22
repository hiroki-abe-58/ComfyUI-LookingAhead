import json
import sys
import types

import pytest
import torch

from looking_ahead_comfy import runtime

# Tiny Wan 2.1 and Wan 2.2 TI2V shaped backbones (two blocks, 64 wide): the code paths of the 1.3B and 5B models at a size
# a CPU runs in seconds.
TINY = dict(model_type="t2v", dim=64, eps=1e-06, ffn_dim=128, freq_dim=32, in_dim=16, num_heads=2, num_layers=2, out_dim=16,
            text_len=512)
TINY22 = dict(TINY, model_type="ti2v", in_dim=48, out_dim=48)


def test_frame_counts_follow_the_official_geometry():
    assert runtime.frame_counts(seconds=5) == (80, 22)  # the released recipe: 22 latent frames, 85 decoded, 80 kept
    assert runtime.frame_counts(seconds=10) == (160, 43)
    assert runtime.frame_counts(seconds=1) == (16, 7)
    assert runtime.frame_counts(pixel_frames=81) == (81, 22)
    assert runtime.frame_counts(seconds=5, family=runtime.WAN22) == (120, 31)  # 5B recipe: 31 latent frames at 24 fps
    for family in runtime.FAMILIES:
        for seconds in range(1, 31):
            pixels, latents = runtime.frame_counts(seconds=seconds, family=family)
            assert (latents - 1) % runtime.FRAMES_PER_BLOCK == 0 and 1 + 4 * (latents - 1) >= pixels


def test_family_is_read_from_the_patch_embedding():
    def state(dim, in_dim):
        return {"patch_embedding.weight": torch.empty(dim, in_dim, 1, 2, 2, device="meta")}

    assert runtime.family_of(state(1536, 16)) is runtime.WAN21
    assert runtime.family_of(state(3072, 48)) is runtime.WAN22
    with pytest.raises(ValueError, match="patch embedding"):
        runtime.family_of(state(5120, 16))  # Wan 2.1 14B
    with pytest.raises(ValueError):
        runtime.family_of({})


def test_backbone_state_dict_drops_training_branches_and_prefix():
    payload = {"generator_ema": {"model.patch_embedding.weight": 1, "_fsdp_wrapped_module.model.head.head.weight": 2,
                                 "model.predictive_multi_chunk_branches.offset_1.x": 3,
                                 "model.predictive_multi_chunk_source_layers_metadata": 4}}
    assert runtime.backbone_state_dict(payload) == {"patch_embedding.weight": 1, "head.head.weight": 2}
    with pytest.raises(ValueError):
        runtime.backbone_state_dict({"text_encoder.weight": 0})


def tiny_model(train_frames, seed=0):
    torch.manual_seed(seed)
    model = runtime.CausalWanModel(**TINY, local_attn_size=train_frames, sink_size=runtime.SINK_SIZE, train_frame_size=train_frames)
    return runtime.build_model(model.state_dict(), dtype=torch.bfloat16, config=TINY, train_latent_frames=train_frames)


def tiny_inputs(generated, seed=1, height=4, width=6, channels=16):
    g = torch.Generator().manual_seed(seed)
    positive = torch.zeros(1, 512, 4096)
    negative = torch.zeros(1, 512, 4096)
    positive[:, :7] = torch.randn(1, 7, 4096, generator=g)
    negative[:, :5] = torch.randn(1, 5, 4096, generator=g)
    initial = torch.randn(1, 1, channels, height, width, generator=g)
    noise = torch.randn(1, generated, channels, height, width, generator=g)
    return tuple(t.to(torch.bfloat16) for t in (positive, negative, initial, noise))


def test_sample_runs_and_reports_progress_on_cpu():
    model = tiny_model(7)
    positive, negative, initial, noise = tiny_inputs(6)
    calls = []
    out = runtime.sample(model, positive, negative, initial, noise, steps=3, callback=lambda d, t: calls.append((d, t)))
    assert out.shape == (1, 7, 16, 4, 6)
    assert torch.equal(out[:, :1], initial)
    assert calls == [(i, 6) for i in range(1, 7)]
    again = runtime.sample(model, positive, negative, initial, noise, steps=3)
    assert torch.equal(out, again)


def test_sample_rejects_unaligned_lengths():
    model = tiny_model(7)
    positive, negative, initial, noise = tiny_inputs(4)
    with pytest.raises(ValueError):
        runtime.sample(model, positive, negative, initial, noise, steps=2)


def _official_pipeline(upstream_dir, tmp_path, monkeypatch, train_frames, height, width, model_name="Wan2.1-T2V-1.3B",
                       config=TINY):
    """The official CausalDiffusionInferencePipeline around a tiny WanDiffusionWrapper (random weights, bfloat16 as the official\n    runtime casts it, CPU)."""
    sys.path.insert(0, str(upstream_dir))
    from src.pipeline import causal_diffusion_inference as cdi
    from src.utils import wan_wrapper
    from src.wan.modules import attention as wan_attention
    from src.wan.modules import model as wan_model

    model_dir = tmp_path / model_name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"_class_name": "WanModel", **{k: v for k, v in config.items()}}))
    monkeypatch.setattr(wan_wrapper, "WAN_ROOT", str(tmp_path))
    monkeypatch.setattr(wan_wrapper, "WAN_LOCAL_CACHE_ENABLE", False)
    # Upstream cross-attention calls the FA4-only flash_attention(); on a CPU without FA4 use the same call through attention()
    # (the change in vendor/UPSTREAM.md and TencentARC/Looking-Ahead#2).
    monkeypatch.setattr(wan_model, "flash_attention", wan_attention.attention)
    generator = wan_wrapper.WanDiffusionWrapper(model_name=model_name,
                                                latent_shape=[1, train_frames, config["in_dim"], height, width],
                                                load_diffusion_pretrained=False, initialization_seed=0,
                                                local_attn_size=-1, sink_size=runtime.SINK_SIZE).to(dtype=torch.bfloat16)
    args = types.SimpleNamespace(condition_only_first_frame=True, sampling_steps=3, sample_solver="unipc", timestep_shift=5.0,
                                 num_frame_per_block=3, independent_first_frame=True, guidance_scale=3.0, negative_prompt="neg")
    return cdi, generator, args


@pytest.mark.parametrize("model_name,config,train_frames,generated", [
    ("Wan2.1-T2V-1.3B", TINY, 16, 15),  # one window
    ("Wan2.1-T2V-1.3B", TINY, 7, 12),  # a rolling cache with the 3-frame sink
    ("Wan2.2-TI2V-5B", TINY22, 7, 12),  # the same with 48-channel Wan 2.2 latents and the ti2v backbone
])
def test_sample_matches_the_official_pipeline(upstream_dir, tmp_path, monkeypatch, model_name, config, train_frames, generated):
    cdi, generator, args = _official_pipeline(upstream_dir, tmp_path, monkeypatch, train_frames, 4, 6, model_name, config)
    positive, negative, initial, noise = tiny_inputs(generated, channels=config["in_dim"])
    embeds = {"pos": positive, "neg": negative}
    text_encoder = lambda text_prompts: {"prompt_embeds": embeds["neg" if text_prompts == ["neg"] else "pos"]}  # noqa: E731
    pipeline = cdi.CausalDiffusionInferencePipeline(args, device="cpu", generator=generator, text_encoder=text_encoder, vae=object())
    want = pipeline.inference(noise, ["pos"], initial_latent=initial, return_video=False)

    model = runtime.build_model(generator.model.state_dict(), dtype=torch.bfloat16, config=config, train_latent_frames=train_frames)
    got = runtime.sample(model, positive, negative, initial, noise, steps=3, cfg=3.0, shift=5.0)
    assert torch.equal(got, want)
