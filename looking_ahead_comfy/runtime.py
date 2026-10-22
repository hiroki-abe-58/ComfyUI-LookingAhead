"""Looking Ahead (FCP / TF) causal text-and-image-to-video sampling, 1.3B (Wan 2.1) and 5B (Wan 2.2 TI2V), independent of
ComfyUI.

The backbone is the vendored CausalWanModel (vendor/, TencentARC/Looking-Ahead at the commit in vendor/UPSTREAM.md). The
sampling loop is a line-for-line port of CausalDiffusionInferencePipeline.inference() at that commit, which both model sizes
share: the conditioning frame fills separate positive/negative KV caches, then each 3-latent-frame chunk is denoised with
order-2 flow UniPC and CFG, and its clean result is written back into both caches. Latents use the official layout
[B, T, C, H, W], normalized with the VAE mean/std of the model's family (the same values as ComfyUI's Wan21 and Wan22 latent
formats).
"""
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch

from .vendor.utils.causal_model import CausalWanModel
from .vendor.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .vendor.wan22.modules.model import rope_params

# config.json of Wan2.1-T2V-1.3B and of Wan2.2-TI2V-5B (model_type is the family's task in src/utils/wan_model_family.py).
WAN21_13B = dict(model_type="t2v", dim=1536, eps=1e-06, ffn_dim=8960, freq_dim=256, in_dim=16, num_heads=12, num_layers=30,
                 out_dim=16, text_len=512)
WAN22_5B = dict(model_type="ti2v", dim=3072, eps=1e-06, ffn_dim=14336, freq_dim=256, in_dim=48, num_heads=24, num_layers=30,
                out_dim=48, text_len=512)
SINK_SIZE = 3
FRAMES_PER_BLOCK = 3
VAE_TEMPORAL_STRIDE = 4


@dataclass(frozen=True)
class Family:
    """What differs between the released model sizes (src/utils/wan_model_family.py and the released recipes)."""

    name: str
    config: dict
    train_latent_frames: int  # image_or_video_shape[1] of the recipes; local_attn_size -1 resolves to it
    fps: int
    width: int
    height: int
    vae_spatial_stride: int
    latent_channels: int


# configs/{fcp,tf}_wan21_t2v_1.3b.yaml: image_or_video_shape [1, 22, 16, 54, 96]; configs/{fcp,tf}_wan22_ti2v_5b.yaml:
# [1, 31, 48, 36, 64]. Both use model_kwargs.sink_size 3 and local_attn_size -1.
WAN21 = Family("Wan 2.1 1.3B", WAN21_13B, 22, 16, 768, 432, 8, 16)
WAN22 = Family("Wan 2.2 TI2V 5B", WAN22_5B, 31, 24, 1024, 576, 16, 48)
FAMILIES = (WAN21, WAN22)
# The released inference default (configs/default_config.yaml).
NEGATIVE_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
                   "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
                   "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
                   "background, three legs, many people in the background, walking backwards")
# Training-only tensors in the released checkpoints (src/utils/inference_checkpoint.py _AUX_NAMES).
AUX_NAMES = {"predictive_multi_chunk_branches", "predictive_multi_chunk_shared_direct_modules",
             "predictive_multi_chunk_source_layers_metadata", "predictive_history_context_marker",
             "predictive_offset_source_mode_marker", "predictive_source_unshared_scope_marker",
             "predictive_aux_shared_across_offsets_marker"}


def frame_counts(seconds: int = 0, pixel_frames: int = 0, family: Family = WAN21):
    """(pixel frames kept, latent frames generated incl. the conditioning frame), as src/utils/video_prefix.output_geometry."""
    if pixel_frames <= 0:
        pixel_frames = family.fps * seconds
    if pixel_frames < 2:
        raise ValueError("Looking Ahead needs at least 2 output frames")
    latent = 1 + math.ceil((pixel_frames - 1) / VAE_TEMPORAL_STRIDE)
    latent = 1 + math.ceil((latent - 1) / FRAMES_PER_BLOCK) * FRAMES_PER_BLOCK
    return pixel_frames, latent


def context_geometry(pixel_frames, latent_frames, context_latents=0, available_frames=0):
    """(K context latents, context pixel frames) for video continuation, as src/utils/video_prefix.prefix_geometry and, for
    context_latents=0, automatic_prefix_geometry (the longest aligned context that the video's frames at the model frame rate
    allow, leaving at least one generated chunk). The released recipes use an independent first frame: K = 1 + 3M."""
    if context_latents <= 0:
        if available_frames < 1:
            raise ValueError("the video has no frames at the model's frame rate")
        maximum = min(1 + (available_frames - 1) // VAE_TEMPORAL_STRIDE, 1 + (pixel_frames - 2) // VAE_TEMPORAL_STRIDE,
                      latent_frames - FRAMES_PER_BLOCK)
        context_latents = 1 + (maximum - 1) // FRAMES_PER_BLOCK * FRAMES_PER_BLOCK
        if context_latents < 1 + FRAMES_PER_BLOCK:
            raise ValueError("the video is too short for a context of at least 4 latent frames (13 frames at the model's "
                             "frame rate) followed by a generated chunk")
    elif context_latents == 1 or (context_latents - 1) % FRAMES_PER_BLOCK:
        raise ValueError(f"context_latents must be 1 + {FRAMES_PER_BLOCK}*M (4, 7, 10, ...), got {context_latents}")
    condition_frames = 1 + VAE_TEMPORAL_STRIDE * (context_latents - 1)
    if pixel_frames <= condition_frames or latent_frames <= context_latents:
        raise ValueError(f"the output ({pixel_frames} frames) must be longer than the context ({condition_frames} frames)")
    if (latent_frames - context_latents) % FRAMES_PER_BLOCK:
        raise ValueError("the generated part must be a whole number of 3-latent-frame chunks")
    return context_latents, condition_frames


def context_frame_indices(frame_count, source_fps, fps, condition_frames, automatic):
    """Which frames of a constant-frame-rate video starting at 0 s the context uses, as the official SingleVideoDataset
    samples them: the frames played at `condition_frames` times spaced 1/fps apart, ending at the last frame (automatic)
    or starting at 0 s. The times are exact fractions, so a time on the frame grid gets that frame. (Upstream computes
    them in floating point, which repeats 6 of the 25 frames of its 5B examples; TencentARC/Looking-Ahead#3.)"""
    rate, step = Fraction(source_fps).limit_denominator(100000), Fraction(1, fps)
    last = (frame_count - 1) / rate
    if automatic:
        times = [max(Fraction(0), last - (condition_frames - 1 - i) * step) for i in range(condition_frames)]
    else:
        times = [i * step for i in range(condition_frames)]
        if times[-1] >= frame_count / rate:
            raise ValueError("the video is too short for the requested context at the model's frame rate")
    return [min(frame_count - 1, math.floor(t * rate)) for t in times]


def available_frames(frame_count, source_fps, fps):
    """Frames at the model frame rate up to the video's last frame (SingleVideoDataset's `available`)."""
    return math.floor((frame_count - 1) / Fraction(source_fps).limit_denominator(100000) * fps) + 1

def backbone_state_dict(payload):
    """The EMA generator of a released model_ema.pt as CausalWanModel keys, without the training-only tensors."""
    state = payload.get("generator_ema", payload)
    out = {}
    for key, value in state.items():
        while key.startswith("_fsdp_wrapped_module."):
            key = key[len("_fsdp_wrapped_module."):]
        key = key.replace("._fsdp_wrapped_module.", ".")
        if any(part in AUX_NAMES for part in key.split(".")):
            continue
        if not key.startswith("model."):
            raise ValueError(f"unexpected checkpoint key {key!r}: not a Looking-Ahead generator")
        out[key[len("model."):]] = value
    return out


def family_of(state_dict):
    """The model family of a backbone state dict, from the patch embedding's [dim, in_dim, 1, 2, 2] shape."""
    weight = state_dict.get("patch_embedding.weight")
    for family in FAMILIES:
        if weight is not None and tuple(weight.shape[:2]) == (family.config["dim"], family.config["in_dim"]):
            return family
    shape = None if weight is None else tuple(weight.shape)
    raise ValueError(f"not a Looking Ahead 1.3B (Wan 2.1) or 5B (Wan 2.2 TI2V) checkpoint: patch embedding {shape}")


def rope_freqs(model):
    d = model.dim // model.num_heads
    return torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)


def build_model(state_dict, dtype=torch.bfloat16, config=WAN21_13B, train_latent_frames=WAN21.train_latent_frames,
                sink_size=SINK_SIZE):
    """CausalWanModel with the given backbone weights (strict), on CPU in `dtype`."""
    with torch.device("meta"):
        model = CausalWanModel(**config, local_attn_size=train_latent_frames, sink_size=sink_size,
                               train_frame_size=train_latent_frames)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.freqs = rope_freqs(model)  # a plain attribute (not a parameter), built on the meta device above
    model.num_frame_per_block = FRAMES_PER_BLOCK
    model.independent_first_frame = True
    return model.to(dtype=dtype).eval().requires_grad_(False)


def load_checkpoint(path, dtype=torch.bfloat16):
    """The model of a released model_ema.pt, with its Family as `model.family`."""
    path = Path(path)
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    state = backbone_state_dict(payload)
    try:
        family = family_of(state)
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from None
    model = build_model(state, dtype=dtype, config=family.config, train_latent_frames=family.train_latent_frames)
    model.family = family
    del payload, state
    return model


@dataclass
class Caches:
    kv_pos: list
    kv_neg: list
    cross_pos: list
    cross_neg: list


def make_caches(model, batch_size, frame_seq_len, dtype, device):
    heads, head_dim, blocks = model.num_heads, model.dim // model.num_heads, model.num_layers
    size = model.local_attn_size * frame_seq_len

    def kv():
        return [{"k": torch.zeros([batch_size, size, heads, head_dim], dtype=dtype, device=device),
                 "v": torch.zeros([batch_size, size, heads, head_dim], dtype=dtype, device=device),
                 "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                 "local_end_index": torch.tensor([0], dtype=torch.long, device=device)} for _ in range(blocks)]

    def cross():
        return [{"k": torch.zeros([batch_size, model.text_len, heads, head_dim], dtype=dtype, device=device),
                 "v": torch.zeros([batch_size, model.text_len, heads, head_dim], dtype=dtype, device=device),
                 "is_init": False} for _ in range(blocks)]

    return Caches(kv(), kv(), cross(), cross())


def flow(model, x, timestep, context, kv_cache, crossattn_cache, current_start, cache_start):
    """WanDiffusionWrapper.forward with KV caches: [B, T, C, H, W] in, flow prediction [B, T, C, H, W] out."""
    out = model(x.permute(0, 2, 1, 3, 4), t=timestep, context=context, kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, cache_start=cache_start)
    return out.permute(0, 2, 1, 3, 4)


@torch.no_grad()
def sample(model, positive, negative, initial_latent, noise, steps=50, cfg=3.0, shift=5.0, callback=None):
    """Port of CausalDiffusionInferencePipeline.inference() (independent first frame) for TI2V and TV2V.

    positive / negative: prompt embeddings [B, text_len, 4096] (padding zeroed); initial_latent: [B, K, C, H, W], the start
    image (K = 1) or an encoded video prefix (K = 1 + 3M); noise: [B, T_gen, C, H, W] with T_gen a multiple of 3. The first
    latent frame, then each 3-frame block of the prefix, fills both KV caches at timestep 0 before the generated chunks.
    Returns the latents [B, K + T_gen, C, H, W]. callback(done, total) is called after every denoising step and may raise
    to interrupt.
    """
    batch_size, num_frames, channels, height, width = noise.shape
    num_input_frames = initial_latent.shape[1]
    if num_frames % FRAMES_PER_BLOCK:
        raise ValueError(f"generated latent frames must be a multiple of {FRAMES_PER_BLOCK}, got {num_frames}")
    if (num_input_frames - 1) % FRAMES_PER_BLOCK:
        raise ValueError(f"the conditioning latents must be 1 + {FRAMES_PER_BLOCK}*M frames, got {num_input_frames}")
    device, dtype = noise.device, noise.dtype
    frame_seq_len = height * width // 4
    caches = make_caches(model, batch_size, frame_seq_len, dtype, device)
    output = torch.zeros([batch_size, num_frames + num_input_frames, channels, height, width], device=device, dtype=dtype)
    timestep = torch.zeros([batch_size, 1], device=device, dtype=torch.int64)
    output[:, :1].copy_(initial_latent[:, :1])
    for context, kv, cross in ((positive, caches.kv_pos, caches.cross_pos), (negative, caches.kv_neg, caches.cross_neg)):
        flow(model, initial_latent[:, :1], timestep * 0, context, kv, cross, 0, 0)
    current_start = cache_start = 1
    for _ in range((num_input_frames - 1) // FRAMES_PER_BLOCK):
        block = initial_latent[:, cache_start:cache_start + FRAMES_PER_BLOCK]
        output[:, cache_start:cache_start + FRAMES_PER_BLOCK].copy_(block)
        for context, kv, cross in ((positive, caches.kv_pos, caches.cross_pos), (negative, caches.kv_neg, caches.cross_neg)):
            flow(model, block, timestep * 0, context, kv, cross, current_start * frame_seq_len, cache_start * frame_seq_len)
        current_start += FRAMES_PER_BLOCK
        cache_start += FRAMES_PER_BLOCK
    total = num_frames // FRAMES_PER_BLOCK * steps
    done = 0
    for _ in range(num_frames // FRAMES_PER_BLOCK):
        latents = noise[:, cache_start - num_input_frames:cache_start - num_input_frames + FRAMES_PER_BLOCK]
        scheduler = FlowUniPCMultistepScheduler(shift=1)
        scheduler.set_timesteps(steps, device=device, shift=shift)
        for t in scheduler.timesteps:
            timestep = t * torch.ones([batch_size, FRAMES_PER_BLOCK], device=device, dtype=torch.float32)
            cond = flow(model, latents, timestep, positive, caches.kv_pos, caches.cross_pos,
                        current_start * frame_seq_len, cache_start * frame_seq_len)
            uncond = flow(model, latents, timestep, negative, caches.kv_neg, caches.cross_neg,
                          current_start * frame_seq_len, cache_start * frame_seq_len)
            latents = scheduler.step(uncond + cfg * (cond - uncond), t, latents, return_dict=False)[0]
            done += 1
            if callback is not None:
                callback(done, total)
        output[:, cache_start:cache_start + FRAMES_PER_BLOCK].copy_(latents)
        for context, kv, cross in ((positive, caches.kv_pos, caches.cross_pos), (negative, caches.kv_neg, caches.cross_neg)):
            flow(model, latents, timestep * 0, context, kv, cross, current_start * frame_seq_len, cache_start * frame_seq_len)
        current_start += FRAMES_PER_BLOCK
        cache_start += FRAMES_PER_BLOCK
    return output


def initial_noise(seed, shape, device):
    """inference_runtime.run_inference: torch.Generator(device).manual_seed(seed % 2**63), bf16 randn on the device."""
    generator = torch.Generator(device=device).manual_seed(int(seed) % (2 ** 63))
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)


def kv_cache_bytes(model, latent_height, latent_width, batch_size=1, dtype=torch.bfloat16):
    """GPU memory of the self-attention KV caches (positive and negative) for one generation."""
    frame_seq_len = latent_height * latent_width // 4
    per_tensor = batch_size * model.local_attn_size * frame_seq_len * model.dim * torch.finfo(dtype).bits // 8
    return per_tensor * 2 * 2 * model.num_layers
