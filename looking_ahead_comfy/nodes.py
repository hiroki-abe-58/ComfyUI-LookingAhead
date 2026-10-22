"""ComfyUI nodes for Looking Ahead. Everything runs inside the ComfyUI process; nothing is downloaded at run time."""

from __future__ import annotations

import json
import logging
import os
from fractions import Fraction

from comfy_api.latest import ComfyExtension, InputImpl, Types, io

from . import paths, runtime

LookingAheadModel = io.Custom("LOOKING_AHEAD_MODEL")
log = logging.getLogger("LookingAhead")

# One checkpoint stays in RAM (bf16, about 2.8 GB); loading another replaces it.
_resident = {}


def _load(path):
    import torch

    key = (os.path.realpath(path), os.path.getsize(path), os.path.getmtime(path))
    if key not in _resident:
        unload()
        log.info("loading %s", path)
        _resident[key] = runtime.load_checkpoint(path, dtype=torch.bfloat16)
    return key


def _model(handle):
    key = tuple(handle["key"])
    if key not in _resident:  # unloaded since the loader ran: load again from the same file
        _load(key[0])
    return _resident[key]


def unload():
    import comfy.model_management as mm

    freed = [os.path.basename(k[0]) for k in _resident]
    _resident.clear()
    mm.soft_empty_cache()
    return freed


def _embeddings(conditioning, name):
    """The prompt embeddings of a umT5 (Wan) CONDITIONING, padded to the model's 512 text positions."""
    import torch

    if not conditioning:
        raise ValueError(f"{name}: empty conditioning")
    if len(conditioning) > 1:
        log.warning("%s: %d conditionings given; only the first is used", name, len(conditioning))
    embeds = conditioning[0][0]
    if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != 4096:
        raise ValueError(f"{name}: expected umT5-XXL text embeddings [1, tokens, 4096] (a Wan text encoder in CLIPLoader), "
                         f"got {tuple(embeds.shape)}")
    if embeds.shape[1] > 512:
        raise ValueError(f"{name}: the prompt is {embeds.shape[1]} tokens; the model reads at most 512")
    return torch.nn.functional.pad(embeds, (0, 0, 0, 512 - embeds.shape[1]))


def _prepare_image(image, width, height):
    """The official single-image preprocessing (vendor/dataset/image_preprocessing.py): cover the target size with an
    antialiased resize, then crop the centre. Returns a ComfyUI IMAGE [1, H, W, 3] and the geometry report."""
    import numpy as np
    import torch
    from PIL import Image

    from .vendor.dataset.image_preprocessing import resize_center_crop

    if image.shape[0] > 1:
        log.warning("start_image is a batch of %d; the first image is used", image.shape[0])
    rgb = image[0, :, :, :3].clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    resized, report = resize_center_crop(Image.fromarray(rgb), height, width)
    return torch.from_numpy(np.asarray(resized).copy()).float().div(255).unsqueeze(0), report


class LookingAheadModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LookingAheadModelLoader",
            display_name="Looking Ahead Model Loader",
            category="loaders/looking-ahead",
            description=("Loads a Looking Ahead 1.3B checkpoint (TencentARC/Looking-Ahead, fcp_wan21_t2v_1.3b or tf_wan21_t2v_1.3b "
                         "model_ema.pt) from models/looking_ahead as bfloat16. Training-only branches are dropped. One model "
                         "stays in RAM; loading another frees it."),
            inputs=[io.Combo.Input("checkpoint", options=paths.checkpoints(),
                                   tooltip="model_ema.pt in models/looking_ahead, e.g. fcp_wan21_t2v_1.3b/model_ema.pt")],
            outputs=[LookingAheadModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, checkpoint) -> io.NodeOutput:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Looking Ahead for ComfyUI needs a CUDA GPU")
        return io.NodeOutput({"key": list(_load(paths.resolve(checkpoint))), "name": checkpoint})


class LookingAheadImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LookingAheadImageToVideo",
            display_name="Looking Ahead Image to Video",
            category="video/looking-ahead",
            description=("Text-and-image-to-video with Looking Ahead 1.3B: the start image is resized to cover the frame and "
                         "center-cropped like the official inference, then the video is generated autoregressively in "
                         "3-latent-frame chunks (KV cache, 16 fps). The released models were trained on 5-second clips."),
            inputs=[
                LookingAheadModel.Input("model"),
                io.Conditioning.Input("positive", tooltip="CLIPTextEncode with a umT5-XXL (Wan) text encoder"),
                io.Conditioning.Input("negative", tooltip="The official default negative prompt is in the example workflow"),
                io.Vae.Input("vae", tooltip="The Wan 2.1 VAE (wan_2.1_vae.safetensors)"),
                io.Image.Input("start_image"),
                io.Int.Input("seconds", default=5, min=1, max=30,
                             tooltip="Output length including the start frame, at 16 fps. Longer than 5 s uses the rolling "
                                     "KV cache with a 3-frame attention sink; quality beyond 5 s is not guaranteed upstream."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=200, tooltip="Denoising steps per chunk (official: 50)"),
                io.Float.Input("cfg", default=3.0, min=0.0, max=20.0, step=0.1, tooltip="Official: 3.0"),
                io.Float.Input("shift", default=5.0, min=0.1, max=20.0, step=0.1, tooltip="Timestep shift (official: 5.0)"),
                io.Int.Input("width", default=runtime.WIDTH, min=256, max=1280, step=16, tooltip="Trained at 768x432"),
                io.Int.Input("height", default=runtime.HEIGHT, min=256, max=1280, step=16, tooltip="Trained at 768x432"),
                io.Boolean.Input("keep_model_loaded", default=False,
                                 tooltip="Keep the model on the GPU after sampling (faster reruns, about 2.8 GB of VRAM)"),
            ],
            outputs=[io.Video.Output(display_name="video"), io.Image.Output(display_name="images"),
                     io.Latent.Output(display_name="latent"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, positive, negative, vae, start_image, seconds, seed, steps, cfg, shift, width, height,
                keep_model_loaded) -> io.NodeOutput:
        import time

        import comfy.latent_formats
        import comfy.model_management as mm
        import comfy.utils
        import torch

        device = mm.get_torch_device()
        pixel_frames, latent_frames = runtime.frame_counts(seconds=seconds)
        pos = _embeddings(positive, "positive").to(device, torch.bfloat16)
        neg = _embeddings(negative, "negative").to(device, torch.bfloat16)
        image, geometry = _prepare_image(start_image, width, height)
        fmt = comfy.latent_formats.Wan21()
        raw = vae.encode(image[:, :, :, :3])
        if raw.ndim != 5 or raw.shape[1] != runtime.LATENT_CHANNELS:
            raise ValueError(f"vae: expected the Wan 2.1 VAE (16-channel video latents), got latents {tuple(raw.shape)}")
        initial = fmt.process_in(raw.float()).permute(0, 2, 1, 3, 4).to(device, torch.bfloat16)
        _, _, _, lat_h, lat_w = initial.shape

        net = _model(model)
        param_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
        mm.free_memory(param_bytes + runtime.kv_cache_bytes(net, height, width) + 4 * 2**30, device)
        net.to(device)
        bar = comfy.utils.ProgressBar((latent_frames - 1) // runtime.FRAMES_PER_BLOCK * steps)

        def progress(done, total):
            mm.throw_exception_if_processing_interrupted()
            bar.update_absolute(done, total)

        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            noise = runtime.initial_noise(seed, (1, latent_frames - 1, runtime.LATENT_CHANNELS, lat_h, lat_w), device)
            latents = runtime.sample(net, pos, neg, initial, noise, steps=steps, cfg=cfg, shift=shift, callback=progress)
        finally:
            if not keep_model_loaded:
                net.to(mm.unet_offload_device())
                mm.soft_empty_cache()
        sample_s = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)

        samples = fmt.process_out(latents.float().permute(0, 2, 1, 3, 4)).cpu()
        images = vae.decode(samples)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        images = images[:pixel_frames]
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=None, frame_rate=Fraction(runtime.FPS)))
        report = {"checkpoint": model.get("name"), "frames": int(images.shape[0]), "fps": runtime.FPS,
                  "latent_frames": latent_frames, "size": [width, height], "seed": seed, "steps": steps, "cfg": cfg,
                  "shift": shift, "sampling_seconds": round(sample_s, 1), "peak_allocated_gib": round(peak / 2**30, 2),
                  "start_image": geometry}
        return io.NodeOutput(video, images, {"samples": samples}, json.dumps(report, indent=1))


class LookingAheadUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LookingAheadUnload",
            display_name="Looking Ahead Unload",
            category="loaders/looking-ahead",
            description="Frees the resident Looking Ahead model (GPU and RAM).",
            inputs=[io.Image.Input("after", optional=True, tooltip="Optional: connect the images so this runs after them")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        text = json.dumps({"unloaded": unload()}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class LookingAheadExtension(ComfyExtension):
    async def get_node_list(self):
        return [LookingAheadModelLoader, LookingAheadImageToVideo, LookingAheadUnload]


async def comfy_entrypoint() -> LookingAheadExtension:
    paths.register()
    return LookingAheadExtension()
