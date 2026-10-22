# Verification

All GPU measurements: Windows 11, RTX 5090 32 GB, PyTorch 2.10.0+cu128, ComfyUI master 1a82e6d0 (2026-10-22).
Reference: the official inference (`inference.py`) of TencentARC/Looking-Ahead at 210794c with the FCP 1.3B EMA checkpoint
(`fcp_wan21_t2v_1.3b/model_ema.pt`, Hub revision 91b4b2e8, SHA-256 415946da…). The settings are the README example: sailboat
image, its prompt, 5 s, seed 42, 50 steps, CFG 3.0, shift 5.0, 768×432.

On Windows, the official code needs the changes in [TencentARC/Looking-Ahead#2](https://github.com/TencentARC/Looking-Ahead/pull/2)
to run; it otherwise stops at `import fcntl`. The reference runs used that branch with PyTorch SDPA. Where FlashAttention 4
imports, as on Linux, the branch takes the same code path as upstream.

## 1. Sampler: bit-identical

The official run's own inputs were saved (`CausalDiffusionInferencePipeline.inference` arguments: prompt and negative
embeddings, encoded first frame, noise). Fed into this pack's `runtime.sample()`, they give output latents that are
**bit-identical** to the official ones (22 latent frames, max abs difference 0). Sampling took 209.5 s, the same as the
official SDPA run (209 s). Peak allocated memory was 13.5 GiB, against 16.4 GiB for the official script, which also keeps
the text encoder and VAE on the GPU.

CI runs the same comparison on every push, on CPU: the official pipeline and `runtime.sample()` run a tiny random-initialized
Wan backbone with identical weights and inputs (`tests/test_runtime.py`). There are two cases, one KV-cache window, and a
rolling cache with the 3-frame attention sink. Both are bit-identical.

## 2. Start image and VAE

| check | result |
| --- | --- |
| Resize and center crop of a ComfyUI `IMAGE` against the official PIL path (four aspect ratios, `tests/test_nodes.py`) | identical pixels and identical geometry report |
| ComfyUI `wan_2.1_vae.safetensors` encode of the sailboat start frame against the official encoder | relative L2 0.38 %, max abs 0.012 |
| ComfyUI VAE decode of the official output latents against the official decode | PSNR 52.1 dB mean (min 51.2), 85 frames |

## 3. Text encoder

The official T5 runs in bfloat16. Against a float32 run of the same official weights (relative L2 over the real tokens):

| embeddings | prompt (68 tokens) | negative prompt (98 tokens) |
| --- | --- | --- |
| official bfloat16 | 2.15 % | 2.25 % |
| ComfyUI `CLIPLoader` (type `wan`), the official weights renamed to ComfyUI keys, bfloat16 | 0.17 % | 0.17 % |
| ComfyUI, `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 5.87 % | 9.31 % |

Token counts are the same as the official tokenizer, and the padding is zeroed the same way, so ComfyUI's text encoding is the
closer one to float32. The official run's own embeddings are what the bit-identical check in section 1 uses.

## 4. End to end through a real ComfyUI queue

These runs used `LoadImage`, `CLIPLoader`, `CLIPTextEncode` ×2, `VAELoader`, the pack's loader and sampler, `SaveVideo` and
`SaveLatent`. The comparison is against the official SDPA run.

| text encoder file | output latents vs official (relative L2) | decoded video vs official (PSNR, mean / min) |
| --- | --- | --- |
| official weights renamed to ComfyUI keys (bf16) | 4.1 % | 28.7 / 27.4 dB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.8 % | 28.0 / 26.2 dB |

The videos show the same scene, composition and motion as the official one (the boat drifts the same way across the lake).
The remaining difference is the sampler's sensitivity, not a mismatch. With the official noise fixed, changing one input at a
time gives:

| input from ComfyUI, the rest official | output latents vs official |
| --- | --- |
| first-frame latent only (0.38 % different) | 3.5 % |
| text embeddings only (bf16, official weights) | 4.1 % |
| both (the real queue) | 4.1 % |

A small change in any input grows to about 3.5–4 % over 21 latent frames × 50 steps of autoregressive sampling.

Other checks in the same queue:

| check | result |
| --- | --- |
| the same workflow queued twice | bit-identical latents |
| `/interrupt` during sampling | stopped within a step (26 s after queueing); the next prompt ran normally |
| 10 s output | 160 frames; 515 s of sampling; peak allocated 13.6 GiB |
| Looking Ahead Unload | frees the model; system commit 43.6 GiB afterwards, against 55.8 GiB during the run before it |
| three example scenes (sailboat, cat, pendulum) | coherent videos; the pendulum swings smoothly from left, through the bottom, to the right |

## 5. Clean install

The setup was a fresh clone of ComfyUI v0.42.1 and a new virtual environment (PyTorch 2.10.0+cu128 and ComfyUI's
requirements, nothing else), with the pack cloned from GitHub into `custom_nodes`. `tools/download_model.py` found the
checkpoint already in place and accepted its SHA-256. Queued unchanged, `workflows/api/looking_ahead_image_to_video_api.json`
produced the same decoded frames, bit for bit, as the development setup on ComfyUI master. The start image was the kite demo
image. The sailboat case with the fp8 text encoder also gave identical latents in both setups.

## 6. Performance (RTX 5090, Windows 11, SDPA)

| | 5 s (80 frames, 768×432) | 10 s (160 frames) |
| --- | --- | --- |
| sampling | 209–211 s | 515 s |
| whole queue, warm text encoder and VAE | 215–219 s | 523 s |
| peak allocated (torch) | 13.44 GiB | 13.61 GiB |
| peak Windows commit (whole system) | 55–60 GiB | 56 GiB |

The KV cache holds 22 latent frames for the positive and negative prompts: about 10.5 GiB at 768×432 in bfloat16. This is
most of the GPU memory. Each 3-latent-frame chunk is slower than the last while the cache fills: 16 s for the first and 43 s
for the seventh, in the official SDPA run. Past 5 s the cache rolls, and the chunk time stops growing.
