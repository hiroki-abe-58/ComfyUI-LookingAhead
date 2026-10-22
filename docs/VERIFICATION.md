# Verification

All GPU measurements: Windows 11, RTX 5090 32 GB, PyTorch 2.10.0+cu128, ComfyUI master 1a82e6d0 (2026-10-22).
Reference: the official inference (`inference.py`) of TencentARC/Looking-Ahead at 210794c with the FCP 1.3B EMA checkpoint
(`fcp_wan21_t2v_1.3b/model_ema.pt`, Hub revision 91b4b2e8, SHA-256 415946da…). The settings are the README example: sailboat
image, its prompt, 5 s, seed 42, 50 steps, CFG 3.0, shift 5.0, 768×432.

Sections 1–6 are about this 1.3B model. Section 7 covers the 5B model the same way, and section 8 the video continuation of both.

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

`tools/download_model.py --tf` downloaded the teacher-forcing baseline (`tf_wan21_t2v_1.3b/model_ema.pt`) and checked its
SHA-256. In the same clean install it loaded and generated the sailboat example: 80 frames, 209.8 s of sampling, 13.44 GiB peak.
Being a different model, its video differs from FCP's (23.9 dB).

## 6. Performance (RTX 5090, Windows 11, SDPA)

| | 5 s (80 frames, 768×432) | 10 s (160 frames) |
| --- | --- | --- |
| sampling | 209–211 s | 515 s |
| whole queue, warm text encoder and VAE | 215–219 s | 523 s |
| peak allocated (torch) | 13.44 GiB | 13.61 GiB |
| peak Windows commit (whole system) | 55–60 GiB | 56 GiB |

The KV cache holds 22 latent frames for the positive and negative prompts: about 9.8 GiB at 768×432 in bfloat16. This is
most of the GPU memory. Each 3-latent-frame chunk is slower than the last while the cache fills: 16 s for the first and 43 s
for the seventh, in the official SDPA run. Past 5 s the cache rolls, and the chunk time stops growing.

## 7. The 5B model (Wan 2.2 TI2V)

Reference: the official inference with the FCP 5B EMA checkpoint (`fcp_wan22_ti2v_5b/model_ema.pt`, Hub revision 91b4b2e8,
SHA-256 0c47d3fe…), on the paper-fan example image and prompt of the Looking-Ahead repository: 5 s (120 frames at 24 fps),
1024×576, seed 42, 50 steps, CFG 3.0, shift 5.0. It ran on the TencentARC/Looking-Ahead#2 branch with SDPA, as in section 1.

On this machine (RTX 5090 32 GB, 64 GB RAM), the official script's VAE decode did not fit right after sampling. The generator
and both KV caches were still on the GPU, VRAM filled up, and the Windows commit charge passed 104 GiB, where I stopped the
run. The reference run therefore stopped after sampling and saved its tensors. The official VAE wrapper then decoded the saved
latents in a separate process, with the same call (`decode_to_pixel`, bfloat16) the pipeline makes: 121 frames, 11.1 s,
8.2 GiB.

| check | result |
| --- | --- |
| sampler fed the official run's inputs | **bit-identical** latents (31 latent frames, max abs difference 0); 231.7 s against the official 231 s; peak allocated 22.95 GiB against 24.29 GiB |
| the node's noise for seed 42 | identical to the official noise |
| start image resize and crop | the same geometry as the official report (1672×941 → 1024×577 → crop to 1024×576) |
| ComfyUI `wan2.2_vae.safetensors` encode of the start frame | relative L2 0.82 %, max abs 0.030 |
| ComfyUI VAE decode of the official latents | PSNR 60.3 dB mean (min 58.0), 121 frames |
| text encoder against the official bfloat16 embeddings | official weights in ComfyUI: 3.3 % (prompt, 46 tokens) / 2.3 % (negative, 98 tokens); fp8 file: 7.9 % / 9.5 % |

CI runs the CPU comparison of section 1 with a tiny Wan 2.2 TI2V shaped model as well (48-channel latents, rolling cache):
bit-identical.

### End to end

The real ComfyUI queue used the shipped 5B graph, with the paper-fan image and prompt.

| text encoder file | output latents vs official (relative L2) | decoded video vs official (PSNR mean / min) |
| --- | --- | --- |
| official weights renamed to ComfyUI keys (bf16) | 57.2 % | 25.2 / 12.9 dB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 59.7 % | 23.3 / 12.7 dB |

This is much further from the official run than with 1.3B (4.1 %), so I checked where it comes from:

- **Growth over time.** The difference grows steadily with time. It is 0.8 % on the start frame (the VAE encode). It stays under 5 % for the first second, reaches about 40 % at 2.5 s and about 90 % on the last latent frames.
- **What the frames show.** For the first half the videos are nearly the same: the same hands open the same red fan the same way. In the second half both open the fan into a full semicircle, as the prompt asks, but the fan's size and the hands' positions differ.
- **One input at a time.** With the official noise fixed (the node's noise for seed 42 is identical to it), changing one input at a time gives:

| input from ComfyUI, the rest official | output latents vs official |
| --- | --- |
| first-frame latent only (0.82 % different) | 59.0 % |
| text embeddings only (bf16, official weights) | 59.9 % |
| both | 57.2 %, and **bit-identical to the real queue's latents** |

So the queue computes exactly what the sampler computes from ComfyUI's encoded inputs. Each small input difference on its own grows to about 60 % over 30 latent frames × 50 steps. For 1.3B the same first-frame test gives 3.5 %. The 5B model's sampling is far more sensitive to its inputs: ComfyUI reproduces the scene and the action, but not the official frames.

### Other checks in the 5B queue

| check | result |
| --- | --- |
| the same 5B workflow in two separate ComfyUI processes | bit-identical latents (a repeat in the same process is served from ComfyUI's cache and proves nothing, so the second run was in a new process) |
| `wan_2.1_vae.safetensors` with the 5B model | stops before sampling: "needs the Wan 2.2 VAE (wan2.2_vae.safetensors) with 48-channel video latents" |
| 768×432 with the 5B model | stops before sampling: "width and height must be multiples of 32" |
| the 1.3B sailboat case right after the 5B runs, in the same ComfyUI process | latents bit-identical to 0.1.1 |
| 10 s output | 240 frames, 556 s of sampling, peak allocated 23.15 GiB. Past 5 s the fan keeps widening and its paper turns blotchy; upstream does not guarantee quality beyond the 5 s training length. |
| two more example scenes of the Looking-Ahead repository | can crush: the hand crushes the can, but the dents partly smooth out in the last second. Book: the hand leaves the frame, and then a page turns by itself. |
| Looking Ahead Unload | frees the 5B model |
| 5B TF baseline | `tools/download_model.py --5b --tf` downloaded `tf_wan22_ti2v_5b/model_ema.pt` (20.0 GB) and checked its SHA-256. In the clean install it loaded and generated the paper-fan example: 120 frames at 24 fps, 252.9 s for the whole prompt. Being a different model, its video differs from FCP's (18.0 dB); its fan's ribs come apart while opening. |
| clean install (section 5's fresh ComfyUI v0.42.1, the pack cloned again from GitHub) | `tools/download_model.py --5b --wan` accepted the checkpoint and text encoder by SHA-256 and downloaded the Wan 2.2 VAE. Queued unchanged with the kite start image, `workflows/api/looking_ahead_5b_image_to_video_api.json` gave the same decoded frames, bit for bit, as the development setup on ComfyUI master. |

| | 5 s (120 frames, 1024×576) | 10 s (240 frames) |
| --- | --- | --- |
| sampling | 232.8–233.5 s | 556 s |
| whole queue, warm text encoder and VAE | 248–254 s | 591 s |
| peak allocated (torch) | 22.94 GiB | 23.15 GiB |
| peak Windows commit (whole system) | 71–76 GiB, while loading | 84 GiB, while loading |

The model is 9.3 GiB in bfloat16. The KV cache holds 31 latent frames for both prompts, 12.3 GiB at 1024×576. Loading
memory-maps the 34 GB float32 checkpoint and keeps a bfloat16 copy of the backbone, which is why the commit charge peaks while
loading. During sampling it was about 60 GiB.

## 8. Video continuation (TV2V)

Reference: the official inference with `--video_path` (automatic context), on `milk_pouring.mp4` of the Looking-Ahead
repository (25 frames, 24 fps, 1024×576) and its prompt, 5 s, seed 42.
- **1.3B:** the context is 13 frames at 16 fps (K = 4), followed by 67 generated frames.
- **5B:** the context is 25 frames at 24 fps (K = 7), followed by 95 generated frames; the decode was again done in a separate process.
- **Code:** TencentARC/Looking-Ahead#2 for Windows, and for 5B also #3. Without #3, the official 5B context repeats 6 of its 25 frames, as the README explains.

| check | result |
| --- | --- |
| sampler fed the official run's inputs (context latents, embeddings, noise) | **bit-identical**: 1.3B 22 latent frames, 5B 31; sampling 193 s and 202 s |
| CPU comparison with the official pipeline (`tests/test_runtime.py`) | bit-identical with a 4-latent context (Wan 2.1) and a 7-latent context with a rolling cache (Wan 2.2) |
| context length and frame choice against `src/utils/video_prefix.py` and `SingleVideoDataset` | the same K and context length in 6440 combinations of model, output length, available frames and requested K. The same source frames as torchcodec's `get_frames_played_at` in every case checked (both example videos, 16 and 24 fps, automatic and fixed K), with #3's time expression |
| context encoded by ComfyUI (`LoadVideo` decode → frame choice → crop → ComfyUI VAE) | 5.7 % (1.3B) and 12.3 % (5B) relative L2 from the official context latents |

Almost all of that context difference comes from decoding. ComfyUI decodes video to float RGB, while torchcodec (and PyAV's
8-bit RGB) convert to 8-bit RGB. Both use the same BT.601 matrix but interpolate chroma differently: on the example the frames
differ by 0.75 levels on average and up to 33 near colour edges. Fed the official decoder's frames instead, the same ComfyUI
path is 0.42 % (1.3B) and 0.71 % (5B) from the official context, the same as for a start image. The node keeps ComfyUI's
decode, so that it works with ComfyUI's own video loading, trimming and cropping.

End to end, through the real queue:

| model, text encoder | output latents vs official | decoded video vs official (PSNR mean / min) |
| --- | --- | --- |
| 1.3B, official weights (bf16) | 12.1 % | 36.5 / 27.9 dB |
| 1.3B, fp8 | 10.3 % | 36.5 / 27.0 dB |
| 5B, official weights (bf16) | 20.5 % | 33.0 / 26.2 dB |
| 5B, fp8 | 20.6 % | 33.1 / 26.4 dB |

The milk pours and fills the glass the same way, with the same splashes. The context anchors the motion: with 5B, the
start-image mode differed by 57 % (section 7). With the official noise fixed, one input at a time:

| input from ComfyUI, the rest official | 1.3B | 5B |
| --- | --- | --- |
| context latents only | 13.5 % | 20.5 % |
| text embeddings only (bf16; this prompt is 186 tokens, 6.6 % from the official embeddings) | 11.2 % | 13.9 % |
| both | 12.1 %, **bit-identical to the real queue** | 20.5 %, **bit-identical to the real queue** |

Other checks in the same queue:

| check | result |
| --- | --- |
| `context_latents` = 3 | stops: "context_latents must be 1 + 3*M (4, 7, 10, ...)" |
| `context_latents` = 4 (from the start of the video) | source frames 0, 1, 3, 4, … 18 (every 1/16 s), 80 output frames |
| `include_context` off | the 67 new frames only; the latents are the same as with the context |
| Image to Video after the shared-code change | latents bit-identical to before for 1.3B (0.1.1) and 5B (0.2.0) |
| speed | 1.3B: 193 s of sampling for 18 generated latent frames, 13.44 GiB peak; 5B: 203 s for 24, 22.94 GiB |
