# Verification

All GPU measurements: Windows 11, RTX 5090 32 GB, PyTorch 2.10.0+cu128, ComfyUI master 1a82e6d0 (2026-10-22).
Reference: the official inference (`inference.py`) of TencentARC/Looking-Ahead at 210794c with the FCP 1.3B EMA checkpoint
(`fcp_wan21_t2v_1.3b/model_ema.pt`, Hub revision 91b4b2e8, SHA-256 415946da…). The settings are the README example: sailboat
image, its prompt, 5 s, seed 42, 50 steps, CFG 3.0, shift 5.0, 768×432.

On Windows, the official code needs the changes in [TencentARC/Looking-Ahead#2](https://github.com/TencentARC/Looking-Ahead/pull/2)
to run; it otherwise stops at `import fcntl`. The reference runs used that branch with PyTorch SDPA. With FlashAttention 4
made importable for a test, the branch gives the same frames as unmodified upstream (bit-identical), and FA4 against SDPA is
33.5 dB PSNR.

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

<!-- E2E -->
