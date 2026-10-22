# Looking Ahead for ComfyUI

Unofficial ComfyUI nodes for **Looking Ahead** ([paper](https://arxiv.org/abs/2610.29989),
[code](https://github.com/TencentARC/Looking-Ahead), [weights](https://huggingface.co/TencentARC/Looking-Ahead)), the
Future Chunk Prediction (FCP) autoregressive video model from Tencent ARC Lab, Nanjing University and Zhejiang University.
The pack runs both released text-and-image-to-video models, 1.3B (built on Wan 2.1) and 5B (built on Wan 2.2 TI2V): give it
a start image and a prompt, and it generates the video in 3-latent-frame chunks with a KV cache, exactly like the official
inference.

[日本語の説明はこちら](README.ja.md)

![Four 5-second clips made with this node and the 1.3B model](docs/images/demo_grid.gif)

These four 1.3B clips are each the first try at seed 42, at the default settings (768×432, 16 fps, 50 steps). They are shown as
they came out, scaled down. The windmill turns and the train crosses the bridge. The dog runs along the beach, but loses the toy
in its mouth. The kite is a miss: it falls out of the frame instead of climbing. The start images were made for this demo with
MOSS-Image (Apache-2.0); prompts and details are in [docs/images](docs/images/README.md).

![The same four start images and prompts with the 5B model](docs/images/demo_grid_5b.gif)

The same start images and prompts with the 5B model (1024×576, 24 fps), again the first try at seed 42. The kite now stays in
the air and the train crosses the bridge. The windmill's sails smear in some frames, and the dog loses the toy again.

![Video continuation of the 5B train clip, with 5B and 1.3B](docs/images/demo_continuation.gif)

Video continuation: the 5B train clip above, continued to 10 s by both models with the shipped workflows (first try, seed 42).
The label says whether a frame is input or generated. Both keep the train moving across the bridge. Later, the 5B cars pick up
red details and the 1.3B cars blur and turn brown.
## What is checked

- **Sampler.** Fed the official run's prompt embeddings, first-frame latent and noise, the node's sampler returns latents that
  are bit-identical to the official `CausalDiffusionInferencePipeline`, for both models (5 s, seed 42, 50 steps, CFG 3.0, on an
  RTX 5090; 1.3B at 768×432, 5B at 1024×576). CI repeats this comparison on every push, on CPU, with tiny random models inside
  the official pipeline. One case fits in the training window; the others roll the KV cache with the attention sink, with
  Wan 2.1 and with Wan 2.2 latents.
- **Start image.** The start image goes through the official resize-and-center-crop. The pixels match exactly (tested).
  ComfyUI's VAE encodes it to within 0.4 % (Wan 2.1) and 0.8 % (Wan 2.2) relative L2 of the official encoder.
- **Text encoder.** ComfyUI's umT5-XXL (Wan) gives the same token count as the official one. With the same weights it is closer
  to a float32 reference than the official bfloat16 path: 0.17 % against 2.1 %. The common fp8 file is 6–9 % from it.
- **End to end, 1.3B.** A real ComfyUI queue reproduces the official video, with the same scene, composition and motion. Frame
  PSNR is 28.7 dB with the official text-encoder weights and 28.0 dB with the fp8 file. The difference comes from the
  autoregressive sampler amplifying small input changes: the 0.4 % first-frame difference alone moves the output latents by
  3.5 %, close to the 4.1 % of the whole run.
- **End to end, 5B.** The 5B model amplifies input differences far more. The ComfyUI video matches the official one closely for
  about the first second. After that it shows the same scene and action (the fan opens into a semicircle), but different
  frames: 25 dB mean PSNR, 57 % latent difference. Fed ComfyUI's encoded start frame and text, the sampler reproduces the
  queue bit for bit. Either input alone, swapped into the official run (0.8 % and about 3 % different), already moves the
  output by about 60 %.
- **Video continuation.** Fed the official TV2V run's context latents, embeddings and noise, the sampler is bit-identical to
  the official one for both models. The context length and frame choice match the official code. End to end, the continued
  milk pour matches the official one closely: 36.5 dB (1.3B) and 33 dB (5B) mean PSNR. Most of the remaining difference
  starts in ComfyUI's video decoder, which converts YUV to RGB slightly differently from the official one.
- **Robustness.** The same workflow queued twice gives bit-identical output. Interrupting stops sampling within a step, and the
  next prompt runs normally.

Details and numbers: [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Nodes

| node | what it does |
| --- | --- |
| **Looking Ahead Model Loader** | Loads `model_ema.pt` (FCP or the TF baseline, 1.3B or 5B) from `models/looking_ahead` as bfloat16 and drops the training-only branches. The model size is read from the weights. One model stays in RAM. |
| **Looking Ahead Image to Video** | Start image + positive/negative conditioning (umT5-XXL) + VAE (Wan 2.1 for 1.3B, Wan 2.2 for 5B) → `VIDEO` (16 fps for 1.3B, 24 fps for 5B), `IMAGE` frames, `LATENT` and a JSON report. Seed, seconds (1–30), steps, CFG, shift and size; it frees VRAM from other models first, shows progress and can be interrupted. |
| **Looking Ahead Video Continuation** | The official TV2V mode: a `VIDEO` (for example from `LoadVideo`) + conditioning + VAE → the same outputs as Image to Video. Frames of the input are taken at the model's frame rate, encoded as the context, and the rest of the output is generated after them. By default the context is the longest one the input allows, taken from its end, as upstream does; `context_latents` picks a fixed K = 4, 7, 10, … from its start instead. |
| **Looking Ahead Unload** | Frees the resident model. |

The text encoder and VAE are ComfyUI's own `CLIPLoader` (type `wan`) and `VAELoader`, so files you already use for Wan 2.1 and
Wan 2.2 work as they are. Both models use the same umT5-XXL text encoder.

## Install

ComfyUI 0.42.1 or newer (V3 node API). No extra Python packages.

```
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-LookingAhead
```

## Models

| file | folder | size | source |
| --- | --- | --- | --- |
| `fcp_wan21_t2v_1.3b/model_ema.pt` (1.3B) | `models/looking_ahead/` | 8.0 GB | [TencentARC/Looking-Ahead](https://huggingface.co/TencentARC/Looking-Ahead/tree/main/fcp_wan21_t2v_1.3b) |
| `fcp_wan22_ti2v_5b/model_ema.pt` (5B) | `models/looking_ahead/` | 34.2 GB | [TencentARC/Looking-Ahead](https://huggingface.co/TencentARC/Looking-Ahead/tree/main/fcp_wan22_ti2v_5b) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (or `umt5_xxl_fp16`), both models | `models/text_encoders/` | 6.7 GB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders) |
| `wan_2.1_vae.safetensors` (1.3B) | `models/vae/` | 0.25 GB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae) |
| `wan2.2_vae.safetensors` (5B) | `models/vae/` | 1.4 GB | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae) |

Keep each checkpoint in its own folder (`fcp_wan21_t2v_1.3b`, `fcp_wan22_ti2v_5b`). The 5B file is the released float32
checkpoint with its training-only branches; the loader keeps a 10 GB bfloat16 copy of the backbone. The teacher-forcing
baselines (`tf_wan21_t2v_1.3b`, 5.7 GB; `tf_wan22_ti2v_5b`, 20.0 GB; `--tf` in the script) load and run the same way; use
them to compare FCP against plain teacher forcing. Do not download the whole Looking-Ahead repository (four checkpoints,
about 68 GB) unless you want all of them.

Or let the script fetch exactly these files at pinned revisions and check their SHA-256 (run it with ComfyUI's Python):

```
python custom_nodes/ComfyUI-LookingAhead/tools/download_model.py --models-dir models --wan        # 1.3B
python custom_nodes/ComfyUI-LookingAhead/tools/download_model.py --models-dir models --5b --wan   # 5B
```

## Use

Load `workflows/gui/looking_ahead_image_to_video.json` (1.3B) or `workflows/gui/looking_ahead_5b_image_to_video.json` (5B),
pick a start image, and write what should happen in the prompt. The negative prompt in the workflows is the official default.
The workflows use the released settings: 5 s, 50 steps, CFG 3.0 and shift 5.0, at 768×432 and 16 fps for 1.3B, and at
1024×576 and 24 fps for 5B. The frame rate follows the loaded model.

- **Start image.** It is resized to cover the frame and center-cropped, as in the official inference; content at the edges
  can be cut off.
- **Length.** The models were trained on 5-second clips. Longer outputs use the rolling KV cache with a 3-frame attention sink,
  as the official code does. Upstream does not guarantee quality beyond 5 s.
- **Size.** Other sizes run, but 1.3B was trained at 768×432 and 5B at 1024×576. For 5B, width and height must be multiples
  of 32.

To continue a video, load `workflows/gui/looking_ahead_video_continuation.json` (1.3B) or
`workflows/gui/looking_ahead_5b_video_continuation.json` (5B), pick the video, and describe what happens next.

- **Length.** `seconds` is the whole output, context included, as the official `--duration`. With the automatic context, the
  input fills as much of it as it can: up to the training window, leaving at least one generated chunk. A 5-second input
  continued to 10 s uses about 4.5 s of it as context and adds about 5.5 s. The output starts with the context frames, decoded
  again, as the official output does; turn off `include_context` to get only the new frames.
- **Input frames.** The whole input video is decoded, so trim long videos first. Frames are taken at the model's frame rate,
  assuming a constant source frame rate. On the frame grid the node takes the exact frame. Upstream's floating-point times
  repeat some frames there (6 of 25 in its 5B examples); [TencentARC/Looking-Ahead#3](https://github.com/TencentARC/Looking-Ahead/pull/3)
  fixes that upstream.

## Speed and memory

RTX 5090, Windows 11, PyTorch 2.10 SDPA, 768×432:

| length | frames | sampling | peak VRAM (allocated) |
| --- | --- | --- | --- |
| 5 s | 80 | 210 s | 13.4 GiB |
| 10 s | 160 | 515 s | 13.6 GiB |

Most of the VRAM is the KV cache (22 latent frames for the positive and negative prompts, about 9.8 GiB). PyTorch reserved
up to 15.8 GiB, so a 24 GB GPU is comfortable and 16 GB is borderline. The node frees other models from VRAM before it samples,
and moves its own model back to RAM afterwards unless `keep_model_loaded` is on. In system RAM the model takes about 2.8 GB
(bfloat16; the 8 GB checkpoint is memory-mapped while loading). During a run, Windows commit on this machine went from about
30 GB idle to 55–60 GB, with ComfyUI holding the text encoder and VAE as well.

5B, same machine, 1024×576 at 24 fps:

| length | frames | sampling | peak VRAM (allocated) |
| --- | --- | --- | --- |
| 5 s | 120 | 233 s | 22.9 GiB |
| 10 s | 240 | 556 s | 23.2 GiB |

The KV cache (31 latent frames, both prompts) takes 12.3 GiB and the bfloat16 model 9.3 GiB. PyTorch reserved up to 27 GiB,
so at the trained size the 5B model needs a 32 GB GPU. In system RAM the model takes about 10 GB. The 34 GB checkpoint is
memory-mapped while loading: Windows commit on this machine peaked at 71–84 GB while loading, and was about 60 GB while sampling.

## Limits

- The two official modes are covered: text-and-image-to-video and video continuation.
- Batch size 1. CUDA GPU required.
- Attention runs on PyTorch SDPA. If FlashAttention 4 imports (Linux), it is used, as upstream does.

## License

Apache-2.0 (LICENSE). Code from Looking Ahead (Copyright (C) 2026 Tencent, Apache-2.0) and Wan2.1/Wan2.2 (Alibaba Wan Team,
Apache-2.0) is vendored in `looking_ahead_comfy/vendor`. Each file and change is listed in
[vendor/UPSTREAM.md](looking_ahead_comfy/vendor/UPSTREAM.md); see NOTICE. The model weights are not included; they keep their
own licenses (Looking Ahead, Wan 2.1 and Wan 2.2: Apache-2.0).

This is an unofficial integration. If you use the model, please cite the paper:

```bibtex
@misc{gao2026lookingahead,
  title         = {Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation},
  author        = {Gao, Ruopeng and Huang, Yukun and Hu, Jiarui and Wang, Shuai and Hu, Wenbo and Shan, Ying and Wang, Limin},
  year          = {2026},
  eprint        = {2610.29989},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.29989}
}
```
