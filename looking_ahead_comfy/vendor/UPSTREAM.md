# Vendored code

`scripts/vendor_patch.py` writes every `.py` file in this folder except `compat.py` (which is this pack's own code). It
copies the files from the two pinned sources below and applies the changes listed here. Each change is an exact text
replacement that must match exactly once. Each file starts with a line naming its source.

| source | commit | license |
| --- | --- | --- |
| https://github.com/TencentARC/Looking-Ahead | `210794c758f386d3a1eaf3673b51b7cdd8374dfc` | Apache-2.0, Copyright (C) 2026 Tencent (`LICENSE.Looking-Ahead.txt`, NOTICE in `NOTICE.Looking-Ahead.txt`) |
| https://github.com/Wan-Video/Wan2.2, `wan/modules/model.py` (sha256 `8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432`) | `1ea34ff48f87168174e12956e200b1d908b1c5ff` | Apache-2.0, Copyright 2024-2025 The Alibaba Wan Team Authors (`LICENSE.Wan2.2.txt`) |

`tests/test_vendor.py` checks the files against the hashes below. With the upstream sources present (CI), it also regenerates them with
`vendor_patch.py --check` and compares the syntax trees of `wan22/modules/model.py` with Looking Ahead's own copy.

## Files and changes

| file | from | changes |
| --- | --- | --- |
| `utils/causal_model.py` | Looking-Ahead `src/utils/causal_model.py` | Imports only (below), and the body of `CausalWanModel._forward_train` |
| `utils/causal_rope.py` | Looking-Ahead `src/utils/causal_rope.py` | none |
| `utils/attention.py` | Looking-Ahead `src/utils/attention.py` | FA4 probe and import guard |
| `wan/modules/attention.py` | Looking-Ahead `src/wan/modules/attention.py` | FA4 probe and import guard |
| `wan/modules/model.py` | Looking-Ahead `src/wan/modules/model.py` | cross-attention calls `attention()` |
| `wan/utils/fm_solvers_unipc.py` | Looking-Ahead `src/wan/utils/fm_solvers_unipc.py` | diffusers names from `compat.py` |
| `dataset/image_preprocessing.py` | Looking-Ahead `src/dataset/image_preprocessing.py` | none |
| `wan22/modules/model.py` | Wan2.2 `wan/modules/model.py` | five definitions kept, two one-line changes |

### Why each change

- **Imports in `causal_model.py`.** `src.*` absolute imports become relative imports inside this package, so the pack cannot collide with another
  node's `src` package. The diffusers `ModelMixin` / `ConfigMixin` / `register_to_config` come from `compat.py` (the model only
  needs `config` to hold its constructor arguments), so the pack does not depend on diffusers. The module-level
  `torch.compile(flex_attention)` is removed.
- **Training-only code in `causal_model.py`.** The causal-mask mixin and its visualization (which imports matplotlib) are removed, along with `is_main_process` (torch.distributed).
  These are only used by teacher-forcing training. `_forward_train` (that training forward; it also reads the `WAN_VIS_*` environment
  variables and writes mask images) is replaced by `raise NotImplementedError`. Inference calls `_forward_inference`, which is
  unchanged.
- **FA4 probe and import guard.** The FA4 availability probe comes from `compat.py`, which falls back to "not available"
  when transformers lacks `is_flash_attn_4_available`. A failing `import flash_attn.cute` marks FA4 unavailable, so
  attention falls back to PyTorch SDPA; on Windows, FA4's quack dependency imports fcntl. Upstream PR:
  https://github.com/TencentARC/Looking-Ahead/pull/2.
- **Cross-attention in `wan/modules/model.py`.** It calls `attention()` instead of the FA4-only `flash_attention()`. Its caller passes
  `context_lens=None`, so the SDPA fallback computes the same attention; with FA4 the call is unchanged. This is the same upstream PR.
- **`wan22/modules/model.py`.** Looking Ahead's `src/wan22/modules/model.py` keeps `sinusoidal_embedding_1d`, `rope_params`, `rope_apply`,
  `WanRMSNorm` and `WanLayerNorm` through a Wan2.2-TI2V-5B-Turbo lineage, on a maintainer-provided permission described in
  its NOTICE. This pack takes the same five definitions from the official Wan2.2 repository instead and changes two lines:
  - `WanLayerNorm.forward` passes `x` to LayerNorm without `.float()`, which works with bfloat16 weights;
  - `rope_apply` returns `.type_as(x)` instead of `.float()`.

  With these two changes the definitions have the same syntax trees as Looking Ahead's, except for an unused
  `WanRMSNorm.reset_parameters`.

## Hashes (UTF-8, LF line endings)

The test recomputes these.

| file | sha256 |
| --- | --- |
| `dataset/image_preprocessing.py` | `b6b26e92c765b230b3b971926465e59da2eb06a9ffae0326aa338f8649886811` |
| `utils/attention.py` | `439f69d8e8baedcec53a6bb4e3a9559fa7710cee6cf2297459adc204bac47e1b` |
| `utils/causal_model.py` | `85722ce9f331a679dd95bf35d3654b28fde553c75e83380556d27878f36f272a` |
| `utils/causal_rope.py` | `ca85360a2b103569ba1184505c88c25a2d8fd8677c95250023c36366c2ccff46` |
| `wan/modules/attention.py` | `310034b90f186c117bef80ba3670f6e5df51ad6ea10d0c5655091cc8fec2fa62` |
| `wan/modules/model.py` | `8b66d4a15ab09ffbf629a911e8f55df8fc12fb98745843250d956640bfab2fef` |
| `wan/utils/fm_solvers_unipc.py` | `b5f48cc464ec69628a686e3347508cabad88130d7f5d4ff81513ce0c10722ca8` |
| `wan22/modules/model.py` | `f2f306d48409d956c20ada25fc42cde0f3877b6337d07275ed99dc2e9b9ca5c0` |
