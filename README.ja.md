# Looking Ahead for ComfyUI

**Looking Ahead**（[論文](https://arxiv.org/abs/2610.29989)、[コード](https://github.com/TencentARC/Looking-Ahead)、
[重み](https://huggingface.co/TencentARC/Looking-Ahead)）の非公式 ComfyUI ノードです。Looking Ahead は Tencent ARC Lab・南京大学・浙江大学による
Future Chunk Prediction（FCP）の自己回帰動画モデルです。このパックは 1.3B のテキスト＋画像→動画モデル（Wan 2.1 ベース）を動かします。
開始画像とプロンプトを与えると、公式推論と同じく KV キャッシュを使って 3 潜在フレームずつ動画を生成します。

[English README](README.md)

<!-- DEMO -->

## 確認したこと

- **サンプラー**：公式実行のプロンプト埋め込み・先頭フレームの潜在・ノイズをそのまま渡すと、出力潜在は公式の `CausalDiffusionInferencePipeline`
  とビット単位で一致しました（5 秒、768×432、seed 42、50 ステップ、CFG 3.0、RTX 5090）。CI でもプッシュのたびに CPU で同じ比較をしています。
  公式パイプラインに小さなランダムモデルを入れて比べ、学習窓に収まる場合と、アテンションシンク付きで KV キャッシュがロールする場合の 2 通りです。
- **開始画像**：公式と同じリサイズ＋中央クロップを通し、画素は完全一致します（テスト済み）。ComfyUI の Wan 2.1 VAE によるエンコード結果は、公式との差が
  0.4 %（相対 L2）です。
- **テキストエンコーダ**：ComfyUI の umT5-XXL（Wan）は公式とトークン数が一致します。同じ重みで比べると、float32 基準との差は公式の bfloat16 経路
  （2.1 %）より小さい 0.17 % です。よく使われる fp8 ファイルでは 6〜9 % です。
- **エンドツーエンド**：ComfyUI の実キューで公式の動画を再現しました。動きと構図は一致し、フレーム PSNR は平均 28.7 dB です。

詳細と数値は [docs/VERIFICATION.md](docs/VERIFICATION.md) にあります。

## ノード

| ノード | 内容 |
| --- | --- |
| **Looking Ahead Model Loader** | `models/looking_ahead` の `model_ema.pt`（FCP または TF ベースライン）を bfloat16 で読み込み、学習専用ブランチを除きます。RAM に常駐するのは 1 モデルだけです。 |
| **Looking Ahead Image to Video** | 開始画像、ポジティブ／ネガティブの conditioning（umT5-XXL）、Wan 2.1 VAE から、`VIDEO`（16 fps）、`IMAGE` フレーム、`LATENT`、JSON レポートを出力します。シード、秒数（1〜30）、ステップ、CFG、shift、サイズを指定できます。生成前にほかのモデルの VRAM を解放し、進捗を表示し、中断にも対応します。 |
| **Looking Ahead Unload** | 常駐モデルを解放します。 |

テキストエンコーダと VAE には ComfyUI 標準の `CLIPLoader`（type `wan`）と `VAELoader` を使います。Wan 2.1 用のファイルがあれば、そのまま使えます。

## インストール

ComfyUI 0.42.1 以降（V3 ノード API）が必要です。追加の Python パッケージは不要です。

```
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-LookingAhead
```

## モデル

| ファイル | フォルダ | サイズ | 入手先 |
| --- | --- | --- | --- |
| `fcp_wan21_t2v_1.3b/model_ema.pt` | `models/looking_ahead/` | 8.0 GB | [TencentARC/Looking-Ahead](https://huggingface.co/TencentARC/Looking-Ahead/tree/main/fcp_wan21_t2v_1.3b) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors`（または `umt5_xxl_fp16`） | `models/text_encoders/` | 6.7 GB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders) |
| `wan_2.1_vae.safetensors` | `models/vae/` | 0.25 GB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae) |

チェックポイントは `fcp_wan21_t2v_1.3b` フォルダごと置いてください。TF ベースライン（`tf_wan21_t2v_1.3b/model_ema.pt`）も同じように使えます。
Looking-Ahead リポジトリ全体はダウンロードしないでください。このパックがまだ対応していない 5B のチェックポイント（約 50 GB）も含まれています。

これらのファイルだけを固定リビジョンで取得し、SHA-256 を確認するスクリプトもあります（ComfyUI の Python で実行してください）。

```
python custom_nodes/ComfyUI-LookingAhead/tools/download_model.py --models-dir models --wan
```

## 使い方

`workflows/gui/looking_ahead_image_to_video.json` を読み込み、開始画像を選び、プロンプトに起きてほしい動きを書きます。ネガティブプロンプトは公式の既定値です。
既定の設定は公開された設定（768×432、16 fps、5 秒、50 ステップ、CFG 3.0、shift 5.0）です。

- **開始画像**：公式推論と同じく、フレームを覆うようにリサイズしてから中央をクロップします。画像の端は切れることがあります。
- **長さ**：モデルは 5 秒のクリップで学習されています。5 秒より長い出力は、公式コードと同じく、3 フレームのアテンションシンク付きでロールする KV キャッシュを使います。
  5 秒を超える品質は上流でも保証されていません。
- **サイズ**：ほかのサイズでも動きますが、学習時の解像度は 768×432 です。

<!-- PERF -->

## 制限

- 対応は 1.3B のテキスト＋画像→動画だけです。5B（Wan 2.2）と動画の続き生成にはまだ対応していません。
- バッチサイズは 1 で、CUDA GPU が必要です。
- アテンションは PyTorch SDPA で計算します。FlashAttention 4 を import できる環境（Linux）では、上流と同じくそちらを使います。

## ライセンス

Apache-2.0（LICENSE）です。Looking Ahead（Copyright (C) 2026 Tencent、Apache-2.0）と Wan2.1／Wan2.2（Alibaba Wan Team、Apache-2.0）のコードを
`looking_ahead_comfy/vendor` に同梱しています。各ファイルと変更点は [vendor/UPSTREAM.md](looking_ahead_comfy/vendor/UPSTREAM.md) に、帰属表示は NOTICE にあります。
モデルの重みは含まれておらず、それぞれのライセンスに従います（Looking Ahead と Wan 2.1 は Apache-2.0）。

非公式の統合です。モデルを使った成果では、論文を引用してください（引用は英語版 README にあります）。
