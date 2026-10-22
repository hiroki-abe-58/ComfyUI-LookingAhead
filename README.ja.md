# Looking Ahead for ComfyUI

**Looking Ahead**（[論文](https://arxiv.org/abs/2610.29989)、[コード](https://github.com/TencentARC/Looking-Ahead)、
[重み](https://huggingface.co/TencentARC/Looking-Ahead)）の非公式 ComfyUI ノードです。Looking Ahead は Tencent ARC Lab・南京大学・浙江大学による
Future Chunk Prediction（FCP）の自己回帰動画モデルです。このパックは 1.3B のテキスト＋画像→動画モデル（Wan 2.1 ベース）を動かします。
開始画像とプロンプトを与えると、公式推論と同じく KV キャッシュを使って 3 潜在フレームずつ動画を生成します。

[English README](README.md)

![このノードで作った 5 秒のクリップ 4 本](docs/images/demo_grid.gif)

4 本とも、既定の設定（768×432、16 fps、50 ステップ）で seed 42 の 1 回目の出力を、縮小しただけでそのまま並べています。風車は回り、列車は橋を渡ります。
犬は浜辺を走りますが、くわえていたおもちゃが消えます。凧は失敗例で、上昇せずにフレームの外へ落ちていきます。開始画像はこのデモのために MOSS-Image（Apache-2.0）で
生成しました。プロンプトと詳細は [docs/images](docs/images/README.md) にあります。

## 確認したこと

- **サンプラー**：公式実行のプロンプト埋め込み・先頭フレームの潜在・ノイズをそのまま渡すと、出力潜在は公式の `CausalDiffusionInferencePipeline`
  とビット単位で一致しました（5 秒、768×432、seed 42、50 ステップ、CFG 3.0、RTX 5090）。CI でもプッシュのたびに CPU で同じ比較をしています。
  公式パイプラインに小さなランダムモデルを入れて比べ、学習窓に収まる場合と、アテンションシンク付きで KV キャッシュがロールする場合の 2 通りです。
- **開始画像**：公式と同じリサイズ＋中央クロップを通し、画素は完全一致します（テスト済み）。ComfyUI の Wan 2.1 VAE によるエンコード結果は、公式との差が
  0.4 %（相対 L2）です。
- **テキストエンコーダ**：ComfyUI の umT5-XXL（Wan）は公式とトークン数が一致します。同じ重みで比べると、float32 基準との差は公式の bfloat16 経路
  （2.1 %）より小さい 0.17 % です。よく使われる fp8 ファイルでは 6〜9 % です。
- **エンドツーエンド**：ComfyUI の実キューで、公式と同じ場面・構図・動きの動画を再現しました。フレーム PSNR は、公式のテキストエンコーダの重みで 28.7 dB、
  fp8 ファイルで 28.0 dB です。差の原因は、自己回帰サンプラーが入力の小さな差を増幅することです。先頭フレームの 0.4 % の差だけでも出力潜在は 3.5 % 変わり、
  全体を ComfyUI で処理したときの 4.1 % に近い値になります。
- **安定性**：同じワークフローを 2 回実行するとビット単位で同じ出力になります。中断すると 1 ステップ以内に止まり、次のプロンプトは通常どおり実行されます。

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

チェックポイントは `fcp_wan21_t2v_1.3b` フォルダごと置いてください。teacher forcing のベースライン（`tf_wan21_t2v_1.3b/model_ema.pt`、5.7 GB、
スクリプトでは `--tf`）も同じように読み込んで動かせます（動作確認済み）。FCP と通常の teacher forcing を比べるときに使ってください。
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

## 速度とメモリ

RTX 5090、Windows 11、PyTorch 2.10 の SDPA、768×432 での値です。

| 長さ | フレーム | サンプリング | ピーク VRAM（allocated） |
| --- | --- | --- | --- |
| 5 秒 | 80 | 210 秒 | 13.4 GiB |
| 10 秒 | 160 | 515 秒 | 13.6 GiB |

VRAM の大半は KV キャッシュです（ポジティブとネガティブそれぞれ 22 潜在フレーム分、約 10.5 GiB）。PyTorch の reserved は最大 15.8 GiB でした。
24 GB の GPU なら余裕があり、16 GB はぎりぎりです。ノードはサンプリング前にほかのモデルを VRAM から解放し、終了後は自分のモデルを RAM に戻します
（`keep_model_loaded` をオンにした場合を除く）。システム RAM 上のモデルは約 2.8 GB です（bfloat16。8 GB のチェックポイントは読み込み時にメモリマップします）。
この PC では、実行中の Windows のコミットチャージが、待機時の約 30 GB から 55〜60 GB になりました（ComfyUI がテキストエンコーダと VAE も保持した状態）。

## 制限

- 対応は 1.3B のテキスト＋画像→動画だけです。5B（Wan 2.2）と動画の続き生成にはまだ対応していません。
- バッチサイズは 1 で、CUDA GPU が必要です。
- アテンションは PyTorch SDPA で計算します。FlashAttention 4 を import できる環境（Linux）では、上流と同じくそちらを使います。

## ライセンス

Apache-2.0（LICENSE）です。Looking Ahead（Copyright (C) 2026 Tencent、Apache-2.0）と Wan2.1／Wan2.2（Alibaba Wan Team、Apache-2.0）のコードを
`looking_ahead_comfy/vendor` に同梱しています。各ファイルと変更点は [vendor/UPSTREAM.md](looking_ahead_comfy/vendor/UPSTREAM.md) に、帰属表示は NOTICE にあります。
モデルの重みは含まれておらず、それぞれのライセンスに従います（Looking Ahead と Wan 2.1 は Apache-2.0）。

非公式の統合です。モデルを使った成果では、論文を引用してください（引用は英語版 README にあります）。
