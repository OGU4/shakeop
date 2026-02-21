# [F-002] 「バイトの時間です」動画テキスト認識 — 要求仕様書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-002 |
| 作成日 | 2026-02-22 |
| ステータス | ✅ 完了 |
| 関連実験 | exp_002_video_baito_text_recognition |
| 前提機能 | F-001（静止画版「バイトの時間です」テキスト認識） |

## 背景・目的

F-001では静止画に対する「バイトの時間です」テキスト認識をpHashで実現し、精度100%・処理時間0.09msの結果を得た。

本機能には2つの目的がある：

1. **動作検証**: F-001の認識ロジックが動画フレームに対しても安定して動作するかを検証する
2. **パイプライン雛形**: 将来のメインアプリ統合に向けた、リアルタイム認識パイプラインの基礎構造を作る

## 要求事項

### 必須要求 (MUST)

- カメラデバイス（/dev/video10）からFHD (1920x1080) の動画フレームを取得できること
- F-001の `BaitoTextRecognizer` を `experiments/exp_001_baito_text_recognition/main.py` から import して使用すること（認識ロジックの再実装は行わない）
- 取得した全フレームに対して認識処理を行うこと（フレームスキップなし）
- 各フレームの判定結果（True/False）と信頼度（0.0-1.0）をコンソールに毎フレーム改行で出力すること
- `--camera` オプションでカメラデバイス番号を指定できること（デフォルト: 10）
- Ctrl+C でプログラムを終了できること
- `--debug` オプションでOpenCVウィンドウによるデバッグ表示を有効化できること
  - ROI矩形の描画
  - 判定結果（True/False）と信頼度のリアルタイム表示

### 推奨要求 (SHOULD)

- F-001のテンプレートハッシュ（template_hash8.npy）をそのまま流用できること
- テンプレートハッシュのパスを `--template` オプションで指定できること

### 将来要求 (COULD)

- フレームスキップによる処理FPS制御
- 時系列安定化（Nフレーム多数決によるチラつき防止）
- 認識結果の変化検知（False→True 等）時の通知
- 他の固定テキスト（Wave開始テキスト等）への拡張
- メインアプリ統合時の TextIdentifierProtocol 準拠

## 入出力

- **入力**: カメラデバイス（デフォルト: /dev/video10）からのBGR動画ストリーム (1920x1080)
- **出力**: コンソールに毎フレーム改行で `判定結果 (True/False)` と `信頼度 (float: 0.0-1.0)` を出力

## カメラデバイスの準備

入力には仮想カメラデバイス `/dev/video10` を使用する。
以下のコマンドでサンプル動画をループ再生して仮想カメラに流す：

```bash
ffmpeg -re -stream_loop -1 \
  -i /home/ogu4/Videos/sample/20251130_111506.mp4 \
  -vf "scale=1920:1080:flags=lanczos,colorspace=all=bt709:iall=bt709:fast=1,scale=in_range=limited:out_range=full,format=yuv420p" \
  -pix_fmt yuv420p \
  -f v4l2 /dev/video10 \
  -f v4l2 /dev/video11
```

## ROI定義

F-001と同一のROIを使用する。

| 項目 | 値 |
|---|---|
| 座標系 | FHD (1920x1080) 基準 |
| 形式 | (x1, y1, x2, y2) |
| 値 | (750, 545, 1170, 600) |
| 対象テキスト | 「バイトの時間です」 |

## CLIインターフェース

```bash
# 基本実行（/dev/video10 から認識）
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10

# デバッグ表示付き
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 --debug

# テンプレート指定
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 \
    --template experiments/exp_001_baito_text_recognition/template_hash8.npy
```

## コンソール出力例

```
detected=False  confidence=0.5156
detected=False  confidence=0.5625
detected=True   confidence=0.9844
detected=True   confidence=1.0000
detected=True   confidence=0.9844
detected=False  confidence=0.5938
```

## 受け入れ基準

- [ ] カメラデバイス（/dev/video10）から動画フレームを取得しリアルタイム認識が動作すること
- [ ] サンプル動画を流しながら目視確認し、明らかな誤判定がないこと
- [ ] ミニアプリが `--camera` モードで単独動作すること
- [ ] `--debug` モードでOpenCVウィンドウにフレーム全体 + ROI矩形 + 判定結果・信頼度がリアルタイム表示されること
- [ ] `--debug` モード時は `q` キーでも終了できること
- [ ] Ctrl+C で正常に終了すること
- [ ] 実験結果が experiments/exp_002_video_baito_text_recognition/README.md に記録されていること

## テストデータ

- テンプレートハッシュ: F-001で作成済みの `experiments/exp_001_baito_text_recognition/template_hash8.npy` を流用
- 動画入力: `/home/ogu4/Videos/sample/20251130_111506.mp4` を仮想カメラ経由で使用

## 備考

- 認識ロジックはF-001の `BaitoTextRecognizer` を import して使用する。再実装はしない
- `--image` モードは持たない。静止画テストは exp_001 を使う
- 処理フレームレートの制御は初期実装では行わない（全フレーム処理）。必要に応じて将来追加
- pHashの認識処理自体は0.09msと非常に高速なため、ボトルネックはフレーム取得やデバッグ描画側になると想定される
