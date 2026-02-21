# exp_002: 「バイトの時間です」動画テキスト認識

## ステータス

| 項目 | 値 |
|---|---|
| 状態 | ✅ 完了 |
| 管理番号 | F-002 |
| 統合先 | - (未統合) |
| 作成日 | 2026-02-22 |
| 最終更新 | 2026-02-22 |

## 目的

F-001（静止画pHash認識）を動画入力に拡張し、リアルタイムで「バイトの時間です」テキストの有無を判定する。
動画フレームでの認識安定性の検証と、将来のメインアプリ統合に向けたパイプライン雛形の構築が目的。

## 手法

- exp_001 の `BaitoTextRecognizer` を import して使用（認識ロジックの再実装なし）
- `cv2.VideoCapture` でカメラデバイスからフレームを取得
- 毎フレーム `recognize()` を呼び出し、結果をコンソール出力

## 実行方法

### 前提: 仮想カメラの準備

```bash
ffmpeg -re -stream_loop -1 \
  -i /home/ogu4/Videos/sample/20251130_111506.mp4 \
  -vf "scale=1920:1080:flags=lanczos,colorspace=all=bt709:iall=bt709:fast=1,scale=in_range=limited:out_range=full,format=yuv420p" \
  -pix_fmt yuv420p \
  -f v4l2 /dev/video10 \
  -f v4l2 /dev/video11
```

### 実行

```bash
# 基本実行
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10

# デバッグ表示付き
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 --debug
```

## 結果

### 精度

| 検証内容 | 結果 | 備考 |
|---|---|---|
| 動画フレームでの認識精度 | - | 目視確認予定 |

### 速度

| 処理 | 時間 | 環境 |
|---|---|---|
| 認識処理 (CPU) | - | - |

### 課題・知見

- (実験後に記録)

## 統合メモ

- カメラキャプチャ部分は統合時に capture/ モジュールに分離する
- debug描画のコードは統合時に除去
- 認識ロジック自体は F-001 の統合に含まれる
