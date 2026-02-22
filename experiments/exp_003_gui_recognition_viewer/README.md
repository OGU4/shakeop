# exp_003: GUI認識ビューワー

## 概要

F-002（動画テキスト認識）をPySide6 GUIで操作できるミニアプリ。
カメラ入力・認識プラグイン切替・ログ出力をGUI上で制御する。

## 関連ドキュメント

- 要求仕様書: `docs/issues/F-003_gui_recognition_viewer/requirements.md`
- 機能設計書: `docs/issues/F-003_gui_recognition_viewer/design.md`

## 実行方法

```bash
uv run python experiments/exp_003_gui_recognition_viewer/main.py
```

### 仮想カメラで動画をテストする場合

```bash
# 別ターミナルで仮想カメラを起動
ffmpeg -re -stream_loop -1 \
  -i /path/to/sample.mp4 \
  -vf "scale=1920:1080" \
  -pix_fmt yuv420p \
  -f v4l2 /dev/video10

# ミニアプリ起動後、GUIでカメラ /dev/video10 を選択して開始
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `main.py` | エントリーポイント |
| `main_window.py` | メインウィンドウ（UI + Signal/Slot接続） |
| `capture_worker.py` | カメラ取得ワーカー（QThread） |
| `recognition_worker.py` | 認識処理ワーカー（QThread） |
| `recognition_plugin.py` | 認識プラグインProtocol定義 |
| `fps_counter.py` | FPS計算ユーティリティ |
| `log_writer.py` | テキストログ出力 |
| `plugins/baito_text.py` | 「バイトの時間です」認識プラグイン |

## 実験結果

| 項目 | 結果 |
|---|---|
| 実施日 | |
| テスト環境 | |
| カメラデバイス | |
| 認識FPS | |
| 検出精度 | |
| 備考 | |
