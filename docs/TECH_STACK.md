# docs/TECH_STACK.md — 技術スタック定義

> **本ファイルの役割**: プロジェクト全体の技術スタック情報を一元管理する。
> 設計書・実コードとの整合性は `docs/REVIEW_CRITERIA.md` の基準9に基づきレビューする。
>
> **最終更新**: 2026-03-07（ソース: pyproject.toml, CLAUDE.md, docs/issues/*, 実コードimport文）

---

## プロジェクト基盤

| 項目 | 値 | ソース |
|---|---|---|
| 言語 | Python | pyproject.toml |
| バージョン指定 | `>=3.12`（pyproject.toml で定義） | pyproject.toml, CLAUDE.md |
| パッケージ管理 | uv（pyproject.toml ベース、lockfile対応） | CLAUDE.md |
| 対象OS | Windows / Linux（クロスプラットフォーム） | CLAUDE.md |

---

## ライブラリ一覧

### 本番依存（pyproject.toml `[project].dependencies`）

| ライブラリ名 | バージョン指定 | 用途（1行） | 使用箇所（モジュール名） | 選定理由（1行） |
|---|---|---|---|---|
| opencv-contrib-python | `>=4.9` | 画像処理全般（ROI切り出し、色空間変換、DCT、リサイズ、ビデオキャプチャ） | `shared/recognition/phash.py`, `experiments/exp_001`〜`exp_007` 全ミニアプリ | OBS仮想カメラ対応、画像処理の業界標準 |
| numpy | `>=1.26` | 配列演算、pHashビット操作（packbits/unpackbits/bitwise_xor） | `shared/recognition/phash.py`, `experiments/exp_001`〜`exp_007` 全ミニアプリ | OpenCVのデータ形式(ndarray)と直結、高速な数値演算 |
| PySide6 | `>=6.6` | GUI（メインウィンドウ、QThread、Signal/Slot）、将来的に音声再生（QSoundEffect） | `experiments/exp_003_gui_recognition_viewer/`（main.py, main_window.py, capture_worker.py, recognition_worker.py） | LGPL、ネイティブ音声再生可能、Qt 6ベース |
| onnxruntime | `>=1.17` | ONNX推論（Scene分類、Object検出） | **未使用**（実コードにimportなし。将来の認識パイプラインに使用予定） | CPU推論前提、GPU非搭載PCでも動作可能 |
| tomli-w | `>=1.0` | TOML書き出し（設定ファイル出力用） | **未使用**（実コードにimportなし。将来の設定管理で使用予定） | pyproject.tomlと同じTOML形式で設定管理 |

### 開発依存（pyproject.toml `[dependency-groups].dev`）

| ライブラリ名 | バージョン指定 | 用途（1行） | 実行方法 |
|---|---|---|---|
| pytest | `>=8.0` | ユニットテスト・統合テスト | `uv run pytest tests/` |
| ruff | `>=0.3` | リンター + フォーマッター | `uv run ruff check` / `uv run ruff format` |
| mypy | `>=1.8` | 静的型チェック（`--strict` モード） | `uv run mypy --strict src/ shared/` |

### 標準ライブラリ（主要なもの）

| モジュール名 | 用途 | 使用箇所 |
|---|---|---|
| argparse | ミニアプリのCLI引数解析 | `experiments/exp_001`〜`exp_007` の `main.py` |
| pathlib | ファイルパス操作 | 全モジュール |
| typing | Protocol, runtime_checkable 等の型ヒント | `experiments/exp_003/recognition_plugin.py` |
| collections | deque（FPSカウンタのスライディングウィンドウ） | `experiments/exp_003/fps_counter.py` |
| time | 処理時間計測 | 各ミニアプリの `main.py` |
| datetime | ログ出力のタイムスタンプ | 各ミニアプリの `main.py`, `log_writer.py` |
| glob | プラグイン動的検出 | `experiments/exp_003/main_window.py` |

---

## バージョン固定ポリシー

| 項目 | 内容 |
|---|---|
| バージョン管理ファイル | `pyproject.toml`（唯一の定義元） |
| ロックファイル | `uv.lock`（uv が自動生成・管理） |
| バージョン指定方針 | `>=下限`（互換性のある最小バージョンを指定） |
| 固定(pin)方針 | `uv.lock` で実行時バージョンを固定。`pyproject.toml` では `==` 固定は使用していない |

---

## 制約・禁止事項

### 使用禁止ライブラリ・手法

| 禁止対象 | 理由 | ソース |
|---|---|---|
| 汎用OCR（Tesseract, PaddleOCR 等） | ゲーム画面の独自フォント・動く背景・透過・斜め配置に不向き。pHashで十分な精度が得られている | F-001, F-004 設計書 |
| CNN（固定パターン数が少ない認識対象） | Wave番号（5パターン）等はpHashで十分。CNNはオーバーキル | F-004 設計書 |
| MAE テンプレートマッチング（`cv2.matchTemplate`） | 明るさ・コントラスト変動に弱い。pHashの方がノイズ耐性・速度面で有利 | F-001, F-004 設計書 |
| `cv2.img_hash.PHash`（8x8, 64bit） | **非推奨（deprecated）**。R-001で `shared/recognition/` の16x16版(256bit)に統一済み | R-001 設計書 |

### 技術的制約

| 制約 | 内容 | ソース |
|---|---|---|
| 解像度 | FHD (1920x1080) キャプチャ入力を前提。すべてのROI座標はFHD基準のピクセル座標 | CLAUDE.md, 全要求仕様書 |
| pHashサイズ | 16x16 (256bit, DCTベース) を標準とする。新規実装で8x8は使用しない | R-001, F-004以降の設計書 |
| テンプレート保存形式 | NumPy `.npy` ファイル（uint8, shape=(32,)）。コード内へのハッシュ値インライン記述は禁止 | F-004〜F-007 設計書 |
| 推論環境 | CPU推論前提（GPU非搭載PCでも動作可能にする） | CLAUDE.md |
| 音声形式 | QSoundEffect は `.wav` のみ対応（mp3不可）。44.1kHz/16bit PCM推奨 | CLAUDE.md |

### 不整合事項

現時点で検出された不整合はなし。
