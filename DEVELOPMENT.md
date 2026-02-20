# shakeop — 開発方法論ドキュメント

> **このドキュメントはプロジェクト全体の開発ルールの最上位文書である。**
> すべての開発作業（ミニアプリ作成、メインアプリ開発、統合作業）はこのルールに従う。

---

## 目次

1. [開発の基本原則](#1-開発の基本原則)
2. [管理番号とドキュメント管理](#2-管理番号とドキュメント管理)
3. [モノレポ構成](#3-モノレポ構成)
4. [ミニアプリの作り方](#4-ミニアプリの作り方)
5. [共有インターフェース（契約）](#5-共有インターフェース契約)
6. [メインアプリへの統合手順](#6-メインアプリへの統合手順)
7. [ドキュメント管理ルール](#7-ドキュメント管理ルール)
8. [デバッグとテストの戦略](#8-デバッグとテストの戦略)
9. [Git運用ルール](#9-git運用ルール)
10. [具体例: シーン分類の開発フロー全体](#10-具体例-シーン分類の開発フロー全体)

---

## 1. 開発の基本原則

### 1.1 ミニアプリ駆動開発 (Mini-App Driven Development)

```
【原則】
  機能は必ず「ミニアプリ」として単独で動作確認してから、メインアプリに統合する。
  いきなりメインアプリに機能を書かない。
```

**なぜこうするか:**

- ゲーム画面認識は試行錯誤が多い。ミニアプリなら数秒で起動して即テストできる
- バグの原因が「認識ロジックの問題」か「統合の問題」かを切り分けられる
- 各ミニアプリが「この認識はこの精度で動く」という証拠（エビデンス）になる
- 後から別の手法に差し替えるときも、ミニアプリ単位で比較実験できる

### 1.2 開発フロー概要

```
┌──────────────────────────────────────────────────────────────┐
│                      開発フロー                               │
│                                                              │
│  Step 0: 管理番号を発行し、要求仕様書と機能設計書を書く       │
│  ┌────────────────────────────────────────┐                 │
│  │ docs/issues/F-001_scene_classify/      │                 │
│  │   requirements.md ← 何を・なぜ作るか   │                 │
│  │   design.md       ← どう作るか         │                 │
│  └────────────────────────────────────────┘                 │
│           │                                                  │
│           │ レビュー・承認                                    │
│           ▼                                                  │
│  Step 1: ミニアプリを作る                                    │
│  ┌────────────────────────────────────────┐                 │
│  │ experiments/exp_001_scene_classify/     │                 │
│  │   main.py  ← 単独で動く                │                 │
│  │   README.md ← 結果を記録               │                 │
│  └────────────────────────────────────────┘                 │
│           │                                                  │
│           │ 精度・速度が基準を満たしたら                      │
│           ▼                                                  │
│  Step 2: 共有インターフェースに合わせてコードを整形            │
│  ┌────────────────────────────────────────┐                 │
│  │ 入出力を Protocol/dataclass に合わせる  │                 │
│  │ ハードコードを設定値に置き換える         │                 │
│  └────────────────────────────────────────┘                 │
│           │                                                  │
│           ▼                                                  │
│  Step 3: メインアプリの該当モジュールに移植                   │
│  ┌────────────────────────────────────────┐                 │
│  │ src/salmon_buddy/recognition/          │                 │
│  │   scene_classifier.py                  │                 │
│  └────────────────────────────────────────┘                 │
│           │                                                  │
│           ▼                                                  │
│  Step 4: 統合テストで動作確認                                │
│  ┌────────────────────────────────────────┐                 │
│  │ tests/integration/                     │                 │
│  └────────────────────────────────────────┘                 │
│           │                                                  │
│           ▼                                                  │
│  Step 5: 管理番号のステータスを「完了」に更新                 │
│                                                              │
│  ※ ミニアプリは統合後も残す（回帰テスト・比較実験用）         │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 守るべき4つのルール

| # | ルール | 理由 |
|---|---|---|
| **R1** | **実装の前に必ず管理番号を発行し、要求仕様書と機能設計書を書く** | 何を・なぜ・どう作るかを明文化してから手を動かす |
| **R2** | ミニアプリは**30秒以内に起動して動作確認できる**こと | 試行錯誤のサイクルを速くする |
| **R3** | ミニアプリとメインアプリは**共通のデータ型（Protocol/dataclass）を使う** | 統合時のインターフェース不一致を防ぐ |
| **R4** | ミニアプリのREADMEに**実験結果（精度・速度・課題）を必ず記録する** | 何を試して何がダメだったかの知識を残す |

---

## 2. 管理番号とドキュメント管理

### 2.1 管理番号の体系

すべての開発作業（機能追加・バグ修正）に管理番号を付与する。

```
F-NNN   機能 (Feature)      例: F-001, F-002, F-010
B-NNN   バグ修正 (Bug)       例: B-001, B-002
```

### 2.2 ドキュメント管理ディレクトリ

```
docs/
└── issues/
    ├── index.md                       # 全管理番号の一覧表
    │
    ├── F-001_scene_classify/          # 機能: シーン分類
    │   ├── requirements.md            # 要求仕様書
    │   ├── design.md                  # 機能設計書
    │   └── notes/                     # 補足資料（スクショ・ログ等）
    │
    ├── F-002_digit_recognition/       # 機能: 数字認識
    │   ├── requirements.md
    │   ├── design.md
    │   └── notes/
    │
    ├── F-003_audio_notification/      # 機能: 音声通知
    │   ├── requirements.md
    │   ├── design.md
    │   └── notes/
    │
    ├── B-001_camera_linux_crash/      # バグ: Linux カメラクラッシュ
    │   ├── requirements.md            # バグの場合は「問題記述 + 修正要件」
    │   ├── design.md                  # 修正方針
    │   └── notes/
    │
    └── ...
```

ディレクトリ名は `{管理番号}_{短い説明}` とする。

### 2.3 開発作業の流れ

```
1. 管理番号を発行 → docs/issues/index.md に追記
       │
       ▼
2. 要求仕様書を書く → docs/issues/F-NNN_xxx/requirements.md
       │
       ▼
3. 機能設計書を書く → docs/issues/F-NNN_xxx/design.md
       │
       ▼
4. 実装（ミニアプリ作成 or メインアプリ修正）
       │
       ▼
5. テスト・検証
       │
       ▼
6. 統合（該当する場合）
       │
       ▼
7. 管理番号のステータスを「完了」に更新
```

**重要: Step 2, 3 を飛ばして Step 4 に進まない。**
ドキュメントは完璧でなくてよい。箇条書きレベルでもよいが、「何を作るか」「どう作るか」を書いてから手を動かす。

### 2.4 管理番号一覧 (docs/issues/index.md)

```markdown
# 管理番号一覧

## 機能 (Feature)

| 番号 | 名称 | 状態 | 実験 | 統合先 | 備考 |
|---|---|---|---|---|---|
| F-001 | シーン分類 | 📋 仕様作成中 | exp_002 | recognition/scene_classifier.py | |
| F-002 | 数字認識 | 📋 仕様作成中 | exp_003a | recognition/digit_recognizer.py | pHash方式 |
| F-003 | 固定テキスト識別 | 📋 未着手 | exp_010 | recognition/text_identifier.py | |
| F-004 | オオモノシャケ検出 | 📋 未着手 | exp_004 | recognition/object_detector.py | |
| F-005 | 音声通知 | 📋 未着手 | exp_006 | audio/notifier.py | |
| F-006 | GUI基本構成 | 📋 未着手 | exp_007 | gui/main_window.py | |
| F-007 | カメラキャプチャ | 📋 未着手 | exp_001 | capture/obs_camera.py | |
| F-008 | ゲーム状態管理(FSM) | 📋 未着手 | exp_009 | game_state/state_machine.py | |
| F-009 | 特徴量マッチング | 📋 未着手 | exp_005 | recognition/feature_matcher.py | |
| F-010 | 武器アイコン分類 | 📋 未着手 | exp_008 | - | 優先度低 |

## バグ修正 (Bug)

| 番号 | 名称 | 状態 | 原因 | 修正先 | 備考 |
|---|---|---|---|---|---|
| (まだなし) | | | | | |

### ステータス凡例
- 📋 未着手
- 📝 仕様作成中
- 🔬 実装中
- 🧪 テスト中
- ✅ 完了
- ❌ 取り下げ（理由をrequirements.mdに記載）
```

### 2.5 要求仕様書テンプレート (requirements.md)

```markdown
# [F-NNN] <機能名> — 要求仕様書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-NNN |
| 作成日 | YYYY-MM-DD |
| ステータス | 📋 未着手 / 📝 作成中 / ✅ 承認済 |
| 関連実験 | exp_NNN_xxx |

## 背景・目的

<なぜこの機能が必要か。どの課題を解決するか。>

## 要求事項

### 必須要求 (MUST)

- <この機能が満たすべき最低条件>
- <例: 数字0-9を95%以上の精度で認識できること>
- <例: 1フレームあたり5ms以内で処理完了すること>

### 推奨要求 (SHOULD)

- <あると望ましい条件>
- <例: 明るさ±30%の環境差でも精度が維持されること>

### 将来要求 (COULD)

- <今回は不要だが将来欲しい条件>

## 入出力

- **入力**: <何を受け取るか。例: BGR画像 (1920x1080)>
- **出力**: <何を返すか。例: SceneType enum + 信頼度>

## 受け入れ基準

<この機能が「完了」と言えるための具体的条件>

- [ ] 基準1: <例: test_fixturesの全画像に対して精度90%以上>
- [ ] 基準2: <例: 推論時間が目標以内>
- [ ] 基準3: <例: ユニットテストが全て通る>

## 備考

<制約・前提条件・注意事項など>
```

### 2.6 機能設計書テンプレート (design.md)

```markdown
# [F-NNN] <機能名> — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-NNN |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | YYYY-MM-DD |
| ステータス | 📋 未着手 / 📝 作成中 / ✅ 承認済 |

## 設計方針

<どのアプローチで実装するか。選択理由も書く。>

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| <手法A> | ✅ 採用 | <理由> |
| <手法B> | ❌ 不採用 | <理由> |

## モジュール構成

<作成・変更するファイルの一覧>

- `experiments/exp_NNN_xxx/main.py` — ミニアプリ
- `src/salmon_buddy/recognition/xxx.py` — 統合先

## クラス・関数設計

```python
class XxxRecognizer:
    def __init__(self, ...):
        ...

    def recognize(self, frame: np.ndarray) -> XxxResult:
        ...
```

## データフロー

<入力から出力までの処理の流れ>

## 依存関係

- 前提: <他の管理番号への依存。例: F-007（カメラキャプチャ）が必要>
- 使用ライブラリ: <例: OpenCV, ONNX Runtime>

## テスト計画

- ユニットテスト: `tests/unit/test_xxx.py`
- テストデータ: `data/test_fixtures/xxx/`

## 備考

<設計上の懸念点・将来の拡張性・既知の制約>
```

### 2.7 バグ修正の場合

バグ修正は requirements.md に「問題記述 + 修正要件」を書く。

```markdown
# [B-NNN] <バグ名> — 要求仕様書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | B-NNN |
| 作成日 | YYYY-MM-DD |
| 重要度 | 🔴 致命的 / 🟡 重要 / 🟢 軽微 |

## 問題の説明

<何が起きているか。再現手順。>

## 期待される動作

<本来どう動くべきか。>

## 修正要件

- <修正後に満たすべき条件>

## 受け入れ基準

- [ ] <バグが再現しなくなること>
- [ ] <既存テストが全て通ること>
```

### 2.8 管理番号とミニアプリの対応

管理番号とミニアプリ（実験）は 1:1 とは限らない。

```
F-002 数字認識
  ├── exp_003a_digit_phash     ← 手法A: pHash（第一候補）
  └── exp_003b_digit_cnn       ← 手法B: CNN（フォールバック）

F-001 シーン分類
  └── exp_002_scene_classify   ← 1:1対応
```

要求仕様書はF番号に1つ。ミニアプリは手法ごとに複数作ってよい。
機能設計書に「どのミニアプリで検証するか」を記載する。

---

## 3. モノレポ構成

すべてを1つのリポジトリで管理する（モノレポ）。
ミニアプリとメインアプリは同じリポジトリ内に共存する。

```
salmon-buddy/
│
├── CLAUDE.md                          # Claude Code用（最上位ガイド）
├── DEVELOPMENT.md                     # ← このドキュメント（開発方法論）
├── DESIGN.md                          # 設計ドキュメント
├── LICENSE                            # MIT
├── README.md                          # ユーザー向け
├── pyproject.toml                     # プロジェクト全体の依存関係
├── uv.lock
│
├── src/
│   └── salmon_buddy/                  # === メインアプリ ===
│       ├── __init__.py
│       ├── __main__.py
│       ├── main.py
│       ├── app.py
│       ├── capture/
│       ├── recognition/
│       ├── game_state/
│       ├── audio/
│       ├── gui/
│       └── config/
│
├── shared/                            # === 共有コード ===
│   └── salmon_types/                  # ミニアプリ・メインアプリ共通の型定義
│       ├── __init__.py
│       ├── protocols.py               # Protocol定義（インターフェース）
│       ├── models.py                  # dataclass定義（データ型）
│       └── constants.py               # 定数（Enum等）
│
├── experiments/                        # === ミニアプリ群 ===
│   ├── README.md                      # 実験一覧と状態サマリー
│   │
│   ├── exp_001_camera_capture/        # 実験001: カメラ取得
│   │   ├── README.md                  # 実験の目的・結果・知見
│   │   ├── main.py                    # 単独実行可能
│   │   └── notes/                     # スクショ・ログ等
│   │
│   ├── exp_002_scene_classify/        # 実験002: シーン分類
│   │   ├── README.md
│   │   ├── main.py
│   │   ├── train.py                   # 学習スクリプト
│   │   └── notes/
│   │
│   ├── exp_003_ocr_digits/            # 実験003: 数字認識（pHash）
│   │   ├── README.md
│   │   ├── main.py
│   │   └── notes/
│   │
│   ├── exp_004_boss_detection/        # 実験004: オオモノ検出
│   │   ├── README.md
│   │   ├── main.py
│   │   ├── train.py
│   │   └── notes/
│   │
│   ├── exp_005_feature_match/         # 実験005: 特徴量マッチング
│   │   ├── README.md
│   │   ├── main.py
│   │   └── notes/
│   │
│   ├── exp_006_audio_playback/        # 実験006: 音声再生
│   │   ├── README.md
│   │   └── main.py
│   │
│   └── exp_007_gui_prototype/         # 実験007: GUI試作
│       ├── README.md
│       └── main.py
│
├── models/                            # ONNXモデル
├── assets/                            # 音声・アイコン・テンプレート
│
├── data/                              # === 学習データ・テストデータ ===
│   ├── .gitkeep
│   ├── raw/                           # 生フレーム（gitignore）
│   ├── classified/                    # 分類済み（gitignore）
│   ├── annotated/                     # アノテーション済み（gitignore）
│   └── test_fixtures/                 # テスト用固定画像（gitで管理）
│       ├── scene_lobby.png
│       ├── scene_wave_active.png
│       ├── scene_wave_active_ht.png
│       ├── ocr_quota_18.png
│       └── ...
│
├── tests/                             # === テスト ===
│   ├── unit/                          # ユニットテスト
│   │   ├── test_scene_classifier.py
│   │   ├── test_digit_recognizer.py
│   │   ├── test_text_identifier.py
│   │   ├── test_recognition_smoother.py
│   │   ├── test_game_state.py
│   │   └── test_audio.py
│   ├── integration/                   # 統合テスト
│   │   ├── test_pipeline.py
│   │   └── test_pipeline_to_state.py
│   └── conftest.py                    # pytest共通fixture
│
├── tools/                             # ユーティリティスクリプト
│   ├── generate_voices.py
│   ├── collect_training_data.py
│   └── roi_calibrator.py
│
├── docs/
│   ├── issues/                            # === 管理番号別ドキュメント ===
│   │   ├── index.md                       # 全管理番号の一覧表
│   │   ├── F-001_scene_classify/
│   │   │   ├── requirements.md            # 要求仕様書
│   │   │   ├── design.md                  # 機能設計書
│   │   │   └── notes/
│   │   ├── F-002_digit_recognition/
│   │   │   ├── requirements.md
│   │   │   ├── design.md
│   │   │   └── notes/
│   │   └── ...
│   └── integration_log.md             # 統合作業の記録
```

### 3.1 pyproject.toml での共有コード管理

```toml
[project]
name = "salmon-buddy"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "PySide6>=6.6",
    "opencv-python>=4.9",
    "onnxruntime>=1.17",
    "numpy>=1.26",
    "tomli>=2.0; python_version < '3.12'",
    "tomli-w>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.3",
    "mypy>=1.8",
]
train = [
    "ultralytics>=8.1",
    "albumentations>=1.3",
]

# ミニアプリからも shared/ を import できるようにする
[tool.setuptools.packages.find]
where = ["src", "shared"]

# もしくは uv workspace を使う場合:
# [tool.uv.workspace]
# members = ["src/*", "shared/*", "experiments/*"]
```

### 3.2 importの仕組み

ミニアプリからもメインアプリからも、共有コードを同じように import できる:

```python
# experiments/exp_002_scene_classify/main.py からの import
from salmon_types.protocols import SceneClassifierProtocol
from salmon_types.models import RecognitionResult
from salmon_types.constants import SceneType

# src/salmon_buddy/recognition/scene_classifier.py からの import
from salmon_types.protocols import SceneClassifierProtocol
from salmon_types.models import RecognitionResult
from salmon_types.constants import SceneType
```

この統一により、ミニアプリで動作確認したコードを最小限の変更で統合できる。

---

## 4. ミニアプリの作り方

### 4.1 ミニアプリの命名規則

```
exp_{連番3桁}_{短い説明}/
```

例:
- `exp_001_camera_capture/`
- `exp_002_scene_classify/`
- `exp_003_ocr_digits/`
- `exp_010_scene_classify_v2/`（v2: 手法を変えてやり直し）

### 4.2 ミニアプリの必須構成

```
exp_NNN_description/
├── README.md          # 【必須】目的・手法・結果・知見
├── main.py            # 【必須】単独実行のエントリーポイント
├── requirements.txt   # 【任意】追加依存がある場合のみ
├── train.py           # 【任意】学習スクリプト
└── notes/             # 【任意】スクショ・ログ・メモ
    ├── screenshot_001.png
    └── benchmark.txt
```

### 4.3 main.py のテンプレート

すべてのミニアプリは以下のパターンに従う:

```python
#!/usr/bin/env python3
"""
exp_NNN: <実験タイトル>

目的: <1行で何を検証するか>
手法: <使う技術・ライブラリ>
実行: uv run python experiments/exp_NNN_description/main.py
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加（共有コードの import 用）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from salmon_types.constants import SceneType  # 共有コードを使う例
from salmon_types.models import RecognitionResult


def parse_args():
    parser = argparse.ArgumentParser(description="exp_NNN: <タイトル>")
    parser.add_argument("--camera", type=int, default=0, help="カメラデバイス番号")
    parser.add_argument("--image", type=str, default=None, help="静止画テスト用パス")
    parser.add_argument("--debug", action="store_true", help="デバッグ表示ON")
    return parser.parse_args()


def run_with_camera(args):
    """カメラ入力でリアルタイムテスト"""
    import cv2

    cap = cv2.VideoCapture(args.camera)
    print(f"Camera {args.camera} opened: {cap.isOpened()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== ここに認識処理を書く =====
        result = process_frame(frame)
        # ================================

        # デバッグ表示
        if args.debug:
            debug_frame = draw_debug(frame, result)
            cv2.imshow("exp_NNN debug", debug_frame)

        # コンソール出力
        print(f"\r{result}", end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # スクリーンショット保存
            save_path = Path(__file__).parent / "notes" / f"capture_{int(time.time())}.png"
            save_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), frame)
            print(f"\nSaved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


def run_with_image(args):
    """静止画でテスト（デバッグ・精度検証用）"""
    import cv2

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Cannot read {args.image}")
        return

    result = process_frame(frame)
    print(f"Result: {result}")

    if args.debug:
        debug_frame = draw_debug(frame, result)
        cv2.imshow("exp_NNN debug", debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_frame(frame):
    """メインの処理ロジック（ここを実装する）"""
    # TODO: 実装
    pass


def draw_debug(frame, result):
    """デバッグ用の可視化（バウンディングボックス描画等）"""
    debug = frame.copy()
    # TODO: 結果を描画
    return debug


if __name__ == "__main__":
    args = parse_args()
    if args.image:
        run_with_image(args)
    else:
        run_with_camera(args)
```

**ポイント:**
- `--camera` と `--image` の2モードを必ず用意する
  - camera: リアルタイムテスト
  - image: 固定画像での精度検証・デバッグ（再現性がある）
- `--debug` でOpenCVの可視化ウィンドウを出す
- `s` キーでスクリーンショット保存（学習データ収集も兼ねる）
- `q` キーで終了
- 共有の型（Protocol/dataclass）を import して使う

### 4.4 ミニアプリ README.md テンプレート

```markdown
# exp_NNN: <実験タイトル>

## ステータス

| 項目 | 値 |
|---|---|
| 状態 | 🔬 実験中 / ✅ 完了・統合済み / ❌ 不採用 / 🔄 v2で再実験 |
| 統合先 | `src/salmon_buddy/recognition/scene_classifier.py` (未統合の場合は「-」) |
| 作成日 | 2025-XX-XX |
| 最終更新 | 2025-XX-XX |

## 目的

<この実験で何を検証するか。1〜3行で。>

## 手法

<使う技術・アルゴリズム・ライブラリ。>

## 実行方法

```bash
# カメラ入力
uv run python experiments/exp_NNN_description/main.py --camera 0 --debug

# 静止画テスト
uv run python experiments/exp_NNN_description/main.py --image data/test_fixtures/scene_lobby.png --debug
```

## 結果

### 精度

| 対象 | 正解率 | テスト画像数 | 備考 |
|---|---|---|---|
| lobby | 98% | 50枚 | |
| wave_active | 95% | 100枚 | 干潮時にやや不安定 |
| ... | | | |

### 速度

| 処理 | 時間 | 環境 |
|---|---|---|
| 推論 (CPU) | 3.2ms | i7-12700, 32GB RAM |
| 前処理含む | 5.1ms | 同上 |

### 課題・知見

- <箇条書きで、試行錯誤の中で得た知見を書く>
- <うまくいかなかったことも書く（後の自分やチームメイトの助けになる）>
- <「こうすれば改善しそう」というアイデアも書く>

## 統合メモ

<メインアプリに統合する際の注意点。>
<例: ROI座標はハードコードしてあるので settings.py から読むように変更する>
<例: debug描画のコードは統合時に除去する>
```

---

## 5. 共有インターフェース（契約）

ミニアプリとメインアプリの間で「入出力の形」を先に決めておく。
これが **契約 (Contract)** であり、統合をスムーズにする鍵。

### 5.1 Protocol定義（インターフェース）

```python
# shared/salmon_types/protocols.py
"""
認識モジュールのインターフェース定義。
ミニアプリもメインアプリも、この Protocol に従って実装する。
"""

from typing import Protocol, runtime_checkable
import numpy as np

from .models import (
    SceneResult,
    BossDetectionResult,
    DigitData,
    FeatureMatchResult,
    RecognitionResult,
)


@runtime_checkable
class SceneClassifierProtocol(Protocol):
    """シーン分類器のインターフェース"""

    def classify(self, frame: np.ndarray) -> SceneResult:
        """
        1フレームを分類する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            SceneResult: シーンラベルと信頼度
        """
        ...

    def classify_stable(self, frame: np.ndarray) -> SceneResult:
        """
        時系列安定化付きの分類。

        内部で直近Nフレームの多数決を取る。
        """
        ...


@runtime_checkable
class ObjectDetectorProtocol(Protocol):
    """オブジェクト検出器のインターフェース"""

    def detect(self, frame: np.ndarray) -> BossDetectionResult:
        """
        フレームからオオモノシャケを検出する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            BossDetectionResult: 検出リスト
        """
        ...


@runtime_checkable
class DigitRecognizerProtocol(Protocol):
    """数字認識器のインターフェース"""

    def recognize(self, roi_image: np.ndarray) -> tuple[str, float]:
        """
        ROI画像から数字列を読み取る。

        Args:
            roi_image: BGR画像（ROI切り出し済み）
        Returns:
            (認識結果の文字列, 平均信頼度)
        """
        ...


@runtime_checkable
class TextIdentifierProtocol(Protocol):
    """固定テキスト識別器のインターフェース"""

    def identify(self, roi_image: np.ndarray) -> tuple[str | None, float]:
        """
        ROI画像からテキストIDを識別する。

        Args:
            roi_image: BGR画像（ROI切り出し済み）
        Returns:
            (テキストID or None, 信頼度)
        """
        ...


@runtime_checkable
class FeatureMatcherProtocol(Protocol):
    """特徴量マッチング器のインターフェース"""

    def match(self, frame: np.ndarray, scene: "SceneType") -> FeatureMatchResult:
        """
        フレーム内のUIアイコン等を特徴量マッチングで検出する。

        Args:
            frame: BGR画像 (1920x1080)
            scene: 現在のシーン（検索対象の絞り込みに使用）
        Returns:
            FeatureMatchResult: マッチ結果リスト
        """
        ...


@runtime_checkable
class RecognitionPipelineProtocol(Protocol):
    """認識パイプライン全体のインターフェース"""

    def process(self, frame: np.ndarray) -> RecognitionResult:
        """
        1フレームを全Stageで処理する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            RecognitionResult: 全Stage統合結果
        """
        ...
```

### 5.2 データモデル定義

```python
# shared/salmon_types/models.py
"""
全モジュール共通のデータ型。
ミニアプリもメインアプリもこの型を使うこと。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .constants import SceneType, BossType


# --- Stage 1: シーン分類 ---

@dataclass(frozen=True)
class SceneResult:
    """シーン分類の結果"""
    scene: SceneType
    confidence: float           # 0.0 - 1.0
    raw_scores: dict[SceneType, float] = field(default_factory=dict)  # 全クラスのスコア（デバッグ用）


# --- Stage 2: オオモノ検出 ---

@dataclass(frozen=True)
class BossDetection:
    """単一オオモノの検出結果"""
    boss_type: BossType
    confidence: float           # 0.0 - 1.0
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) FHD座標

@dataclass(frozen=True)
class BossDetectionResult:
    """オオモノ検出の全結果"""
    detections: list[BossDetection] = field(default_factory=list)
    inference_time_ms: float = 0.0


# --- Stage 3: 数字認識 ---

@dataclass
class DigitData:
    """数字認識の結果"""
    golden_egg_current: int | None = None
    golden_egg_quota: int | None = None
    timer_seconds: int | None = None         # 秒に変換済み
    timer_raw: str | None = None             # 生テキスト（デバッグ用）
    wave_number: int | None = None
    total_golden_eggs: int | None = None
    total_power_eggs: int | None = None
    hazard_level: str | None = None
    inference_time_ms: float = 0.0

    # 各フィールドの信頼度
    confidences: dict[str, float] = field(default_factory=dict)


# --- Stage 4: 特徴量マッチング ---

@dataclass(frozen=True)
class FeatureMatch:
    """単一の特徴量マッチ結果"""
    name: str                   # テンプレート名
    confidence: float           # 0.0 - 1.0
    match_count: int            # 良好マッチ数

@dataclass(frozen=True)
class FeatureMatchResult:
    """特徴量マッチングの全結果"""
    matches: list[FeatureMatch] = field(default_factory=list)
    inference_time_ms: float = 0.0


# --- パイプライン統合結果 ---

@dataclass
class RecognitionResult:
    """認識パイプラインの統合結果（全Stage）"""
    timestamp: float
    scene: SceneResult
    bosses: BossDetectionResult = field(default_factory=BossDetectionResult)
    ocr: DigitData = field(default_factory=DigitData)
    features: FeatureMatchResult = field(default_factory=FeatureMatchResult)
    total_inference_time_ms: float = 0.0
```

### 5.3 定数定義

```python
# shared/salmon_types/constants.py
"""
プロジェクト全体で使う定数・Enum。
"""

from enum import Enum


class SceneType(Enum):
    UNKNOWN = "unknown"
    TITLE = "title"
    LOBBY = "lobby"
    LOADING = "loading"
    WAVE_INTRO = "wave_intro"
    WAVE_ACTIVE = "wave_active"
    WAVE_ACTIVE_HT = "wave_active_ht"
    WAVE_ACTIVE_LT = "wave_active_lt"
    SPECIAL_RUSH = "special_rush"
    SPECIAL_GRILL = "special_grill"
    SPECIAL_MOTHERSHIP = "special_mothership"
    SPECIAL_FOG = "special_fog"
    SPECIAL_MUDMOUTH = "special_mudmouth"
    SPECIAL_COHOCK = "special_cohock"
    SPECIAL_GIANT = "special_giant"
    WAVE_RESULT = "wave_result"
    GAME_CLEAR = "game_clear"
    GAME_OVER = "game_over"


class BossType(Enum):
    STEELHEAD = "bakudan"
    FLYFISH = "katagata"
    STEEL_EEL = "hebi"
    SCRAPPER = "teppan"
    STINGER = "tower"
    MAWS = "mogura"
    DRIZZLER = "koumori"
    FISH_STICK = "hashira"
    FLIPPER_FLOPPER = "diver"
    SLAMMIN_LID = "nabebuta"
    BIG_SHOT = "teppou"
    COHOZUNA = "yokozuna"
    HORRORBOROS = "tatsu"
    MEGALODONTIA = "jaw"


# シーングループ（頻繁に使うのでここで定義）
WAVE_ACTIVE_SCENES = frozenset({
    SceneType.WAVE_ACTIVE,
    SceneType.WAVE_ACTIVE_HT,
    SceneType.WAVE_ACTIVE_LT,
    SceneType.SPECIAL_RUSH,
    SceneType.SPECIAL_GRILL,
    SceneType.SPECIAL_MOTHERSHIP,
    SceneType.SPECIAL_FOG,
    SceneType.SPECIAL_MUDMOUTH,
    SceneType.SPECIAL_COHOCK,
    SceneType.SPECIAL_GIANT,
})

RESULT_SCENES = frozenset({
    SceneType.WAVE_RESULT,
    SceneType.GAME_CLEAR,
    SceneType.GAME_OVER,
})


# FHD基準の解像度
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
```

### 5.4 なぜ Protocol を使うか

```python
# Protocol を使うことで、ミニアプリの実装をそのまま差し替えられる

# ミニアプリで実験した実装 A
class SceneClassifierONNX:
    def classify(self, frame): ...
    def classify_stable(self, frame): ...

# 別のミニアプリで実験した実装 B
class SceneClassifierTemplateBased:
    def classify(self, frame): ...
    def classify_stable(self, frame): ...

# メインアプリ側: どちらの実装でも動く
def create_pipeline(classifier: SceneClassifierProtocol):
    # classifier が Protocol を満たしていれば何でもOK
    result = classifier.classify_stable(frame)
```

型チェッカー（mypy）が Protocol 適合をチェックしてくれるので、
「統合したら動かない」が事前に検出できる。

---

## 6. メインアプリへの統合手順

### 6.1 統合チェックリスト

ミニアプリをメインアプリに統合するときは、以下を**毎回確認する**:

```markdown
## 統合チェックリスト: F-NNN / exp_NNN → src/salmon_buddy/xxx/yyy.py

- [ ] 要求仕様書 (docs/issues/F-NNN/requirements.md) が書かれている
- [ ] 機能設計書 (docs/issues/F-NNN/design.md) が書かれている
- [ ] ミニアプリのREADMEに結果が記録されている
- [ ] 要求仕様書の受け入れ基準を満たしている
  - シーン分類: 正解率 90%以上、推論 10ms以下
  - オオモノ検出: mAP 70%以上、推論 50ms以下
  - 数字認識(pHash): 正解率 95%以上
- [ ] Protocol に準拠している（mypy でチェック）
- [ ] ハードコードされたパス・閾値を config から読むように変更した
- [ ] デバッグ用コード（cv2.imshow, print等）を除去 or ログレベルで制御に変更
- [ ] ユニットテストを作成した（tests/unit/test_xxx.py）
- [ ] テスト用fixture画像を data/test_fixtures/ に追加した
- [ ] 統合テストで他モジュールとの結合を確認した
- [ ] ミニアプリのREADME「ステータス」を「✅ 完了・統合済み」に更新
- [ ] docs/integration_log.md に統合記録を追記
```

### 6.2 統合パターン

ミニアプリ → メインアプリの典型的なコード変更:

```python
# === ミニアプリ版 (experiments/exp_002/main.py) ===

MODEL_PATH = "models/scene_classifier.onnx"  # ハードコード
THRESHOLD = 0.7                               # ハードコード

class SceneClassifier:
    def __init__(self):
        self.session = ort.InferenceSession(MODEL_PATH)
        self.threshold = THRESHOLD

    def classify(self, frame):
        result = self._run_inference(frame)
        print(f"Scene: {result}")           # デバッグ print
        cv2.imshow("debug", frame)          # デバッグ表示
        return result


# === メインアプリ版 (src/salmon_buddy/recognition/scene_classifier.py) ===

import logging
from salmon_types.protocols import SceneClassifierProtocol
from salmon_types.models import SceneResult

logger = logging.getLogger(__name__)

class SceneClassifier:  # Protocol に準拠
    def __init__(self, model_path: str, threshold: float = 0.7):
        self.session = ort.InferenceSession(model_path)  # 設定から受け取る
        self.threshold = threshold                        # 設定から受け取る

    def classify(self, frame: np.ndarray) -> SceneResult:
        result = self._run_inference(frame)
        logger.debug("Scene: %s (%.2f)", result.scene, result.confidence)  # logging
        return result

    # ★ Protocol 準拠チェック
    assert isinstance(SceneClassifier(...), SceneClassifierProtocol)
```

変更点のまとめ:
1. ハードコード → コンストラクタ引数（設定から注入）
2. `print()` → `logging`
3. `cv2.imshow()` → 削除（GUIが担当）
4. Protocol に型が一致することを確認

---

## 7. ドキュメント管理ルール

### 7.1 ドキュメント一覧と役割

| ファイル | 対象読者 | 内容 |
|---|---|---|
| `CLAUDE.md` | Claude Code | プロジェクトの全体像、ビルド方法、コーディング規約 |
| `DEVELOPMENT.md` | 開発者（自分） | **このドキュメント**。開発方法論・ルール |
| `DESIGN.md` | 開発者（自分） | アーキテクチャ設計、認識パイプライン詳細 |
| `README.md` | ユーザー | インストール方法、使い方 |
| `docs/issues/index.md` | 開発者（自分） | **全管理番号の一覧表とステータス** |
| `docs/issues/F-NNN/requirements.md` | 開発者（自分） | 各機能の要求仕様書 |
| `docs/issues/F-NNN/design.md` | 開発者（自分） | 各機能の機能設計書 |
| `experiments/README.md` | 開発者（自分） | 全実験の一覧表と現在ステータス |
| `experiments/exp_NNN/README.md` | 開発者（自分） | 各実験の目的・結果・知見 |
| `docs/integration_log.md` | 開発者（自分） | 統合作業の履歴 |

### 7.2 experiments/README.md の形式

```markdown
# 実験一覧

| # | 名称 | 状態 | 統合先 | 概要 |
|---|---|---|---|---|
| 001 | camera_capture | ✅ 統合済 | capture/obs_camera.py | OBS仮想カメラ取得 |
| 002 | scene_classify | 🔬 実験中 | - | YOLOv8-clsでシーン分類 |
| 003 | digit_phash | 🔬 実験中 | - | pHashで数字認識 |
| 004 | boss_detection | 📋 未着手 | - | YOLOv8でオオモノ検出 |
| 005 | feature_match | 📋 未着手 | - | AKAZE特徴量マッチング |
| 006 | audio_playback | ✅ 統合済 | audio/notifier.py | QSoundEffect音声再生 |
| 007 | gui_prototype | 🔬 実験中 | - | PySide6 GUI試作 |
| 010 | scene_classify_v2 | 🔬 実験中 | - | ResNetベースに変更 |

### ステータス凡例
- 📋 未着手
- 🔬 実験中
- ✅ 完了・統合済み
- ❌ 不採用（理由はREADMEに記載）
- 🔄 別バージョンで再実験
```

### 7.3 docs/integration_log.md の形式

```markdown
# 統合ログ

## 2025-XX-XX: exp_001_camera_capture → capture/obs_camera.py

- 統合者: Atsushi
- 変更点:
  - デバイスインデックスをハードコードからconfig読み込みに変更
  - cv2.imshow を除去
  - QThread化（メインアプリのスレッドモデルに合わせる）
- テスト: tests/unit/test_capture.py 追加
- 課題: Linuxでデバイス名取得ができない → issue#XX で対応予定

## 2025-XX-XX: exp_002_scene_classify → recognition/scene_classifier.py

- 統合者: Atsushi
- 変更点:
  - ...
```

---

## 8. デバッグとテストの戦略

### 8.1 3段階テスト

```
Level 1: ミニアプリ内テスト（手動）
  └── main.py --image / --camera で目視確認
  └── 最も頻繁に行う。素早く回す。

Level 2: ユニットテスト（自動）
  └── pytest tests/unit/
  └── 固定画像 (test_fixtures) に対する期待値テスト
  └── 統合前に必ず書く。

Level 3: 統合テスト（自動）
  └── pytest tests/integration/
  └── パイプライン全体の通しテスト
  └── 統合後に書く。
```

### 8.2 テスト用固定画像 (test_fixtures)

ミニアプリで `s` キーを押して保存したスクリーンショットのうち、
テストに適したものを `data/test_fixtures/` にコピーして管理する。

```
data/test_fixtures/
├── scene/
│   ├── lobby_001.png
│   ├── lobby_002_dark.png          # 暗い環境
│   ├── wave_active_001.png
│   ├── wave_active_ht_001.png      # 満潮
│   ├── wave_active_lt_001.png      # 干潮
│   ├── special_rush_001.png
│   └── ...
├── ocr/
│   ├── quota_18_of_18.png          # ノルマ18/18
│   ├── quota_03_of_12.png
│   ├── timer_045.png               # 残り0:45
│   └── ...
├── boss/
│   ├── bakudan_001.png
│   ├── katagata_001.png
│   └── ...
└── metadata.json                    # 各画像の正解ラベル
```

`metadata.json` の例:

```json
{
  "scene/lobby_001.png": {
    "scene": "lobby",
    "note": "通常明るさ"
  },
  "scene/lobby_002_dark.png": {
    "scene": "lobby",
    "note": "暗い部屋でキャプチャ"
  },
  "ocr/quota_18_of_18.png": {
    "golden_egg_current": 18,
    "golden_egg_quota": 18,
    "note": "ノルマ達成状態"
  }
}
```

### 8.3 ユニットテストの書き方

```python
# tests/unit/test_scene_classifier.py

import pytest
import json
from pathlib import Path
from salmon_buddy.recognition.scene_classifier import SceneClassifier
from salmon_types.constants import SceneType

FIXTURES_DIR = Path("data/test_fixtures/scene")
METADATA_PATH = Path("data/test_fixtures/metadata.json")


@pytest.fixture
def classifier():
    return SceneClassifier(model_path="models/scene_classifier.onnx")


@pytest.fixture
def metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def test_classify_lobby(classifier):
    """ロビー画面を正しく判定できる"""
    import cv2
    frame = cv2.imread(str(FIXTURES_DIR / "lobby_001.png"))
    result = classifier.classify(frame)
    assert result.scene == SceneType.LOBBY
    assert result.confidence > 0.8


def test_classify_all_fixtures(classifier, metadata):
    """全fixture画像に対する精度テスト"""
    import cv2

    correct = 0
    total = 0

    for rel_path, info in metadata.items():
        if not rel_path.startswith("scene/"):
            continue

        frame = cv2.imread(str(Path("data/test_fixtures") / rel_path))
        if frame is None:
            continue

        result = classifier.classify(frame)
        expected = SceneType(info["scene"])

        if result.scene == expected:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Scene classification accuracy: {accuracy:.1%} ({correct}/{total})")
    assert accuracy >= 0.90, f"Accuracy {accuracy:.1%} is below 90% threshold"
```

### 8.4 デバッグ支援ツール

各ミニアプリに共通で使えるデバッグユーティリティ:

```python
# shared/salmon_types/debug_utils.py
"""
ミニアプリ・メインアプリ共通のデバッグユーティリティ。
"""

import cv2
import numpy as np
import time
from functools import wraps
from pathlib import Path


class FrameDebugger:
    """フレームにデバッグ情報をオーバーレイ描画する"""

    @staticmethod
    def draw_roi(frame: np.ndarray, roi, color=(0, 255, 0), label: str = ""):
        """ROI矩形を描画"""
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), color, 2)
        if label:
            cv2.putText(frame, label, (roi.x, roi.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    @staticmethod
    def draw_detection(frame: np.ndarray, bbox, label: str, confidence: float, color=(0, 0, 255)):
        """検出結果のバウンディングボックスを描画"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.0%}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def draw_info(frame: np.ndarray, texts: list[str], position=(10, 30)):
        """画面左上に情報テキストを表示"""
        x, y = position
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (x, y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def measure_time(func):
    """処理時間を計測するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[PERF] {func.__name__}: {elapsed_ms:.1f}ms")
        return result
    return wrapper


class FrameRecorder:
    """フレームを連番で保存する（学習データ収集用）"""

    def __init__(self, output_dir: str, prefix: str = "frame"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.count = 0

    def save(self, frame: np.ndarray) -> Path:
        path = self.output_dir / f"{self.prefix}_{self.count:06d}.png"
        cv2.imwrite(str(path), frame)
        self.count += 1
        return path
```

---

## 9. Git運用ルール

### 9.1 ブランチ戦略

```
main                              # 安定版（常に動作する状態）
  │
  ├── F-NNN/exp-NNN-description   # 機能の実験用ブランチ
  │   例: F-001/exp-002-scene-classify
  │   例: F-002/exp-003a-digit-phash
  │
  ├── F-NNN/integrate             # 機能の統合作業用ブランチ
  │   例: F-001/integrate
  │
  ├── B-NNN/fix                   # バグ修正用ブランチ
  │   例: B-001/fix
  │
  └── chore/description           # 管理番号に紐づかない作業
      例: chore/ci-setup
      例: chore/docs-update
```

### 9.2 コミットメッセージ規約

```
[F-NNN] 機能の要求仕様書・機能設計書・実装
  例: [F-001] 要求仕様書を作成
  例: [F-001] 機能設計書を作成
  例: [F-001] シーン分類モデルの学習データ追加

[B-NNN] バグ修正
  例: [B-001] Linux カメラクラッシュの修正

[exp:NNN] ミニアプリ関連の変更
  例: [exp:002] 信頼度閾値の調整 0.7→0.6

[integrate] 統合作業
  例: [integrate] F-001 exp_002 → recognition/scene_classifier.py

[app] メインアプリの変更（管理番号に紐づかない軽微な変更）
  例: [app] GUI: コンパクトモード追加

[shared] 共有コードの変更
  例: [shared] SceneType に SPECIAL_GIANT を追加

[test] テスト関連
  例: [test] scene_classifier のユニットテスト追加

[docs] ドキュメント
  例: [docs] DEVELOPMENT.md 管理番号ルールを追加

[ci] CI/CD
  例: [ci] GitHub Actions にlintジョブ追加
```

**ポイント**: 機能実装・バグ修正のコミットは必ず管理番号（F-NNN / B-NNN）を先頭に付ける。

### 9.3 .gitignore のポイント

```gitignore
# 学習データ（大量なのでgit管理しない）
data/raw/
data/classified/
data/annotated/

# テスト用固定画像は管理する
!data/test_fixtures/

# ONNXモデル（大きいので別途配布）
models/*.onnx

# ミニアプリの一時出力
experiments/*/notes/*.png
experiments/*/notes/*.log
!experiments/*/notes/.gitkeep

# Python
__pycache__/
*.pyc
.mypy_cache/
.ruff_cache/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

---

## 10. 具体例: シーン分類の開発フロー全体

最初から最後までの流れを具体的に示す。

### Step 0: 管理番号発行 + 要求仕様書 + 機能設計書

```bash
# 管理番号ディレクトリ作成
mkdir -p docs/issues/F-001_scene_classify/notes

# docs/issues/index.md に追記
# | F-001 | シーン分類 | 📝 仕様作成中 | exp_002 | recognition/scene_classifier.py | |
```

`docs/issues/F-001_scene_classify/requirements.md` を作成:

```markdown
# [F-001] シーン分類 — 要求仕様書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-001 |
| 作成日 | 2025-XX-XX |
| ステータス | ✅ 承認済 |
| 関連実験 | exp_002_scene_classify |

## 背景・目的

ゲーム画面が「ロビー」「Wave中」「リザルト」等のどのシーンかを判定する。
後段の認識処理（数字認識、オオモノ検出等）を適切に分岐させるゲートの役割。

## 要求事項

### 必須要求 (MUST)

- 18種類のシーンを分類できること
- test_fixturesの全画像に対して正解率 90% 以上
- 1フレームあたり推論 10ms 以下（CPU）
- 時系列安定化（チラつき防止）機能を持つこと

### 推奨要求 (SHOULD)

- 明るさ±30%の環境差でも精度維持
- 5フレーム以内でシーン遷移を検知

## 入出力

- **入力**: BGR画像 (1920x1080)
- **出力**: SceneType enum + 信頼度 (0.0-1.0)

## 受け入れ基準

- [ ] test_fixturesのシーン画像に対して精度 90% 以上
- [ ] CPU推論時間 10ms 以下
- [ ] ユニットテスト tests/unit/test_scene_classifier.py が全て通る
```

`docs/issues/F-001_scene_classify/design.md` を作成:

```markdown
# [F-001] シーン分類 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-001 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2025-XX-XX |
| ステータス | ✅ 承認済 |

## 設計方針

軽量な画像分類モデル (YOLOv8n-cls) を ONNX Runtime で CPU推論する。
多数決による時系列安定化を組み合わせる。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| YOLOv8n-cls + ONNX | ✅ 採用 | 軽量・高速・学習が容易 |
| ResNet18 | ❌ 不採用 | やや重い。まずYOLOv8で十分か検証 |
| テンプレートマッチング | ❌ 不採用 | 環境差に弱い |

## モジュール構成

- `experiments/exp_002_scene_classify/main.py` — ミニアプリ
- `src/salmon_buddy/recognition/scene_classifier.py` — 統合先

## テスト計画

- ユニットテスト: `tests/unit/test_scene_classifier.py`
- テストデータ: `data/test_fixtures/scene/`
```

```bash
# コミット
git add docs/issues/
git commit -m "[F-001] 要求仕様書・機能設計書を作成"
```

### Step 1: ミニアプリ作成

```bash
# ブランチ作成
git checkout -b F-001/exp-002-scene-classify

# ディレクトリ作成
mkdir -p experiments/exp_002_scene_classify/notes
```

`experiments/exp_002_scene_classify/main.py` を作成:

```python
#!/usr/bin/env python3
"""
exp_002: シーン分類

目的: ゲーム画面のシーンをYOLOv8-clsで分類できるか検証
手法: YOLOv8n-cls + ONNX Runtime
実行: uv run python experiments/exp_002_scene_classify/main.py --camera 0 --debug
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from salmon_types.constants import SceneType
from salmon_types.models import SceneResult
from salmon_types.debug_utils import FrameDebugger, measure_time


class SceneClassifierExperiment:
    """シーン分類の実験実装"""

    LABEL_MAP = [scene.value for scene in SceneType]  # 要調整

    def __init__(self, model_path: str, threshold: float = 0.7):
        self.session = ort.InferenceSession(model_path)
        self.threshold = threshold

    @measure_time
    def classify(self, frame: np.ndarray) -> SceneResult:
        # 前処理
        input_tensor = self._preprocess(frame)
        # 推論
        outputs = self.session.run(None, {"images": input_tensor})
        probs = self._softmax(outputs[0][0])
        # 結果
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        scene = SceneType(self.LABEL_MAP[top_idx])

        return SceneResult(
            scene=scene,
            confidence=confidence,
            raw_scores={SceneType(self.LABEL_MAP[i]): float(p) for i, p in enumerate(probs)},
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        # ImageNet正規化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        return normalized.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


def run_camera(args):
    # classifier = SceneClassifierExperiment(args.model)  # モデルがまだない場合はダミー
    cap = cv2.VideoCapture(args.camera)
    print(f"Camera {args.camera}: {cap.isOpened()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: classifier.classify(frame) を呼ぶ
        # 今はフレーム表示とキャプチャ保存だけ

        if args.debug:
            FrameDebugger.draw_info(frame, [
                f"Frame: {frame.shape[1]}x{frame.shape[0]}",
                "Press 's' to save, 'q' to quit",
            ])
            cv2.imshow("exp_002: scene classify", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_dir = Path(__file__).parent / "notes"
            save_dir.mkdir(exist_ok=True)
            path = save_dir / f"capture_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--model", type=str, default="models/scene_classifier.onnx")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.image:
        # 静止画テスト
        frame = cv2.imread(args.image)
        # result = SceneClassifierExperiment(args.model).classify(frame)
        # print(result)
    else:
        run_camera(args)
```

### Step 2: 実験・改善・結果記録

実験を繰り返し、README.md に結果を記録:

```bash
# 学習データ収集
uv run python experiments/exp_002_scene_classify/main.py --camera 0 --debug
# → s キーでスクショを保存

# モデル学習
uv run python experiments/exp_002_scene_classify/train.py

# 精度テスト
uv run python experiments/exp_002_scene_classify/main.py --image data/test_fixtures/scene/lobby_001.png --debug

# 結果をREADMEに記録
# experiments/exp_002_scene_classify/README.md を更新
```

### Step 3: 統合

```bash
# 統合ブランチ作成
git checkout -b F-001/integrate

# メインアプリ側にファイル作成
# src/salmon_buddy/recognition/scene_classifier.py
# - ハードコード → config引数
# - print → logging
# - cv2.imshow → 削除

# ユニットテスト作成
# tests/unit/test_scene_classifier.py

# 統合テスト作成
# tests/integration/test_pipeline.py

# チェックリスト確認
# docs/integration_log.md に記録

# experiments/exp_002_scene_classify/README.md のステータスを更新
# 🔬 実験中 → ✅ 完了・統合済み

# docs/issues/index.md のステータスを更新
# F-001: 📝 仕様作成中 → ✅ 完了
```

### Step 4: マージ

```bash
# テスト全通し
uv run pytest

# マージ
git checkout main
git merge F-001/integrate
```

---

## 付録A: 想定されるミニアプリ一覧

開発順序の目安付き。

| 優先度 | # | 名称 | 内容 | 依存 |
|---|---|---|---|---|
| ★★★ | 001 | camera_capture | OBS仮想カメラ取得＋表示 | なし |
| ★★★ | 002 | scene_classify | シーン分類（YOLOv8-cls） | 001 |
| ★★★ | 003a | digit_phash | pHashで数字認識 | 001 |
| ★★☆ | 003b | digit_cnn | 軽量CNNで数字認識（pHashで不足なら） | 001 |
| ★★★ | 010 | text_phash | pHashで固定テキスト認識 | 001 |
| ★★☆ | 004 | boss_detection | オオモノ検出（YOLOv8） | 001 |
| ★★☆ | 005 | feature_match | 特徴量マッチング | 001 |
| ★★★ | 006 | audio_playback | QSoundEffect音声再生 | なし |
| ★★☆ | 007 | gui_prototype | PySide6 GUI基本構成 | なし |
| ★☆☆ | 008 | weapon_classify | 武器アイコン分類（優先度低） | 001 |
| ★☆☆ | 009 | fsm_prototype | FSM状態管理単体 | 002, 003a |
| ★☆☆ | 0XX+ | (手法変更) | v2, v3 の再実験 | 各種 |

**「001 → 006 → 007 を最初に作れば、映像取得・音声再生・GUI表示の骨格が揃う」**
→ その上に 002, 003, 004, 005 を順次追加していく流れ。

---

## 付録B: CLAUDE.md に追記すべき内容

```markdown
## 開発方法論

このプロジェクトは「ミニアプリ駆動開発」で進める。
詳細は DEVELOPMENT.md を参照。

### 重要ルール
1. 新機能は必ず experiments/ にミニアプリとして作る（いきなり src/ に書かない）
2. ミニアプリは shared/salmon_types/ の共通型を使う
3. 精度・速度が基準を満たしたら統合チェックリストに沿って src/ に移植する
4. ミニアプリは統合後も削除しない（回帰テスト用に残す）

### ミニアプリの作り方
uv run python experiments/exp_NNN_description/main.py --camera 0 --debug

### テスト
uv run pytest tests/unit/       # ユニットテスト
uv run pytest tests/integration/ # 統合テスト

### ドキュメント更新
- 実験結果 → experiments/exp_NNN/README.md
- 統合記録 → docs/integration_log.md
- 実験一覧 → experiments/README.md
```
