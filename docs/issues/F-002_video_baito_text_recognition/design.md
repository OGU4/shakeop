# [F-002] 「バイトの時間です」動画テキスト認識 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-002 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-22 |
| ステータス | ✅ 完了 |

## 設計方針

F-001で実装・検証済みの `BaitoTextRecognizer` を動画入力に適用する。
カメラデバイスから `cv2.VideoCapture` でフレームを取得し、毎フレーム `BaitoTextRecognizer.recognize()` を呼び出して認識を行う。

認識ロジックの再実装は行わず、exp_001の `BaitoTextRecognizer` クラスを `sys.path` 経由で import して使用する。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| exp_001の BaitoTextRecognizer を import + cv2.VideoCapture | ✅ 採用 | 認識精度・速度は F-001 で検証済み。動画ループの追加のみ |
| 認識ロジックを再実装 | ❌ 不採用 | F-001 のコードで十分。重複を避ける |
| 認識ロジックを shared/ に移動 | ❌ 不採用 | 時期尚早。統合時に移動する |

## モジュール構成

- `experiments/exp_002_video_baito_text_recognition/main.py` — ミニアプリ本体
- `experiments/exp_002_video_baito_text_recognition/README.md` — 実験結果記録
- `experiments/exp_001_baito_text_recognition/main.py` — BaitoTextRecognizer を import（依存）
- `experiments/exp_001_baito_text_recognition/template_hash8.npy` — テンプレートハッシュ（依存）

## exp_001 からの import 方法

```python
import sys
from pathlib import Path

# exp_001 のディレクトリをパスに追加
EXP_001_DIR = Path(__file__).resolve().parent.parent / "exp_001_baito_text_recognition"
sys.path.insert(0, str(EXP_001_DIR))

from main import BaitoTextRecognizer  # exp_001 の BaitoTextRecognizer を import
```

テンプレートハッシュも exp_001 のものをデフォルトで参照する：

```python
DEFAULT_TEMPLATE_PATH = EXP_001_DIR / "template_hash8.npy"
```

## 関数設計

```python
def parse_args() -> argparse.Namespace:
    """CLIオプションをパースする。
    --camera: カメラデバイス番号（デフォルト: 10）
    --template: テンプレートハッシュパス（デフォルト: exp_001の template_hash8.npy）
    --threshold: ハミング距離閾値（デフォルト: 10）
    --debug: OpenCVデバッグ表示ON
    """
    ...


def run_camera(args: argparse.Namespace) -> None:
    """カメラ入力でリアルタイム認識を行うメインループ。

    1. テンプレートハッシュを読み込み BaitoTextRecognizer を初期化
    2. cv2.VideoCapture でカメラを開く
    3. フレームごとに recognize() を呼び出し
    4. 結果をコンソール出力
    5. debug 時は OpenCV ウィンドウにも描画
    6. Ctrl+C または q キー（debug時）で終了
    """
    ...


def draw_debug(frame: np.ndarray, detected: bool, confidence: float) -> np.ndarray:
    """デバッグ用の可視化フレームを生成する。

    - フレーム全体を表示
    - ROI矩形を描画（検出時: 緑、未検出時: 赤）
    - 判定結果と信頼度をテキスト描画
    """
    ...
```

## データフロー

```
カメラデバイス (/dev/video10)
  │
  ▼
1. フレーム取得
   cv2.VideoCapture(device_id) → BGR画像 (1920x1080)
   ※ フレーム取得失敗時はループ継続（カメラ切断はループ終了）
  │
  ▼
2. 認識処理
   BaitoTextRecognizer.recognize(frame) → (detected: bool, confidence: float)
   ※ exp_001 の実装をそのまま呼び出し
  │
  ▼
3. コンソール出力
   print(f"detected={detected}  confidence={confidence:.4f}")
   ※ 毎フレーム改行で出力
  │
  ▼
4. (debug時) OpenCVウィンドウ描画
   draw_debug(frame, detected, confidence) → debug_frame
   cv2.imshow("exp_002: video baito text", debug_frame)
  │
  ▼
5. 終了判定
   ├── Ctrl+C (KeyboardInterrupt) → ループ終了
   └── (debug時) q キー (cv2.waitKey) → ループ終了
  │
  ▼
6. クリーンアップ
   cap.release()
   cv2.destroyAllWindows()  # debug時のみ
```

## デバッグウィンドウの描画仕様

```
┌─────────────────────────────────────────┐
│ フレーム全体 (1920x1080)                  │
│                                           │
│                                           │
│          ┌──────────────────┐             │
│          │ ROI (緑 or 赤)    │             │
│          └──────────────────┘             │
│          DETECTED conf=0.9844             │
│                                           │
│                                           │
└─────────────────────────────────────────┘
```

- ROI矩形の色: 検出時=緑 (0, 255, 0)、未検出時=赤 (0, 0, 255)
- テキスト: ROI矩形の下に `DETECTED conf=X.XXXX` または `NOT DETECTED conf=X.XXXX` を描画

## 終了処理

```python
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # ... 認識・出力 ...
        if args.debug:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if args.debug:
        cv2.destroyAllWindows()
```

- 通常モード: `Ctrl+C` で `KeyboardInterrupt` を捕捉してループ終了
- debugモード: `cv2.waitKey(1)` で `q` キーも受け付ける
- `finally` ブロックでカメラ解放とウィンドウ破棄を保証

## CLIインターフェース

```bash
# 基本実行（/dev/video10 から認識）
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10

# デバッグ表示付き
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 --debug

# テンプレート・閾値を指定
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 \
    --template /path/to/template_hash8.npy --threshold 10
```

## 検証項目

ミニアプリで以下を検証し、README.md に結果を記録する。

| # | 検証内容 | 判定基準 | 結果 |
|---|---|---|---|
| 1 | カメラデバイスからのフレーム取得 | /dev/video10 から 1920x1080 フレームが取得できる | - |
| 2 | 動画フレームでの認識精度 | 目視で明らかな誤判定がないこと | - |
| 3 | リアルタイム動作 | フレーム取得・認識・出力が途切れなく動作する | - |
| 4 | debugモードの動作 | ROI矩形・判定結果が正しく描画される | - |
| 5 | 終了処理 | Ctrl+C / qキー(debug時) で正常終了する | - |

## 依存関係

- 前提: F-001（BaitoTextRecognizer の実装 + テンプレートハッシュ）
- 使用ライブラリ:
  - OpenCV (`cv2.VideoCapture`, `cv2.img_hash.PHash`) — カメラ入力 + pHash計算
  - NumPy — テンプレートハッシュ読み込み

## テスト計画

- 手動テスト: サンプル動画を仮想カメラに流しながら目視確認
- 認識ロジック自体のユニットテストは F-001 のテストでカバー済み
- 動画ループ・フレーム取得は手動テストで確認

## 備考

- `--image` モードは持たない。静止画テストは exp_001 を使う
- 非debugモードでは `cv2.waitKey` を呼ばないため、OpenCVへの依存は認識処理のみ
- フレームスキップ・時系列安定化は将来要求。本設計では対応しない
