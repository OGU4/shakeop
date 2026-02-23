# [F-005] Extra Wave判定 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-005 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-23 |
| ステータス | ✅ 実装完了 |

## 設計方針

FHD画像の固定位置 (ROI) を切り出し、HSVフィルタで白色テキストのみを抽出（二値化）した後、
pHashでテンプレートとのハミング距離を算出して「EXTRA WAVE」の有無を判定する。

F-004（通常Wave）はWAVEテキスト部と数字部の2段判定だったが、
EXTRA WAVEは単一テキストのためROI全体の **1段判定** で済む。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| ROI全体 + pHash 1段判定 | ✅ 採用 | EXTRA WAVEは1パターンのみ。ROI分割不要でシンプル |
| ROI分割 + pHash 2段判定 | ❌ 不採用 | 数字がないため分割の意味がない |

## モジュール構成

```
experiments/exp_005_extra_wave_recognition/
├── main.py                        # CLIエントリポイント
├── extra_wave_recognizer.py       # ExtraWaveRecognizer クラス
└── README.md                      # 実験結果記録
```

**GUI統合:**

```
experiments/exp_003_gui_recognition_viewer/plugins/
└── extra_wave.py                  # ExtraWavePlugin (RecognitionPlugin Protocol準拠)
```

**変更対象（既存ファイル）:**

```
experiments/exp_003_gui_recognition_viewer/
└── main.py                        # _load_plugins() に ExtraWavePlugin を追加登録
```

## クラス・関数設計

### ExtraWaveRecognizer（認識ロジック本体）

```python
class ExtraWaveRecognizer:
    """Extra WaveのpHash認識器（1段判定）"""

    # ROI座標 (FHD 1920x1080 基準)
    EXTRA_WAVE_ROI = (38, 35, 238, 80)  # 200x45 px

    # HSVフィルタ（白色テキスト抽出）— F-004と同一パラメータ
    HSV_LOWER = np.array([0, 0, 210])
    HSV_UPPER = np.array([180, 80, 255])

    # モルフォロジー演算カーネル（ノイズ除去用）
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        extra_wave_hash: np.ndarray,
        threshold: int = 110,
    ):
        """
        Args:
            extra_wave_hash: 「EXTRA WAVE」テンプレートのpHashハッシュ
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        ...

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームからExtra Waveの有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "EXTRA", "NONE"
        """
        ...

    def recognize_debug(self, frame: np.ndarray) -> dict:
        """
        デバッグ用の詳細情報付き認識。

        Returns:
            {
                "result": str,
                "confidence": float,
                "distance": int,
                "roi": np.ndarray,
                "binary": np.ndarray,
            }
        """
        ...

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出す (200x45 BGR)"""
        ...

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """HSVフィルタで白色テキストを抽出し、モルフォロジー演算でノイズ除去した二値画像を返す"""
        ...
```

pHash計算には `shared/recognition/` の共通関数を使用する:

```python
from shared.recognition import compute_phash, hamming_distance
```

### ExtraWavePlugin（GUI統合用）

```python
class ExtraWavePlugin:
    """Extra Wave判定の認識プラグイン (RecognitionPlugin Protocol準拠)"""

    ROI = (38, 35, 238, 80)

    COLOR_DETECTED = (0, 255, 0)      # 緑
    COLOR_NOT_DETECTED = (0, 0, 255)  # 赤

    def __init__(self, template_dir: Path | None = None, threshold: int = 116) -> None:
        ...

    @property
    def name(self) -> str:
        return "Extra Wave判定"

    def process(self, frame: np.ndarray) -> dict:
        """
        Returns:
            {
                "detected": bool,
                "confidence": float,
                "extra_result": str,       # "EXTRA" or "NONE"
                "distance": int,
                "threshold": int,
            }
        """
        ...

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """ROI矩形と判定結果テキストをオーバーレイ描画する。"""
        ...

    def format_log(self, result: dict) -> str:
        ...
```

## データフロー

```
入力: BGR画像 (1920x1080)
  │
  ▼
1. ROI切り出し
   frame[35:80, 38:238] → ROI画像 (200x45 BGR)
  │
  ▼
2. HSV白色テキスト抽出（前処理）
   BGR → HSV変換
   InRange(lower=[0,0,210], upper=[180,80,255])
   モルフォロジーオープニング (3x3)
   → 二値画像 (200x45)
  │
  ▼
3. pHash判定
   二値画像全体の pHash 計算
   extra_wave_hash とハミング距離比較
  │
  ├── 距離 ≤ 閾値 → "EXTRA" + 信頼度
  └── 距離 > 閾値 → "NONE" + 信頼度
```

## テンプレート

| ファイル | 元画像の領域 | サイズ | 説明 |
|---|---|---|---|
| `extra_wave.npy` | 二値化ROI全体 (200x45) | 16x16 pHash (32bytes) | 「EXTRA WAVE」全体のpHashハッシュ |

配置先: `assets/templates/wave/extra_wave.npy`（保存済み）

## CLIインターフェース

```bash
# テンプレート作成
uv run python experiments/exp_005_extra_wave_recognition/main.py \
    --create-template --image <EXTRA WAVE表示の画像>

# ディレクトリ一括判定
uv run python experiments/exp_005_extra_wave_recognition/main.py \
    --image-dir data/test_fixtures/wave/

# 閾値を変えて検証
uv run python experiments/exp_005_extra_wave_recognition/main.py \
    --image-dir data/test_fixtures/wave/ --threshold 100

# デバッグモード
uv run python experiments/exp_005_extra_wave_recognition/main.py \
    --image-dir data/test_fixtures/wave/ --debug
```

### CLIオプション一覧

| オプション | 説明 | デフォルト |
|---|---|---|
| `--image-dir` | 判定対象の画像ディレクトリパス | (必須 ※判定モード時) |
| `--create-template` | テンプレート作成モード | — |
| `--image` | テンプレート作成時のソース画像パス | (必須 ※テンプレート作成時) |
| `--template-dir` | テンプレートの配置ディレクトリ | `assets/templates/wave/` |
| `--threshold` | ハミング距離の閾値 | 110 |
| `--debug` | ROI可視化・詳細情報表示 | off |

## 検証項目

| # | 検証内容 | 判定基準 |
|---|---|---|
| 1 | EXTRA WAVE の正検出率 | 95%以上 |
| 2 | 非 EXTRA WAVE 画面の正棄却率 | 95%以上 |
| 3 | 最適な閾値の特定 | ハミング距離分布から決定 |
| 4 | 処理時間 | 1フレームあたり 1ms 以内 |

## 依存関係

- 前提: F-004（pHash共通化 `shared/recognition/`）
- 使用モジュール:
  - `shared/recognition/` — `compute_phash`, `hamming_distance`
  - OpenCV (`cv2.cvtColor`, `cv2.inRange`) — HSV変換・白色抽出
  - NumPy — 配列操作

## 備考

- 閾値のデフォルト値は実データから最適化し **110** に決定（有効範囲: 75〜111）
- テンプレートは `assets/templates/wave/extra_wave.npy` に保存済み
- 精度: 150/150 (100%), 平均処理時間: 0.19ms
