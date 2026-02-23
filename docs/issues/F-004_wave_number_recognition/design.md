# [F-004] Wave数判定 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-004 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-22 |
| ステータス | ✅ 完了 |

## 設計方針

FHD画像の固定位置 (ROI) を切り出し、HSVフィルタで白色テキストのみを抽出（二値化）した後、
ROIを「WAVEテキスト部」と「数字部」に分割し、それぞれpHashでテンプレートとのハミング距離を算出して判定する。

Wave表示の背景は透過でゲーム画面が映り込むため、**前処理（HSV白色抽出）で背景を除去してから認識する**ことが本設計の要。

認識フローは2段の判定:
1. 左90pxの「WAVE」テキスト有無を判定
2. 一致 → 右33pxの数字(1~5)を判定

**最終パラメータ**: ROI (76, 35, 199, 80) = 123x45px, pHash 16x16 (256bit), 閾値=116, 精度100% (104/104)

ShakeScouter-NWで実績のあるROI分割方式を踏襲し、数字認識をCNNからpHashに置き換える。

> Extra Wave の認識は F-005 に分離。ROI位置・サイズが通常Waveと異なるため独立管理とする。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| ROI分割 + pHash (OpenCV) | ✅ 採用 | 分割により各サブ画像が小さくなりノイズに強い。認識対象が5パターン固定でありpHashで十分 |
| ROI全体 + pHash 5パターン一括比較 | ❌ 不採用 | WAVE 1~5は数字1文字分(数px)しか違わず、200px全体のpHashでは区別困難な可能性がある |
| CNN (PyTorch) による数字認識 | ❌ 不採用 | プロジェクト方針でpHash採用。5パターンの判定にCNNはオーバーキル |
| MAE (平均絶対誤差) テンプレート比較 | ❌ 不採用 | ShakeScouter-NWで使用しているがpHashの方がノイズ耐性・速度面で有利 |
| 汎用OCR (Tesseract等) | ❌ 不採用 | ゲーム画面の独自フォント・透過背景に不向き（プロジェクト方針） |

## モジュール構成

### ステップ1（CLI版ミニアプリ）

```
experiments/exp_004_wave_number_recognition/
├── main.py                  # CLIエントリポイント
├── wave_recognizer.py       # WaveNumberRecognizer クラス（認識ロジック本体）
├── README.md                # 実験結果記録
└── results/                 # 判定結果テキストファイル出力先
```

### ステップ2（GUI統合版）

```
experiments/exp_003_gui_recognition_viewer/plugins/
└── wave_number.py           # WaveNumberPlugin (RecognitionPlugin Protocol準拠)
```

**変更対象（既存ファイル）:**

```
experiments/exp_003_gui_recognition_viewer/
└── main.py                  # _load_plugins() に WaveNumberPlugin を追加登録
```

### ステップ3（共通化リファクタリング）

```
shared/recognition/
├── __init__.py              # 公開APIの re-export
└── phash.py                 # compute_phash, hamming_distance
```

**変更対象（既存ファイル）:**

```
experiments/exp_004_wave_number_recognition/
└── wave_recognizer.py       # compute_phash, hamming_distance を shared/ からの import に置き換え
```

## クラス・関数設計

### WaveNumberRecognizer（認識ロジック本体）

```python
class WaveNumberRecognizer:
    """Wave数のpHash認識器（ROI分割方式）"""

    # ROI座標 (FHD 1920x1080 基準)
    WAVE_ROI = (76, 35, 199, 80)          # 全体: 123x45 px
    WAVE_TEXT_WIDTH = 90                   # 左側「WAVE」テキスト部の幅
    WAVE_DIGIT_OFFSET = 90                # 右側 数字部の開始位置

    # HSVフィルタ（白色テキスト抽出）
    HSV_LOWER = (0, 0, 210)
    HSV_UPPER = (180, 80, 255)

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        wave_text_hash: np.ndarray,
        digit_hashes: dict[int, np.ndarray],   # {1: hash, 2: hash, ..., 5: hash}
        threshold: int = 116,
    ):
        """
        Args:
            wave_text_hash: 「WAVE」テキスト部のテンプレートハッシュ
            digit_hashes: 数字1~5のテンプレートハッシュ辞書
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        ...

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームからWave数を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "WAVE_1" ~ "WAVE_5", "NONE"
        """
        ...

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出す (123x45 BGR)"""
        ...

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """HSVフィルタで白色テキストを抽出し、モルフォロジー演算でノイズ除去した二値画像を返す"""
        ...

    def _split_wave_text(self, binary: np.ndarray) -> np.ndarray:
        """二値画像から左90pxの「WAVE」テキスト部を取得"""
        ...

    def _split_digit(self, binary: np.ndarray) -> np.ndarray:
        """二値画像から右33pxの数字部を取得"""
        ...

    @staticmethod
    def compute_phash(image: np.ndarray) -> np.ndarray:
        """16x16 pHashを計算する (256bit, DCTベース独自実装)"""
        ...

    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        """2つのハッシュ間のハミング距離を算出する"""
        ...
```

### shared/recognition/（ステップ3: 共通化）

#### shared/recognition/__init__.py

公開APIを re-export する。利用側は `from shared.recognition import compute_phash, hamming_distance` で使用する。

```python
from shared.recognition.phash import compute_phash, hamming_distance

__all__ = ["compute_phash", "hamming_distance"]
```

#### shared/recognition/phash.py

`WaveNumberRecognizer` から以下の2関数をそのまま移動する。ロジックは一切変更しない。

```python
import cv2
import numpy as np


def compute_phash(image: np.ndarray) -> np.ndarray:
    """16x16 pHashを計算する (256bit, DCTベース)

    32x32にリサイズ → DCT → 低周波16x16係数 → 平均値で二値化 → 256bitハッシュ
    """
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(resized)
    dct = cv2.dct(float_img)
    dct_low = dct[:16, :16]
    dct_flat = dct_low.flatten()
    mean_val = np.mean(dct_flat[1:])
    hash_bits = (dct_flat > mean_val).astype(np.uint8)
    return np.packbits(hash_bits)


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """2つのハッシュ間のハミング距離を算出する"""
    return int(np.unpackbits(np.bitwise_xor(hash1, hash2)).sum())
```

#### shared/__init__.py

空ファイル（パッケージ認識用）。

### ステップ3: 既存コードの変更

#### wave_recognizer.py の変更

**import追加**（`sys.path.insert` パターンで プロジェクトルートを追加）:

```python
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.recognition import compute_phash, hamming_distance
```

パス解決: `wave_recognizer.py` → `exp_004_wave_number_recognition/` → `experiments/` → プロジェクトルート

**削除**: `compute_phash` と `hamming_distance` の `@staticmethod` 定義（2箇所）

**呼び出し箇所の変更**（8箇所）:

| 変更前 | 変更後 |
|---|---|
| `self.compute_phash(region)` | `compute_phash(region)` |
| `self.hamming_distance(h1, h2)` | `hamming_distance(h1, h2)` |

対象メソッド: `recognize()` 内4箇所、`recognize_debug()` 内4箇所

#### main.py (exp_004) の変更

**import追加**:

```python
from shared.recognition import compute_phash
```

`main.py` は `wave_recognizer.py` と同じディレクトリにあるため、`wave_recognizer.py` が `sys.path` にプロジェクトルートを追加済みであれば追加の `sys.path.insert` は不要。
ただし import 順序に依存するため、`main.py` にも同じ `sys.path.insert` を記述する方が安全。

**呼び出し箇所の変更**（2箇所）:

| 変更前 | 変更後 |
|---|---|
| `WaveNumberRecognizer.compute_phash(region)` | `compute_phash(region)` |

#### wave_number.py (exp_003 plugin) の変更

**変更なし**。`WaveNumberPlugin` は `WaveNumberRecognizer` を内部的に使用しており、`compute_phash` / `hamming_distance` を直接呼び出していないため影響を受けない。

### WaveNumberPlugin（ステップ2: GUI統合用）

#### import方式

`baito_text.py` と同じ `sys.path.insert` パターンで exp_004 の `WaveNumberRecognizer` をインポートする。

```python
import sys
from pathlib import Path

# exp_004 のディレクトリをパスに追加
_EXP_004_DIR = (
    Path(__file__).resolve().parent.parent.parent / "exp_004_wave_number_recognition"
)
sys.path.insert(0, str(_EXP_004_DIR))

from wave_recognizer import WaveNumberRecognizer  # noqa: E402
```

#### テンプレートパスの解決

プロジェクトルートを基準に `assets/templates/wave/` を辿る。

```python
# プロジェクトルート: plugins/ → exp_003_gui_recognition_viewer/ → experiments/ → shakeop/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "assets" / "templates" / "wave"
```

#### テンプレート読み込み

`__init__()` で全6ファイルを読み込み、1つでも欠けていれば `FileNotFoundError` を送出する。

```python
# 必須テンプレートファイル一覧
_REQUIRED_TEMPLATES = [
    "wave_text.npy",
    "digit_1.npy",
    "digit_2.npy",
    "digit_3.npy",
    "digit_4.npy",
    "digit_5.npy",
]
```

読み込み手順:
1. `template_dir` 引数が `None` なら `DEFAULT_TEMPLATE_DIR` を使用
2. `_REQUIRED_TEMPLATES` の各ファイルの存在を確認（欠損があれば `FileNotFoundError`）
3. `np.load()` で各 `.npy` ファイルを読み込み
4. `wave_text_hash`, `digit_hashes` (dict) を構築
5. `WaveNumberRecognizer` を初期化（`threshold=22`）

#### クラス設計

```python
class WaveNumberPlugin:
    """Wave数判定の認識プラグイン (RecognitionPlugin Protocol準拠)"""

    # ROI座標（WaveNumberRecognizer.WAVE_ROI と同値。描画用に保持）
    ROI = (76, 35, 199, 80)

    # オーバーレイ描画の色
    COLOR_DETECTED = (0, 255, 0)      # 緑: Wave検出時
    COLOR_NOT_DETECTED = (0, 0, 255)  # 赤: 未検出時

    # テキスト描画位置: ROI矩形の右横
    TEXT_OFFSET_X = 10   # ROI右端からの水平マージン (px)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2

    def __init__(
        self,
        template_dir: Path | None = None,
        threshold: int = 116,
    ) -> None:
        """
        Args:
            template_dir: テンプレートディレクトリ。Noneなら DEFAULT_TEMPLATE_DIR を使用。
            threshold: pHashハミング距離の閾値。ステップ1で最適化された値。
        """
        if template_dir is None:
            template_dir = DEFAULT_TEMPLATE_DIR

        # 全テンプレートの存在確認
        for filename in _REQUIRED_TEMPLATES:
            path = template_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"テンプレートが見つかりません: {path}")

        # テンプレート読み込み
        wave_text_hash = np.load(str(template_dir / "wave_text.npy"))
        digit_hashes = {
            i: np.load(str(template_dir / f"digit_{i}.npy"))
            for i in range(1, 6)
        }

        # 認識器を初期化
        self._recognizer = WaveNumberRecognizer(
            wave_text_hash=wave_text_hash,
            digit_hashes=digit_hashes,
            threshold=threshold,
        )

    @property
    def name(self) -> str:
        return "Wave数判定"

    def process(self, frame: np.ndarray) -> dict:
        """WaveNumberRecognizer.recognize() を呼び出し、結果をdictで返す。"""
        result, confidence = self._recognizer.recognize(frame)
        return {
            "detected": result != "NONE",
            "confidence": confidence,
            "wave_result": result,        # "WAVE_1"~"WAVE_5", "NONE"
        }

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """ROI矩形と判定結果テキストをオーバーレイ描画する。

        ROI矩形: 検出時=緑、未検出時=赤
        テキスト: ROI矩形の右横に表示（左上は小さな領域のため下部だと見づらい）
        """
        overlay = frame.copy()
        x1, y1, x2, y2 = self.ROI
        detected = result["detected"]
        confidence = result["confidence"]
        wave_result = result["wave_result"]

        color = self.COLOR_DETECTED if detected else self.COLOR_NOT_DETECTED

        # ROI矩形
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # テキストラベル: ROI右横に表示
        label = f"{wave_result} conf={confidence:.4f}"
        text_x = x2 + self.TEXT_OFFSET_X
        text_y = y1 + (y2 - y1) // 2 + 5  # ROI垂直中央に揃える
        cv2.putText(
            overlay, label, (text_x, text_y),
            self.TEXT_FONT, self.TEXT_SCALE, color, self.TEXT_THICKNESS,
        )
        return overlay

    def format_log(self, result: dict) -> str:
        """ログ1行分をフォーマットする。"""
        return f"wave={result['wave_result']:<7}  confidence={result['confidence']:.4f}"
```

#### _load_plugins() への登録

`exp_003_gui_recognition_viewer/main.py` の `_load_plugins()` に以下を追加:

```python
try:
    from plugins.wave_number import WaveNumberPlugin

    plugins.append(WaveNumberPlugin())
except Exception as e:
    print(f"Warning: WaveNumberPlugin の読み込みに失敗: {e}")
```

既存の `BaitoTextPlugin` と同じ try/except パターンで、テンプレート欠損時は警告を出して他のプラグインのみで起動する。

## データフロー

```
入力: BGR画像 (1920x1080)
  │
  ▼
1. ROI切り出し
   frame[35:80, 76:199] → ROI画像 (123x45 BGR)
  │
  ▼
2. HSV白色テキスト抽出（前処理）
   BGR → HSV変換
   InRange(lower=[0,0,210], upper=[180,80,255])
   → 二値画像 (123x45)    ← 背景除去済み
  │
  ▼
3. 「WAVE」テキスト判定
   binary[:, :90] → 左90px切り出し
   pHash計算 → wave_text_hash とハミング距離比較
  │
  ├── 距離 ≤ 閾値（WAVEテキスト検出）
  │     │
  │     ▼
  │   4. 数字判定
  │       binary[:, 90:] → 右33px切り出し
  │       pHash計算 → digit_hashes[1~5] と各ハミング距離比較
  │       最小距離のものを採用（閾値以下の場合）
  │     │
  │     ├── 最小距離 ≤ 閾値 → "WAVE_N" (N=1~5) + 信頼度
  │     └── 全て閾値超え → "NONE" + 信頼度
  │
  └── 距離 > 閾値（WAVEテキスト非検出）→ "NONE" + 信頼度
```

## テンプレート構成

| テンプレート | 元画像の領域 | サイズ | 説明 |
|---|---|---|---|
| `wave_text.npy` | 二値化ROI左90px | 90x45 → 16x16 pHash (32bytes) | 「WAVE」テキスト部のpHashハッシュ |
| `digit_1.npy` | 二値化ROI右33px (Wave 1表示時) | 33x45 → 16x16 pHash (32bytes) | 数字「1」のpHashハッシュ |
| `digit_2.npy` | 同上 (Wave 2表示時) | 33x45 → 16x16 pHash (32bytes) | 数字「2」のpHashハッシュ |
| `digit_3.npy` | 同上 (Wave 3表示時) | 33x45 → 16x16 pHash (32bytes) | 数字「3」のpHashハッシュ |
| `digit_4.npy` | 同上 (Wave 4表示時) | 33x45 → 16x16 pHash (32bytes) | 数字「4」のpHashハッシュ |
| `digit_5.npy` | 同上 (Wave 5表示時) | 33x45 → 16x16 pHash (32bytes) | 数字「5」のpHashハッシュ |

配置先: `assets/templates/wave/`

## テンプレート作成手順

1. 各Wave表示のスクリーンショット（1920x1080）を用意する
2. ミニアプリの `--create-template` モードを実行する
3. 内部処理:
   - ROI切り出し (76, 35, 199, 80)
   - HSVフィルタで白色テキスト抽出（二値化）
   - Wave種類に応じてサブROI切り出し
   - pHash計算 → `.npy` ファイルとして保存

```bash
# Wave 1~5 のテンプレート作成（WAVEテキスト + 数字）
uv run python experiments/exp_004_wave_number_recognition/main.py \
    --create-template --wave-number 1 --image <Wave1表示の画像>
```

※ `--wave-number 1` 実行時に `wave_text.npy` が未作成であれば同時に生成する。

## ミニアプリのCLIインターフェース

```bash
# テンプレート作成
uv run python experiments/exp_004_wave_number_recognition/main.py \
    --create-template --wave-number <1-5> --image <画像パス>

# ディレクトリ一括判定
uv run python experiments/exp_004_wave_number_recognition/main.py \
    --image-dir <画像ディレクトリ>

# ディレクトリ一括判定（デバッグ付き）
uv run python experiments/exp_004_wave_number_recognition/main.py \
    --image-dir <画像ディレクトリ> --debug

# 閾値を変えて検証
uv run python experiments/exp_004_wave_number_recognition/main.py \
    --image-dir <画像ディレクトリ> --threshold 8
```

### CLIオプション一覧

| オプション | 説明 | デフォルト |
|---|---|---|
| `--image-dir` | 判定対象の画像ディレクトリパス | (必須 ※判定モード時) |
| `--create-template` | テンプレート作成モード | — |
| `--image` | テンプレート作成時のソース画像パス | (必須 ※テンプレート作成時) |
| `--wave-number` | テンプレート作成時のWave番号 (1~5) | — |
| `--template-dir` | テンプレートの配置ディレクトリ | `assets/templates/wave/` |
| `--threshold` | ハミング距離の閾値 | 116 |
| `--debug` | ROI可視化・詳細情報表示 | off |

## 出力フォーマット

### 判定結果テキストファイル

ファイル名: `YYMMDD-hhmmss_wave_result.txt`
出力先: `experiments/exp_004_wave_number_recognition/results/`
形式: TSV（タブ区切り）

```
# F-004 Wave Recognition Result
# Date: 2026-02-22 15:30:45
# Image dir: /path/to/images
# Threshold: 10
filename	result	confidence
wave1_001.png	WAVE_1	0.9844
wave1_002.png	WAVE_1	0.9688
wave3_001.png	WAVE_3	0.9531
lobby_001.png	NONE	0.2031
```

### result列の値

| 値 | 意味 |
|---|---|
| `WAVE_1` ~ `WAVE_5` | 通常Wave（番号付き） |
| `NONE` | Wave表示なし（判定不能） |

## 検証項目

ミニアプリで以下を検証し、README.md に結果を記録する。

| # | 検証内容 | 判定基準 |
|---|---|---|
| 1 | HSV白色抽出の有効性 | 背景除去後、テキストのみが残ること（debugモードで目視確認） |
| 2 | 「WAVE」テキスト判定の精度 | Wave表示あり画面の正検出率 95%以上 |
| 3 | 数字(1~5)判定の精度 | 正答率 95%以上 |
| 4 | Wave表示なし画面の正棄却率 | 95%以上 |
| 6 | 最適な閾値の特定 | 各テンプレートとのハミング距離分布から決定 |
| 7 | 処理時間 | 1フレームあたり 1ms 以内 |

## 依存関係

- 前提: F-001（pHash認識パターンの確立）、F-003（GUI統合先、ステップ2のみ）
- 使用ライブラリ:
  - OpenCV (`cv2.dct`) — DCTベースpHash計算（16x16独自実装）
  - OpenCV (`cv2.cvtColor`, `cv2.inRange`) — HSV変換・白色抽出
  - NumPy — 配列操作

## テスト計画

- ユニットテスト: `tests/unit/test_wave_number_recognition.py`（統合時に作成）
- テストデータ: `data/test_fixtures/wave/`
  - Wave 1~5 表示画像: 各3枚以上
  - Wave表示なし画像: 5枚以上（ロビー・リザルト・休憩等）
  - 異なるステージ・潮位のバリエーションを含むこと

## 備考

- ROI座標は (76, 35, 199, 80) = 123x45px に確定（ShakeScouter-NW参考値から右端39px・左端38pxを削減）
- HSVフィルタは V≥210, S≤80 に確定（霧イベント対策でV下限を200→210に引き上げ）
- pHash 16x16 (256bit) のDCTベース独自実装を採用（`cv2.img_hash.PHash` は8x8固定のため不使用）
- 閾値は116に確定（有効範囲 111~122、幅12）
- 「WAVE」テキスト部90px / 数字部33px に確定（左端余白38px削減により判別マージン拡大）
