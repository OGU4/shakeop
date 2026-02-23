# [R-001] pHash共通化リファクタリング — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | R-001 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-23 |
| ステータス | ✅ 設計確定 |

## 設計方針

exp_001 `BaitoTextRecognizer` のpHash実装を `cv2.img_hash.PHash`（8x8, 64bit）から
`shared/recognition`（16x16, 256bit DCT）に差し替える。

変更は最小限に留め、認識フロー（ROI切り出し→グレースケール変換→pHash判定）は維持する。
ハッシュサイズ変更に伴うテンプレート再生成と閾値再チューニングを行う。

### 変更しないもの

- 前処理パイプライン（グレースケール変換のまま、HSVフィルタへの変更なし）
- `recognize()` メソッドの戻り値型 `tuple[bool, float]`
- ROI座標 `(750, 545, 1170, 600)` = 420x55px
- exp_001の全体的なファイル構成

## 変更対象ファイル一覧

| # | ファイル | 変更種別 | 概要 |
|---|---|---|---|
| 1 | `experiments/exp_001_baito_text_recognition/main.py` | 改修 | pHash実装をshared/recognitionに置換 |
| 2 | `experiments/exp_002_video_baito_text_recognition/main.py` | 改修 | テンプレートパスとデフォルト閾値を更新 |
| 3 | `experiments/exp_003_gui_recognition_viewer/plugins/baito_text.py` | 改修 | テンプレートパスを更新 |
| 4 | `assets/templates/text/baito.npy` | 新規 | 256bitテンプレートハッシュ |
| 5 | `experiments/exp_001_baito_text_recognition/template_hash8.npy` | 削除 | 旧8x8テンプレート |
| 6 | `experiments/exp_001_baito_text_recognition/template_hash16.npy` | 削除 | 旧テンプレート（中身は8x8） |

## 詳細設計

### 1. exp_001/main.py — BaitoTextRecognizer の改修

#### import の変更

```python
# 削除
import cv2  # cv2.img_hash の依存のみ削除。cv2.cvtColor, cv2.imread 等は残す

# 追加
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.recognition import compute_phash, hamming_distance  # noqa: E402
```

> パス解決: `main.py` → `exp_001_baito_text_recognition/` → `experiments/` → プロジェクトルート

#### BaitoTextRecognizer クラスの変更

```python
class BaitoTextRecognizer:
    """「バイトの時間です」テキストのpHash認識器"""

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        template_hash: np.ndarray,
        roi: tuple[int, int, int, int] = DEFAULT_ROI,
        threshold: int,           # ← デフォルト値は閾値チューニング後に確定
    ):
        self.template_hash = template_hash
        self.roi = roi
        self.threshold = threshold
        # 削除: self._hasher = cv2.img_hash.PHash.create()
        # 削除: self.hash_size = hash_size

    def recognize(self, frame: np.ndarray) -> tuple[bool, float]:
        roi_gray = self._extract_roi(frame)
        input_hash = compute_phash(roi_gray)                    # ← shared使用
        distance = hamming_distance(input_hash, self.template_hash)  # ← shared使用
        confidence = 1.0 - (distance / self.MAX_DISTANCE)       # ← 256固定
        detected = distance <= self.threshold
        return detected, confidence

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出してグレースケール変換する（変更なし）"""
        x1, y1, x2, y2 = self.roi
        roi = frame[y1:y2, x1:x2]
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 削除: _compute_phash()
    # 削除: _hamming_distance()
```

**変更点まとめ:**

| 項目 | 変更前 | 変更後 |
|---|---|---|
| pHash計算 | `self._hasher.compute(image)` | `compute_phash(image)` |
| ハミング距離 | `self._hasher.compare(h1, h2)` | `hamming_distance(h1, h2)` |
| max_distance | `self.hash_size * self.hash_size`（可変） | `MAX_DISTANCE = 256`（定数） |
| hash_size引数 | あり（8 or 16） | 削除（16x16固定） |
| `_hasher`インスタンス | `cv2.img_hash.PHash.create()` | 削除 |
| `_compute_phash()`メソッド | あり | 削除 |
| `_hamming_distance()`メソッド | あり | 削除 |

#### create_template() 関数の変更

```python
def create_template(image_path: str, roi: tuple[int, int, int, int]) -> np.ndarray:
    """テスト画像からテンプレートハッシュを作成する"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    x1, y1, x2, y2 = roi
    roi_img = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    template_hash = compute_phash(gray)  # ← shared使用
    return template_hash
```

**変更点:**
- `hash_size` 引数を削除（16x16固定）
- `cv2.img_hash.PHash.create()` → `compute_phash()` に置換

#### デフォルトテンプレートパスの変更

```python
# 変更前
default_template_path = exp_dir / f"template_hash{args.hash_size}.npy"

# 変更後
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEMPLATE_PATH = _PROJECT_ROOT / "assets" / "templates" / "text" / "baito.npy"
```

#### CLIインターフェースの変更

| オプション | 変更 |
|---|---|
| `--hash-size` | **削除**（16x16固定） |
| `--threshold` | デフォルト値を閾値チューニング結果に更新 |
| `--template` | デフォルトパスを `assets/templates/text/baito.npy` に変更 |
| `--image`, `--create-template`, `--test-all`, `--debug` | 変更なし |

#### run_test_all() の変更

`BaitoTextRecognizer` のコンストラクタから `hash_size` 引数を削除。閾値のデフォルトは新しい値を使用。
それ以外のロジック（テストフィクスチャの走査・結果表示）は変更なし。

#### run_single_image() の変更

同上。`hash_size` 引数の削除のみ。

### 2. exp_002/main.py — テンプレートパスとデフォルト閾値の更新

```python
# 変更前
DEFAULT_TEMPLATE_PATH = EXP_001_DIR / "template_hash8.npy"

# 変更後
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEMPLATE_PATH = _PROJECT_ROOT / "assets" / "templates" / "text" / "baito.npy"
```

**その他の変更:**
- `--threshold` のデフォルト値を新しい閾値に更新
- `EXP_001_DIR` の定義は `BaitoTextRecognizer` のimport用に残す

### 3. exp_003/plugins/baito_text.py — テンプレートパスの更新

```python
# 変更前
DEFAULT_TEMPLATE_PATH = _EXP_001_DIR / "template_hash8.npy"

# 変更後
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TEMPLATE_PATH = _PROJECT_ROOT / "assets" / "templates" / "text" / "baito.npy"
```

> パス解決: `baito_text.py` → `plugins/` → `exp_003_gui_recognition_viewer/` → `experiments/` → プロジェクトルート

**その他の変更:**
- `_EXP_001_DIR` の定義は `BaitoTextRecognizer` のimport用に残す

### 4. テンプレート生成

#### 生成コマンド

```bash
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --create-template --image data/test_fixtures/text/positive/baito_001.png
```

#### 生成フロー

```
入力: data/test_fixtures/text/positive/baito_001.png (1920x1080)
  │
  ▼
1. ROI切り出し (750, 545, 1170, 600) → 420x55px
  │
  ▼
2. グレースケール変換 (BGR → Gray)
  │
  ▼
3. compute_phash() → 256bitハッシュ (32bytes)
   内部: 32x32リサイズ → DCT → 低周波16x16 → 平均二値化 → packbits
  │
  ▼
4. np.save() → assets/templates/text/baito.npy
```

#### 出力ファイル

| ファイル | 内容 | サイズ |
|---|---|---|
| `assets/templates/text/baito.npy` | 256bitテンプレートハッシュ | 32bytes + npyヘッダ |

### 5. 閾値チューニング

#### 方法

1. テンプレートを `baito_001.png` から生成
2. テストフィクスチャ全4枚に対して `compute_phash` → `hamming_distance` を算出
3. 正例と負例の距離分布を記録
4. 正例の最大距離と負例の最小距離の中間値を閾値候補とする
5. 全4枚で100%正解率を確認

#### 期待される距離分布

F-001（8x8）での距離分布は以下の通り:

| 画像 | 種別 | 8x8距離 |
|---|---|---|
| baito_001.png | 正例 | 0 |
| baito_002.png | 正例 | 1 |
| lobby_001.png | 負例 | 31 |
| wave1_active_001.png | 負例 | 26 |

16x16（256bit）でも同様に正例・負例間に大きなギャップが期待される。
実際の距離分布は実装時に計測し、exp_001 README.md に記録する。

#### 閾値の決定基準

```
閾値 = floor((正例最大距離 + 負例最小距離) / 2)
```

F-001の8x8では推奨閾値10（正例最大1、負例最小26）であった。
256bitでは距離レンジが4倍（64→256）になるため、同比率で拡大すると仮定すると
おおよそ40前後が予想されるが、DCTアルゴリズムの違いにより実際の値は異なる可能性がある。

**閾値は実測値で確定する。**

### 6. 旧テンプレートファイルの削除

以下のファイルを削除する:

- `experiments/exp_001_baito_text_recognition/template_hash8.npy`
- `experiments/exp_001_baito_text_recognition/template_hash16.npy`

## データフロー（変更後）

```
入力: BGR画像 (1920x1080)
  │
  ▼
1. ROI切り出し（変更なし）
   frame[545:600, 750:1170] → ROI画像 (420x55 BGR)
  │
  ▼
2. グレースケール変換（変更なし）
   BGR → Gray
  │
  ▼
3. pHash計算（★変更箇所）
   [変更前] cv2.img_hash.PHash.compute() → 8x8 = 64bitハッシュ
   [変更後] shared.recognition.compute_phash() → 16x16 = 256bitハッシュ
            内部: 32x32リサイズ → DCT → 低周波16x16 → 平均二値化
  │
  ▼
4. ハミング距離比較（★変更箇所）
   [変更前] cv2.img_hash.PHash.compare() → 距離 (0~64)
   [変更後] shared.recognition.hamming_distance() → 距離 (0~256)
  │
  ▼
5. 閾値判定（★閾値値の変更）
   距離 ≤ 閾値 → (True, 信頼度)
   距離 > 閾値 → (False, 信頼度)
```

## 作業手順

以下の順序で作業を行う:

### Step 1: テンプレート生成基盤の準備

1. `assets/templates/text/` ディレクトリを作成
2. exp_001/main.py の `create_template()` を `shared/recognition` に置換
3. テンプレートを生成: `baito_001.png` → `assets/templates/text/baito.npy`

### Step 2: 閾値チューニング

1. テストフィクスチャ4枚の各ハミング距離を計測
2. 正例・負例の距離分布を確認
3. 閾値を決定

### Step 3: BaitoTextRecognizer の改修

1. `cv2.img_hash.PHash` への依存を除去
2. `shared/recognition` のimportを追加
3. `recognize()` メソッドの実装を差し替え
4. `hash_size` 引数を削除、`MAX_DISTANCE = 256` 定数化
5. `_compute_phash()`, `_hamming_distance()` メソッドを削除
6. CLIの `--hash-size` 引数を削除、デフォルト閾値・テンプレートパスを更新

### Step 4: exp_002/exp_003 のパス更新

1. exp_002/main.py: `DEFAULT_TEMPLATE_PATH` を更新、`--threshold` デフォルト更新
2. exp_003/plugins/baito_text.py: `DEFAULT_TEMPLATE_PATH` を更新

### Step 5: 旧テンプレート削除

1. `template_hash8.npy` を削除
2. `template_hash16.npy` を削除

### Step 6: 検証

1. `--test-all data/test_fixtures/text/` で全4枚100%を確認
2. exp_001 README.md に結果を記録

## 検証項目

| # | 検証内容 | 判定基準 |
|---|---|---|
| 1 | テストフィクスチャ全通過 | 正例2枚: detected=True, 負例2枚: detected=False |
| 2 | 閾値の妥当性 | 正例最大距離と負例最小距離の間に十分なマージンがあること |
| 3 | exp_002 動作確認 | テンプレート読み込みエラーなく起動すること（カメラ不要、import確認のみ） |
| 4 | exp_003 BaitoTextPlugin 動作確認 | テンプレート読み込みエラーなくプラグイン初期化されること |

## 依存関係

- `shared/recognition/phash.py` — 既存（変更なし）
  - `compute_phash(image: np.ndarray) -> np.ndarray`
  - `hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int`
- OpenCV (`cv2.cvtColor`, `cv2.imread`, `cv2.rectangle`, `cv2.putText`) — 引き続き使用
- OpenCV `cv2.img_hash` — **依存削除**

## 備考

- `shared/recognition/phash.py` 自体に変更は不要。利用側（exp_001）の改修のみ
- `BaitoTextRecognizer` の `recognize()` の戻り値型 `tuple[bool, float]` は変更しない
  - F-004の `tuple[str, float]` とは異なるが、既存のexp_002/exp_003との互換性を維持するため
- 前処理にHSVフィルタを追加する改善は本リファクタリングのスコープ外（別途検討）
