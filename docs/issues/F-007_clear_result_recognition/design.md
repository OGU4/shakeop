# F-007: "Clear!!" リザルト画面（クリア版）認識 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-007 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-03-01 |
| ステータス | ✅ 完了 |

## 設計方針

FHD画像の固定位置 (ROI) を切り出し、グレースケール変換後、
pHashでテンプレートとのハミング距離を算出して "Clear!!" の有無を判定する。

F-006 ("Work's Over!!") と同じROI全体の **1段判定** 方式。
"Clear!!" は画面左上に表示される固定テキストであり、
HSVフィルタ等の前処理は不要でグレースケール変換のみで十分な分離が得られる。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| ROI全体 + pHash 1段判定 | ✅ 採用 | 固定テキスト1パターン。グレースケールのみで距離差100（十分な分離） |
| HSV緑色抽出 + pHash | ❌ 不採用 | グレースケールのみで十分。前処理の追加は不要 |

## 認識パイプライン

```
入力フレーム (1920x1080)
  → ROI切り出し (18, 12, 370, 115) — 352x103 px
  → グレースケール変換
  → compute_phash (16x16, DCTベース, 256bit)
  → hamming_distance(frame_hash, template_hash)
  → 閾値判定 (≤50) → True/False
```

## 共通モジュール利用

```python
from shared.recognition.phash import compute_phash, hamming_distance
```

- `compute_phash`: 32x32リサイズ → DCT → 低周波16x16係数 → 平均値二値化 → 256bitハッシュ
- `hamming_distance`: ハミング距離計算

## ROI設計

- **座標**: `(18, 12, 370, 115)` — (x1, y1, x2, y2)
- **切り出しサイズ**: 352x103 px
- **基準**: FHD (1920x1080)
- **対象**: 画面左上の "Clear!!" テキスト部分

### ROI決定根拠

正例画像のHSV緑色テキスト検出で得られたバウンディングボックス `(76, 48, 358, 107)` に
上下左右の余裕を加えた座標。周囲の黒背景を含めることでpHashの分離性能を確保。

## テンプレート

| ファイル | 元画像の領域 | サイズ | 説明 |
|---|---|---|---|
| `clear_result.npy` | グレースケールROI全体 (352x103) | 16x16 pHash (32bytes) | "Clear!!" 全体のpHashハッシュ |

- **配置先**: `assets/templates/clear_result/clear_result.npy`
- **元画像**: `data/test_fixtures/clear_result/positive/vlcsnap-2026-02-22-17h10m54s284.png`

## 閾値設計（事前検証結果）

| 項目 | 値 |
|---|---|
| 正例の最大距離 | 0 |
| 負例の最小距離 | 100（wave/wave2カテゴリ） |
| 負例の最大距離 | 144 |
| 負例の平均距離 | 約124 |
| "Failure" 画面の距離 | 138 |
| 有効範囲 | 0〜100（幅100） |
| **最適閾値** | **50** |
| 正例側マージン | 50 |
| 負例側マージン | 50 |

## モジュール構成

```
experiments/exp_007_clear_result_recognition/
├── main.py                          # CLIエントリポイント
├── clear_result_recognizer.py       # ClearResultRecognizer クラス
└── README.md                        # 実験結果記録
```

**GUI統合:**

```
experiments/exp_003_gui_recognition_viewer/plugins/
└── clear_result.py                  # ClearResultPlugin (RecognitionPlugin Protocol準拠)
```

**変更対象（既存ファイル）:**

```
experiments/exp_003_gui_recognition_viewer/
└── main.py                          # _load_plugins() に ClearResultPlugin を追加登録
```

## クラス・関数設計

### ClearResultRecognizer（認識ロジック本体）

```python
class ClearResultRecognizer:
    """'Clear!!' のpHash認識器（1段判定）"""

    # ROI座標 (FHD 1920x1080 基準)
    CLEAR_RESULT_ROI = (18, 12, 370, 115)  # 352x103 px

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        clear_hash: np.ndarray,
        threshold: int = 50,
    ):
        """
        Args:
            clear_hash: "Clear!!" テンプレートのpHashハッシュ
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        ...

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームから "Clear!!" の有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "CLEAR", "NONE"
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
                "gray": np.ndarray,
            }
        """
        ...

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出す (352x103 BGR)"""
        ...

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """グレースケール変換"""
        ...
```

pHash計算には `shared/recognition/` の共通関数を使用する:

```python
from shared.recognition import compute_phash, hamming_distance
```

### ClearResultPlugin（GUI統合用）

```python
class ClearResultPlugin:
    """'Clear!!' 判定の認識プラグイン (RecognitionPlugin Protocol準拠)"""

    ROI = (18, 12, 370, 115)

    COLOR_DETECTED = (0, 255, 0)      # 緑
    COLOR_NOT_DETECTED = (0, 0, 255)  # 赤

    def __init__(self, template_dir: Path | None = None, threshold: int = 50) -> None:
        """
        exp_007の ClearResultRecognizer をラップ。
        テンプレートを読み込み認識器を初期化する。
        """
        ...

    @property
    def name(self) -> str:
        return "Clear!!判定"

    def process(self, frame: np.ndarray) -> dict:
        """
        Returns:
            {
                "detected": bool,
                "confidence": float,
                "clear_result": str,   # "CLEAR" or "NONE"
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
   frame[12:115, 18:370] → ROI画像 (352x103 BGR)
  │
  ▼
2. グレースケール変換（前処理）
   BGR → Grayscale
   → グレースケール画像 (352x103)
  │
  ▼
3. pHash判定
   グレースケール画像全体の pHash 計算
   clear_hash とハミング距離比較
  │
  ├── 距離 ≤ 50 → "CLEAR" + 信頼度
  └── 距離 > 50 → "NONE" + 信頼度
```

## GUI統合詳細

### プラグインロード

`main.py` の `_load_plugins()` に以下を追加:

```python
try:
    from plugins.clear_result import ClearResultPlugin
    plugins.append(ClearResultPlugin())
except Exception as e:
    print(f"Warning: ClearResultPlugin の読み込みに失敗: {e}")
```

### オーバーレイ描画仕様

| 要素 | 検出時 | 未検出時 |
|---|---|---|
| ROI矩形色 | 緑 `(0, 255, 0)` | 赤 `(0, 0, 255)` |
| テキスト位置 | ROI矩形の右横 | 同左 |
| テキスト内容 | `CLEAR: OK dist=0/50` | `CLEAR: NG dist=128/50` |
| フォント | `FONT_HERSHEY_SIMPLEX` | 同左 |
| フォントスケール | 0.7 | 同左 |

### ログ出力フォーマット

```
result=CLEAR    CLEAR:OK(d=0)
result=NONE     CLEAR:NG(d=128)
```

## CLIインターフェース（ミニアプリ）

```bash
# テンプレート作成
uv run python experiments/exp_007_clear_result_recognition/main.py \
    --create-template --image data/test_fixtures/clear_result/positive/vlcsnap-2026-02-22-17h10m54s284.png --debug

# ディレクトリ一括判定
uv run python experiments/exp_007_clear_result_recognition/main.py \
    --image-dir data/test_fixtures/clear_result/ --debug

# 閾値を変えて検証
uv run python experiments/exp_007_clear_result_recognition/main.py \
    --image-dir data/test_fixtures/clear_result/ --threshold 60
```

## 検証項目

| # | 検証内容 | 判定基準 | 結果 |
|---|---|---|---|
| 1 | "Clear!!" の正検出率 | 100% | ✅ 1/1 (100%) |
| 2 | 非 "Clear!!" 画面の正棄却率 | 100% | ✅ 154/154 (100%) |
| 3 | 最適な閾値の特定 | ハミング距離分布から決定 | ✅ 50（有効範囲: 0〜100） |
| 4 | 処理時間 | 1ms 以内 | ✅ 0.17ms |
| 5 | GUI統合：プラグインドロップダウンに表示 | 表示される | ✅ |
| 6 | GUI統合：リアルタイム認識動作 | 正常に判定される | ✅ |
| 7 | GUI統合：既存プラグインへの影響なし | 影響なし | ✅ |

## 依存関係

- 前提: R-001（pHash共通化 `shared/recognition/`）
- 使用モジュール:
  - `shared/recognition/` — `compute_phash`, `hamming_distance`
  - OpenCV (`cv2.cvtColor`) — グレースケール変換
  - NumPy — 配列操作

## 未決事項

- [x] ROI座標の決定 — 確定: (18, 12, 370, 115) 352x103px
- [x] 閾値の事前検証 — 確定: 50（正例0、負例最小100、マージン50）
- [x] テンプレート画像の作成 — `assets/templates/clear_result/clear_result.npy`
- [x] 精度検証 — 100% (155/155), 0.17ms
- [x] GUI統合 — ClearResultPlugin実装 + main.py登録
