# F-006: "Work's Over!!" テキスト認識 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-006 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-23 |
| ステータス | ✅ 完了 |

## 設計方針

FHD画像の固定位置 (ROI) を切り出し、グレースケール変換後、
pHashでテンプレートとのハミング距離を算出して "Work's Over!!" の有無を判定する。

F-005 (Extra Wave) と同じROI全体の **1段判定** 方式だが、
"Work's Over!!" は画面全体に表示される固定の1枚画であり、
HSVフィルタ等の前処理は不要でグレースケール変換のみで十分な分離が得られる。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| ROI全体 + pHash 1段判定 | ✅ 採用 | 固定テキスト1パターン。グレースケールのみで距離差103（十分な分離） |
| HSV白色抽出 + pHash | ❌ 不採用 | グレースケールのみで十分。前処理の追加は不要 |

## 認識パイプライン

```
入力フレーム (1920x1080)
  → ROI切り出し (728, 872, 1048, 976) — 320x104 px
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

- **座標**: `(728, 872, 1048, 976)` — (x1, y1, x2, y2)
- **切り出しサイズ**: 320x104 px
- **基準**: FHD (1920x1080)
- **対象**: "Work's Over!!" テキスト部分

## テンプレート

| ファイル | 元画像の領域 | サイズ | 説明 |
|---|---|---|---|
| `works_over.npy` | グレースケールROI全体 (320x104) | 16x16 pHash (32bytes) | "Work's Over!!" 全体のpHashハッシュ |

- **配置先**: `assets/templates/works_over/works_over.npy`（作成済み）
- **元画像**: `data/test_fixtures/works_over/positive/vlcsnap-2026-02-23-19h02m23s715.png`

## 閾値設計

| 項目 | 値 |
|---|---|
| 正例の最大距離 | 0 |
| 負例の最小距離 | 103 |
| 負例の最大距離 | 146 |
| 負例の平均距離 | 128.4 |
| 有効範囲 | 0〜103（幅103） |
| **最適閾値** | **50** |
| 正例側マージン | 50 |
| 負例側マージン | 53 |

## モジュール構成

```
experiments/exp_006_works_over_recognition/
├── main.py                        # CLIエントリポイント
├── works_over_recognizer.py       # WorksOverRecognizer クラス
└── README.md                      # 実験結果記録
```

**GUI統合:**

```
experiments/exp_003_gui_recognition_viewer/plugins/
└── works_over.py                  # WorksOverPlugin (RecognitionPlugin Protocol準拠)
```

**変更対象（既存ファイル）:**

```
experiments/exp_003_gui_recognition_viewer/
└── main.py                        # _load_plugins() に WorksOverPlugin を追加登録
```

## クラス・関数設計

### WorksOverRecognizer（認識ロジック本体）

```python
class WorksOverRecognizer:
    """'Work's Over!!' のpHash認識器（1段判定）"""

    # ROI座標 (FHD 1920x1080 基準)
    WORKS_OVER_ROI = (728, 872, 1048, 976)  # 320x104 px

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        works_over_hash: np.ndarray,
        threshold: int = 50,
    ):
        """
        Args:
            works_over_hash: "Work's Over!!" テンプレートのpHashハッシュ
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        ...

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームから "Work's Over!!" の有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "WORKS_OVER", "NONE"
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
        """ROI領域を切り出す (320x104 BGR)"""
        ...

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """グレースケール変換"""
        ...
```

pHash計算には `shared/recognition/` の共通関数を使用する:

```python
from shared.recognition import compute_phash, hamming_distance
```

### WorksOverPlugin（GUI統合用）

```python
class WorksOverPlugin:
    """'Work's Over!!' 判定の認識プラグイン (RecognitionPlugin Protocol準拠)"""

    ROI = (728, 872, 1048, 976)

    COLOR_DETECTED = (0, 255, 0)      # 緑
    COLOR_NOT_DETECTED = (0, 0, 255)  # 赤

    def __init__(self, template_dir: Path | None = None, threshold: int = 50) -> None:
        """
        exp_006の WorksOverRecognizer をラップ。
        テンプレートを読み込み認識器を初期化する。
        """
        ...

    @property
    def name(self) -> str:
        return "Work's Over!!判定"

    def process(self, frame: np.ndarray) -> dict:
        """
        Returns:
            {
                "detected": bool,
                "confidence": float,
                "works_over_result": str,   # "WORKS_OVER" or "NONE"
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
   frame[872:976, 728:1048] → ROI画像 (320x104 BGR)
  │
  ▼
2. グレースケール変換（前処理）
   BGR → Grayscale
   → グレースケール画像 (320x104)
  │
  ▼
3. pHash判定
   グレースケール画像全体の pHash 計算
   works_over_hash とハミング距離比較
  │
  ├── 距離 ≤ 50 → "WORKS_OVER" + 信頼度
  └── 距離 > 50 → "NONE" + 信頼度
```

## GUI統合詳細

### プラグインロード

`main.py` の `_load_plugins()` に以下を追加:

```python
try:
    from plugins.works_over import WorksOverPlugin
    plugins.append(WorksOverPlugin())
except Exception as e:
    print(f"Warning: WorksOverPlugin の読み込みに失敗: {e}")
```

### オーバーレイ描画仕様

| 要素 | 検出時 | 未検出時 |
|---|---|---|
| ROI矩形色 | 緑 `(0, 255, 0)` | 赤 `(0, 0, 255)` |
| テキスト位置 | ROI矩形の右横 | 同左 |
| テキスト内容 | `WORKS_OVER: OK dist=0/50` | `WORKS_OVER: NG dist=128/50` |
| フォント | `FONT_HERSHEY_SIMPLEX` | 同左 |
| フォントスケール | 0.7 | 同左 |

### ログ出力フォーマット

```
result=WORKS_OVER  WORKS_OVER:OK(d=0)
result=NONE     WORKS_OVER:NG(d=128)
```

## CLIインターフェース（ミニアプリ）

```bash
# テンプレート作成
uv run python experiments/exp_006_works_over_recognition/main.py \
    --create-template --image data/test_fixtures/works_over/positive/vlcsnap-2026-02-23-19h02m23s715.png --debug

# ディレクトリ一括判定
uv run python experiments/exp_006_works_over_recognition/main.py \
    --image-dir data/test_fixtures/works_over/ --debug

# 閾値を変えて検証
uv run python experiments/exp_006_works_over_recognition/main.py \
    --image-dir data/test_fixtures/works_over/ --threshold 60
```

## 検証項目

| # | 検証内容 | 判定基準 | 結果 |
|---|---|---|---|
| 1 | "Work's Over!!" の正検出率 | 100% | ✅ 1/1 (100%) |
| 2 | 非 "Work's Over!!" 画面の正棄却率 | 100% | ✅ 148/148 (100%) |
| 3 | 最適な閾値の特定 | ハミング距離分布から決定 | ✅ 50（有効範囲: 0〜103） |
| 4 | 処理時間 | 1ms 以内 | ✅ 0.20ms |
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

- [x] ROI座標の決定 — 確定: (728, 872, 1048, 976) 320x104px
- [x] テンプレート画像の作成 — 作成済み: `assets/templates/works_over/works_over.npy`
- [x] 閾値の決定 — 確定: 50（正例0、負例最小103、マージン53）
- [x] GUI統合（WorksOverPlugin実装 + main.py登録）
