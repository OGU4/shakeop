# [F-001] 「バイトの時間です」テキスト認識 — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-001 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-21 |
| ステータス | ✅ 完了 |

## 設計方針

FHD画像の固定位置 (ROI) を切り出し、グレースケール変換後にpHash（パーセプチュアルハッシュ）で
事前登録したテンプレートとのハミング距離を算出し、閾値以下であれば「バイトの時間です」テキストありと判定する。

OpenCV の `cv2.img_hash.PHash` は 8x8 (64bit) 固定のハッシュを生成する。
検証の結果、8x8で精度100%・処理時間0.09msと十分な性能が得られたため、これを採用した。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| ROI切り出し + pHash (OpenCV) | ✅ 採用 | 固定位置の固定テキスト認識に最適。高速かつ環境差に強い |
| テンプレートマッチング (cv2.matchTemplate) | ❌ 不採用 | 明るさ・コントラスト変動に弱い |
| 汎用OCR (Tesseract等) | ❌ 不採用 | ゲーム画面の独自フォント・動く背景に不向き（プロジェクト方針） |

## モジュール構成

- `experiments/exp_001_baito_text_recognition/main.py` — ミニアプリ本体
- `experiments/exp_001_baito_text_recognition/README.md` — 実験結果記録
- `src/salmon_buddy/recognition/text_identifier.py` — 統合先（将来）

## クラス・関数設計

```python
class BaitoTextRecognizer:
    """「バイトの時間です」テキストのpHash認識器"""

    def __init__(
        self,
        template_path: str,
        roi: tuple[int, int, int, int] = (750, 545, 1170, 600),
        threshold: int = 10,
        hash_size: int = 8,
    ):
        """
        Args:
            template_path: テンプレート画像のパス
            roi: (x1, y1, x2, y2) FHD基準
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
            hash_size: pHashのハッシュサイズ (8 or 16)
        """
        ...

    def recognize(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        フレームから「バイトの時間です」テキストの有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (テキストの有無, 信頼度 0.0-1.0)
        """
        ...

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出してグレースケール変換する"""
        ...

    def _compute_phash(self, image: np.ndarray) -> np.ndarray:
        """pHashを計算する"""
        ...

    def _hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> int:
        """2つのハッシュ間のハミング距離を算出する"""
        ...
```

## データフロー

```
入力: BGR画像 (1920x1080)
  │
  ▼
1. ROI切り出し
   frame[545:600, 750:1170] → ROI画像 (420x55)
  │
  ▼
2. グレースケール変換
   cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  │
  ▼
3. pHash計算
   cv2.img_hash.PHash で入力ROIのハッシュを算出
  │
  ▼
4. ハミング距離算出
   テンプレートハッシュとのハミング距離を計算
  │
  ▼
5. 判定
   ハミング距離 ≤ 閾値 → True（テキストあり）
   ハミング距離 > 閾値 → False（テキストなし）
   信頼度 = 1.0 - (ハミング距離 / 最大ハミング距離)
  │
  ▼
出力: (bool, float)
```

## テンプレート作成手順

1. テスト画像（「バイトの時間です」が写っている画像）を読み込む
2. ROI座標 (750, 545, 1170, 600) で切り出す
3. グレースケール変換する
4. pHashを計算して保存する（テンプレートハッシュ）
5. ミニアプリに `--create-template` モードを用意し、上記を自動化する

## ミニアプリのCLIインターフェース

```bash
# テンプレート作成
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --create-template --image <テンプレート元画像>

# 静止画テスト
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --image <テスト画像> --debug

# ハッシュサイズ比較（8x8 vs 16x16）
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --image <テスト画像> --hash-size 8 --debug
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --image <テスト画像> --hash-size 16 --debug
```

## 検証項目

ミニアプリで以下を検証し、README.md に結果を記録する。

| # | 検証内容 | 判定基準 | 結果 |
|---|---|---|---|
| 1 | 8x8 pHashの精度 | テスト画像に対して正解率 95% 以上 | ✅ 100% (4/4) |
| 2 | 16x16 pHashの精度 | テスト画像に対して正解率 95% 以上 | - (OpenCV PHashは64bit固定のため検証不要) |
| 3 | 8x8 vs 16x16 の速度比較 | 両方 5ms 以内であること | - (同上) |
| 4 | 最適な閾値の特定 | テキストあり/なし画像のハミング距離分布から決定 | ✅ 閾値=10 (正例:0-1, 負例:26-31) |

## 依存関係

- 前提: なし（最初のミニアプリ）
- 使用ライブラリ:
  - OpenCV (`cv2.img_hash.PHash`) — pHash計算
  - NumPy — 配列操作

## テスト計画

- ユニットテスト: `tests/unit/test_baito_text_recognition.py`
- テストデータ: `data/test_fixtures/text/`
  - テキストあり画像: 複数枚
  - テキストなし画像: 複数枚（他のシーンの画面キャプチャ）

## 備考

- ハッシュサイズは OpenCV PHash の仕様により 8x8 (64bit) 固定
- 閾値は検証の結果 10 に決定（正例と負例の距離差が25以上あり、十分な安全マージン）
- 将来、他の固定テキスト認識に拡張する際は、テンプレートを追加登録する方式で対応可能
