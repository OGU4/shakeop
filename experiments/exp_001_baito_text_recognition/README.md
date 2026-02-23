# exp_001: 「バイトの時間です」テキスト認識 (pHash)

## ステータス

| 項目 | 値 |
|---|---|
| 状態 | 🔬 実験中 |
| 管理番号 | F-001 |
| 統合先 | `src/salmon_buddy/recognition/text_identifier.py` (未統合) |
| 作成日 | 2026-02-21 |
| 最終更新 | 2026-02-23 |

## 目的

サーモンランNWのゲーム画面静止画から「バイトの時間です」テキストの有無を
ROI切り出し + pHash で認識できるか検証する。

## 手法

- ROI (750, 545, 1170, 600) でテキスト領域を切り出し
- グレースケール変換
- `shared/recognition` の 16x16 DCT pHash (256bit) でハッシュ計算
- テンプレートとのハミング距離で判定

## 実行方法

```bash
# テンプレート作成
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --create-template --image data/test_fixtures/text/positive/baito_001.png

# 静止画テスト
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --image data/test_fixtures/text/positive/baito_001.png --debug

# 全テスト画像で一括テスト
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --test-all data/test_fixtures/text/
```

## 結果

### 精度

| ハッシュサイズ | 正例正解率 | 負例正解率 | 総合正解率 | テスト画像数 | 備考 |
|---|---|---|---|---|---|
| 8x8 (64bit) | 2/2 (100%) | 2/2 (100%) | 4/4 (100%) | 4枚 | threshold=10, cv2.img_hash.PHash |
| 16x16 (256bit) | 2/2 (100%) | 2/2 (100%) | 4/4 (100%) | 4枚 | threshold=62, shared/recognition (R-001) |

### 速度

| 処理 | 時間 | 環境 |
|---|---|---|
| 8x8 推論 (CPU) | 平均 0.09ms | Python 3.12, OpenCV 4.13.0 |
| 16x16 推論 (CPU) | 平均 0.22ms | Python 3.12, OpenCV 4.13.0 (R-001) |

### 閾値調査

#### R-001以前 (8x8, 64bit, cv2.img_hash.PHash)

| 画像 | 種別 | ハミング距離 |
|---|---|---|
| baito_001.png | 正例 | 0 |
| baito_002.png | 正例 | 1 |
| lobby_001.png | 負例 | 31 |
| wave1_active_001.png | 負例 | 26 |

- 正例の距離: 0〜1
- 負例の距離: 26〜31
- 推奨閾値: **10**

#### R-001以後 (16x16, 256bit, shared/recognition)

| 画像 | 種別 | ハミング距離 |
|---|---|---|
| baito_001.png | 正例 | 0 |
| baito_002.png | 正例 | 2 |
| lobby_001.png | 負例 | 130 |
| wave1_active_001.png | 負例 | 122 |

- 正例の距離: 0〜2
- 負例の距離: 122〜130
- **正例と負例の間に大きなギャップ（距離 3〜121）がある → 閾値設定が容易**
- 閾値 = floor((2 + 122) / 2) = **62**（十分な安全マージンあり）

### 課題・知見

- 正例・負例のハミング距離の差が非常に大きく、pHash方式は本ユースケースに極めて有効
- 16x16 (256bit) では正例・負例間のギャップがさらに拡大（8x8: 25 → 16x16: 120）
- 処理時間は 0.22ms と要件 (5ms) を大幅に下回る（8x8の0.09msから増加したが十分高速）
- テスト画像が4枚と少ないため、追加画像での検証が望ましい

## 統合メモ

- ROI座標はハードコード → config引数に変更
- debug描画のコードは統合時に除去
- 閾値は検証結果に基づいて設定（推奨: 62）
