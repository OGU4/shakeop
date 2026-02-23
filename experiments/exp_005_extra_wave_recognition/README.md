# exp_005: Extra Wave判定

## 概要

| 項目 | 内容 |
|------|------|
| 管理番号 | F-005 |
| 目的 | ゲーム画面の「EXTRA WAVE」表示をpHashで判定する |
| 手法 | ROI切り出し → HSV白色テキスト抽出 → pHash (1段判定) |
| 前提機能 | F-004（Wave数判定・pHash共通化） |

## 実行方法

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

## パラメータ

| パラメータ | 値 | 説明 |
|---|---|---|
| ROI座標 | (38, 35, 238, 80) | 200x45 px (FHD基準) |
| HSV下限 | [0, 0, 210] | 白色テキスト抽出 |
| HSV上限 | [180, 80, 255] | 白色テキスト抽出 |
| モルフォロジー | MORPH_OPEN (3x3) | ノイズ除去 |
| pHash | 16x16 (256bit) | DCTベース |
| 閾値 | 110 | ハミング距離（テストデータから最適化） |

## 実験結果

### テストデータ

- 正例: `data/test_fixtures/wave/extra/` (37枚)
- 負例: `data/test_fixtures/wave/negative/` (38枚), `wave1/` (25枚), `wave2/` (15枚), `wave3/` (21枚), `wave4/` (6枚), `wave5/` (8枚)
- 合計: 150枚

### 精度 (閾値=110)

| カテゴリ | 正解数/総数 | 精度 |
|---|---|---|
| extra | 37/37 | 100% |
| negative | 38/38 | 100% |
| wave1 | 25/25 | 100% |
| wave2 | 15/15 | 100% |
| wave3 | 21/21 | 100% |
| wave4 | 6/6 | 100% |
| wave5 | 8/8 | 100% |
| **総合** | **150/150** | **100%** |

### 閾値最適化

- 初期値 116 では wave1/ で5枚の誤検出（距離 111, 115, 116, 116, 116）
- extra/ の最大距離: 75（信頼度 0.7070）
- 有効範囲: 75〜111（幅36）
- 最適閾値: **110**（有効範囲内、全データで100%正解）

### 処理速度

- 平均処理時間: **0.19ms**（基準: 1ms以内）

### テンプレート情報

- テンプレート元画像: `data/test_fixtures/wave/extra/vlcsnap-2026-02-22-17h36m07s995.png`
- テンプレートファイル: `assets/templates/wave/extra_wave.npy`
- テンプレートshape: (32,) — 16x16 pHash, 256bit
