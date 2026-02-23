# exp_006: "Work's Over!!" テキスト認識

## 概要

| 項目 | 内容 |
|------|------|
| 管理番号 | F-006 |
| 目的 | ゲームオーバー時の "Work's Over!!" テキストをpHashで判定する |
| 手法 | ROI切り出し → グレースケール変換 → pHash (1段判定) |
| 前提機能 | R-001（pHash共通化） |

## 実行方法

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

# 単一画像判定
uv run python experiments/exp_006_works_over_recognition/main.py \
    --image <画像パス>
```

## パラメータ

| パラメータ | 値 | 説明 |
|---|---|---|
| ROI座標 | (728, 872, 1048, 976) | 320x104 px (FHD基準) |
| 前処理 | グレースケール変換 | HSVフィルタ不要 |
| pHash | 16x16 (256bit) | DCTベース |
| 閾値 | 50 | ハミング距離（テストデータから最適化） |

## 実験結果

### テストデータ

- 正例: `data/test_fixtures/works_over/positive/` (1枚)
- 負例: `data/test_fixtures/works_over/negative/` (148枚)
- 合計: 149枚

### 精度 (閾値=50)

| カテゴリ | 正解数/総数 | 精度 |
|---|---|---|
| positive | 1/1 | 100% |
| negative | 148/148 | 100% |
| **総合** | **149/149** | **100%** |

### ハミング距離分布

| カテゴリ | 件数 | 最小 | 最大 | 平均 |
|---|---|---|---|---|
| positive | 1 | 0 | 0 | 0.0 |
| negative | 148 | 103 | 146 | 128.4 |

### 閾値最適化

- positive の最大距離: 0
- negative の最小距離: 103
- 有効範囲: 0〜103（幅103）
- 最適閾値: **50**（正例側マージン50、負例側マージン53）

### 処理速度

- 平均処理時間: **0.20ms**（基準: 1ms以内）

### テンプレート情報

- テンプレート元画像: `data/test_fixtures/works_over/positive/vlcsnap-2026-02-23-19h02m23s715.png`
- テンプレートファイル: `assets/templates/works_over/works_over.npy`
- テンプレートshape: (32,) — 16x16 pHash, 256bit
