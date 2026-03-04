# exp_007: "Clear!!" リザルト画面認識

## 概要

| 項目 | 内容 |
|---|---|
| 管理番号 | F-007 |
| 目的 | 全Waveクリア後の "Clear!!" テキストをpHashで判定できるか検証 |
| 手法 | ROI切り出し → グレースケール変換 → pHash (1段判定) |
| ROI | (18, 12, 370, 115) — 352x103 px |
| 閾値 | 50 |
| 結果 | **精度100% (155/155)**, 0.17ms |

## 実行方法

```bash
# テンプレート作成
uv run python experiments/exp_007_clear_result_recognition/main.py \
    --create-template --image data/test_fixtures/clear_result/positive/vlcsnap-2026-02-22-17h10m54s284.png --debug

# ディレクトリ一括判定
uv run python experiments/exp_007_clear_result_recognition/main.py \
    --image-dir data/test_fixtures/clear_result/ --debug

# 単一画像判定
uv run python experiments/exp_007_clear_result_recognition/main.py \
    --image data/test_fixtures/clear_result/positive/vlcsnap-2026-02-22-17h10m54s284.png --debug
```

## パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--create-template` | - | テンプレート作成モード |
| `--image` | - | 単一画像パス |
| `--image-dir` | - | ディレクトリ一括判定 |
| `--template-dir` | `assets/templates/clear_result/` | テンプレートディレクトリ |
| `--threshold` | 50 | ハミング距離の閾値 |
| `--debug` | false | デバッグ情報表示 |

## 実験結果

### 精度

| カテゴリ | 正解数 | 総数 | 精度 |
|---|---|---|---|
| positive | 1 | 1 | 100% |
| negative | 154 | 154 | 100% |
| **総合** | **155** | **155** | **100%** |

### ハミング距離分布

| 項目 | 値 |
|---|---|
| 正例の最大距離 | 0 |
| 負例の最小距離 | 100 |
| 負例の最大距離 | 144 |
| 負例の平均距離 | 124.2 |
| 有効範囲 | 0〜100（幅100） |
| **最適閾値** | **50** |
| 正例側マージン | 50 |
| 負例側マージン | 50 |

### 処理速度

| 指標 | 値 |
|---|---|
| 平均処理時間 | 0.17ms |
| 要件 | 1ms以内 |
| 達成 | ✅ |

## テンプレート

| ファイル | 元画像 | サイズ |
|---|---|---|
| `assets/templates/clear_result/clear_result.npy` | `data/test_fixtures/clear_result/positive/vlcsnap-2026-02-22-17h10m54s284.png` | 32bytes (256bit pHash) |

## テストフィクスチャ

| カテゴリ | パス | 枚数 |
|---|---|---|
| 正例 | `data/test_fixtures/clear_result/positive/` | 1 |
| 負例 | `data/test_fixtures/clear_result/negative/` | 154 |

## GUI統合

- プラグイン: `experiments/exp_003_gui_recognition_viewer/plugins/clear_result.py`
- プラグイン名: "Clear!!判定"
- 登録先: `experiments/exp_003_gui_recognition_viewer/main.py` の `_load_plugins()`
