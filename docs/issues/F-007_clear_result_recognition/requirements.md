# F-007: "Clear!!" リザルト画面（クリア版）認識 — 要求仕様書

## 概要

サーモンランNWにおいて、全Waveクリア後に表示される
**"Clear!!"** テキストを認識する機能を実装する。

## 背景・目的

- バイト中に全Wave（通常Wave 3、Extra Wave含む場合はExtra Wave）をクリアすると、
  画面左上に "Clear!!" という緑色のグラフィティ風テキストが表示される
- この状態を検出することで、ゲームクリアイベントを発火し、
  FSMの状態遷移（Wave中 → ゲームクリア → リザルト）を正確に追跡できる
- 同じ左上領域に表示される "Failure" テキスト（ゲームオーバー時）との区別が必要

## 機能要件

### FR-1: "Clear!!" テキストの認識
- 画面に表示される "Clear!!" テキストをpHashベースで認識する
- `shared/recognition/` の `compute_phash` / `hamming_distance` を使用する

### FR-2: テンプレート画像
- 正例スクリーンショットのROI領域からpHashを算出し、`.npy` 形式で保存する
- テンプレート元画像: `data/test_fixtures/clear_result/positive/` に配置

### FR-3: ROI設定
- ROI座標: `(18, 12, 370, 115)` — (x1, y1, x2, y2)、FHD (1920x1080) 基準
- 切り出しサイズ: 352x103 px
- 画面左上に表示される "Clear!!" テキスト部分を切り出す

### FR-4: GUI認識ビューワーへの統合
- exp_003 GUI認識ビューワー (`experiments/exp_003_gui_recognition_viewer/`) にプラグインとして統合する
- `RecognitionPlugin` Protocolに準拠した `ClearResultPlugin` クラスを実装する
- 以下の4つのインターフェースを満たすこと:
  - `name` プロパティ: GUI表示用のプラグイン名を返す
  - `process(frame)`: フレームを受け取り認識結果を辞書で返す
  - `draw_overlay(frame, result)`: 認識結果をフレーム上にオーバーレイ描画する
  - `format_log(result)`: 認識結果をログ用文字列にフォーマットする
- `process()` の戻り値には以下のキーを含むこと:
  - `detected` (bool): "Clear!!" が検出されたか
  - `confidence` (float): 信頼度 0.0〜1.0
  - `clear_result` (str): 判定結果 `"CLEAR"` または `"NONE"`
  - `distance` (int): テンプレートとのハミング距離
  - `threshold` (int): 判定閾値

### FR-5: オーバーレイ描画
- ROI矩形を描画する（検出時: 緑、未検出時: 赤）
- ROI矩形の右横に判定結果テキストを表示する
  - 表示形式: `"CLEAR: OK dist=<距離>/<閾値>"` または `"CLEAR: NG dist=<距離>/<閾値>"`

### FR-6: プラグイン登録
- `main.py` の `_load_plugins()` に `ClearResultPlugin` を追加する
- 他プラグインと同様に `try-except` で安全にロードすること

## 非機能要件

### NFR-1: 認識精度
- 目標精度: 100%（正例・負例ともに誤判定なし）

### NFR-2: 処理速度
- 1フレームあたり1ms以下（pHash 1段判定を想定）

### NFR-3: 既存プラグインへの影響
- 既存プラグイン（BaitoTextPlugin, WaveNumberPlugin, ExtraWavePlugin, WorksOverPlugin）の動作に影響を与えないこと
- プラグインのロード失敗時はWarning出力のみでアプリ全体が停止しないこと

## 技術方針

- 認識方式: pHash（パーセプチュアルハッシュ）1段判定
  - `shared/recognition/` の `compute_phash` (16x16) と `hamming_distance` を使用
- テンプレートマッチング: ROI領域のpHashとテンプレートのpHashのハミング距離で判定
- 閾値: 50（事前検証結果: 正例距離0、負例最小距離100、マージン50）
- プラグイン実装: exp_007の `ClearResultRecognizer` をラップし、`RecognitionPlugin` Protocolに適合させる
- 参考実装: `plugins/works_over.py`（同じpHash 1段判定方式のGUI統合プラグイン）

## テストフィクスチャ

- **正例**: `data/test_fixtures/clear_result/positive/` — 1枚
  - 元画像: `data/test_fixtures/wave/negative/vlcsnap-2026-02-22-17h10m54s284.png`
- **負例**: `data/test_fixtures/clear_result/negative/` — 既存フィクスチャから収集
  - `wave/negative/` から37枚（正例1枚を除く）
  - `wave/extra/` から37枚
  - `wave/wave1/`〜`wave/wave5/` から全枚
  - `text/positive/`, `text/negative/` から全枚
  - `works_over/positive/` から1枚
  - "Failure" 画面 (`vlcsnap-2026-02-22-17h23m45s387.png`) を含む

## テンプレート

- **ファイル**: `assets/templates/clear_result/clear_result.npy`
- **形式**: NumPy `.npy` — uint8配列、shape=(32,)、256bit pHash
- **元画像**: 正例画像のROI (18, 12, 370, 115) をグレースケール変換して生成

## 制約・前提

- FHD (1920x1080) キャプチャ入力を前提とする
- "Clear!!" は画面左上に表示されるグラフィティ風フォントの固定テキストである
- GUI統合先は `experiments/exp_003_gui_recognition_viewer/` のプラグインアーキテクチャに従う

## 受け入れ基準

- [x] "Clear!!" 画面を正しく検出（exp_007にて検証済み: 1/1 正解）
- [x] 非 "Clear!!" 画面を正しく棄却（exp_007にて検証済み: 154/154 正解）
- [x] 認識精度100%（155/155）
- [x] 処理時間1ms以内（0.17ms）
- [x] `shared/recognition/` 共通関数使用
- [x] ミニアプリREADMEに結果記録（exp_007）
- [x] `ClearResultPlugin` が `RecognitionPlugin` Protocolに準拠
- [x] GUI認識ビューワーで正常に動作（プラグインドロップダウンに表示、認識・描画・ログ出力）
- [x] 既存プラグインの動作に影響なし

## ステータス

✅ 完了
