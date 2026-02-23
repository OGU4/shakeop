# F-006: "Work's Over!!" テキスト認識 — 要求仕様書

## 概要

サーモンランNWにおいて、Wave途中でゲームオーバー（全滅）になった際に表示される
**"Work's Over!!"** テキストを認識する機能を実装する。

## 背景・目的

- バイト中にWaveの制限時間内にチーム全員がやられるとゲームオーバーとなり、
  画面全体に "Work's Over!!" という固定テキストが表示される
- この状態を検出することで、ゲームオーバーイベントを発火し、
  FSMの状態遷移（Wave中 → ゲームオーバー → リザルト）を正確に追跡できる

## 機能要件

### FR-1: "Work's Over!!" テキストの認識
- 画面に表示される "Work's Over!!" テキストをpHashベースで認識する
- `shared/recognition/` の `compute_phash` / `hamming_distance` を使用する

### FR-2: テンプレート画像
- 正例スクリーンショットのROI領域からpHashを算出し、`.npy` 形式で保存する
- テンプレート元画像: `data/test_fixtures/works_over/positive/` に用意済み

### FR-3: ROI設定
- ROI座標: `(728, 872, 1048, 976)` — (x1, y1, x2, y2)、FHD (1920x1080) 基準
- 切り出しサイズ: 320x104 px
- 画面全体が固定の1枚画であり、"Work's Over!!" テキスト部分を切り出す

### FR-4: GUI認識ビューワーへの統合
- exp_003 GUI認識ビューワー (`experiments/exp_003_gui_recognition_viewer/`) にプラグインとして統合する
- `RecognitionPlugin` Protocolに準拠した `WorksOverPlugin` クラスを実装する
- 以下の4つのインターフェースを満たすこと:
  - `name` プロパティ: GUI表示用のプラグイン名を返す
  - `process(frame)`: フレームを受け取り認識結果を辞書で返す
  - `draw_overlay(frame, result)`: 認識結果をフレーム上にオーバーレイ描画する
  - `format_log(result)`: 認識結果をログ用文字列にフォーマットする
- `process()` の戻り値には以下のキーを含むこと:
  - `detected` (bool): "Work's Over!!" が検出されたか
  - `confidence` (float): 信頼度 0.0〜1.0
  - `works_over_result` (str): 判定結果 `"WORKS_OVER"` または `"NONE"`
  - `distance` (int): テンプレートとのハミング距離
  - `threshold` (int): 判定閾値

### FR-5: オーバーレイ描画
- ROI矩形を描画する（検出時: 緑、未検出時: 赤）
- ROI矩形の右横に判定結果テキストを表示する
  - 表示形式: `"WORKS_OVER: OK dist=<距離>/<閾値>"` または `"WORKS_OVER: NG dist=<距離>/<閾値>"`

### FR-6: プラグイン登録
- `main.py` の `_load_plugins()` に `WorksOverPlugin` を追加する
- 他プラグインと同様に `try-except` で安全にロードすること

## 非機能要件

### NFR-1: 認識精度
- 目標精度: 100%（正例・負例ともに誤判定なし）

### NFR-2: 処理速度
- 1フレームあたり1ms以下（pHash 1段判定を想定）

### NFR-3: 既存プラグインへの影響
- 既存プラグイン（BaitoTextPlugin, WaveNumberPlugin, ExtraWavePlugin）の動作に影響を与えないこと
- プラグインのロード失敗時はWarning出力のみでアプリ全体が停止しないこと

## 技術方針

- 認識方式: pHash（パーセプチュアルハッシュ）1段判定
  - `shared/recognition/` の `compute_phash` (16x16) と `hamming_distance` を使用
- テンプレートマッチング: ROI領域のpHashとテンプレートのpHashのハミング距離で判定
- 閾値: 50（exp_006にて検証済み。正例距離0、負例最小距離103、マージン53）
- プラグイン実装: exp_006の `WorksOverRecognizer` をラップし、`RecognitionPlugin` Protocolに適合させる
- 参考実装: `plugins/extra_wave.py`（同じpHash 1段判定方式のGUI統合プラグイン）

## テストフィクスチャ

- **正例**: `data/test_fixtures/works_over/positive/` — 1枚
- **負例**: `data/test_fixtures/works_over/negative/` — 148枚

## テンプレート

- **ファイル**: `assets/templates/works_over/works_over.npy`
- **形式**: NumPy `.npy` — uint8配列、shape=(32,)、256bit pHash
- **作成済み**: exp_006にて生成済み

## 制約・前提

- FHD (1920x1080) キャプチャ入力を前提とする
- "Work's Over!!" は画面全体に表示される固定の1枚画である
- GUI統合先は `experiments/exp_003_gui_recognition_viewer/` のプラグインアーキテクチャに従う

## 受け入れ基準

- [x] "Work's Over!!" 画面を正しく検出（exp_006にて検証済み: 1/1 正解）
- [x] 非 "Work's Over!!" 画面を正しく棄却（exp_006にて検証済み: 148/148 正解）
- [x] 認識精度100%（149/149）
- [x] 処理時間1ms以内（0.20ms）
- [x] `shared/recognition/` 共通関数使用
- [x] ミニアプリREADMEに結果記録（exp_006）
- [x] `WorksOverPlugin` が `RecognitionPlugin` Protocolに準拠
- [x] GUI認識ビューワーで正常に動作（プラグインドロップダウンに表示、認識・描画・ログ出力）
- [x] 既存プラグインの動作に影響なし

## ステータス

✅ 完了
