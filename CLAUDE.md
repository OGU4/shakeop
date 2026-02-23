# CLAUDE.md — shakeop

## プロジェクト概要

Splatoon 3「サーモンラン NEXT WAVE」のリアルタイム解析オペレーターアプリ。
ゲーム画面を解析して Wave情報・ノルマ・オオモノ出現・潮位等を音声＋GUIで通知する。

## 最重要ドキュメント

**作業を始める前に必ず読むこと:**

1. **DEVELOPMENT.md** — 開発方法論（ミニアプリ駆動開発のルール）
2. **DESIGN.md** — アーキテクチャ設計、認識パイプライン詳細
3. **docs/game_rules.md** — サーモンランNWのゲームルール詳細（ドメイン知識）

## 開発の鉄則

```
★ 実装の前に必ず管理番号を発行し、要求仕様書と機能設計書を書く。
★ コード実装前に要求仕様書と機能設計書を必ず完成させる。これは厳守事項である。
★ 新機能は必ず experiments/ にミニアプリとして作る。いきなり src/ に書かない。
★ ミニアプリは shared/salmon_types/ の共通型（Protocol/dataclass）を使う。
★ 精度・速度が基準を満たしたら統合チェックリストに沿って src/ に移植する。
★ ミニアプリは統合後も削除しない（回帰テスト・比較実験用に残す）。
```

## 現在の作業状況

### 進行中

(なし)

### 完了済み

- **F-005**: Extra Wave判定 — ✅ 完了（`experiments/exp_005_extra_wave_recognition/`）
  - ROI全体の1段pHash判定。精度100% (150/150), 0.19ms, 閾値110
  - GUI統合: `exp_003_gui_recognition_viewer/plugins/extra_wave.py`
- **F-004**: Wave数判定 — ✅ 完了（`experiments/exp_004_wave_number_recognition/`）
  - ステップ1: CLI版ミニアプリ、ステップ2: GUI統合版、ステップ3: shared/recognition/ 共通化
- **F-003**: GUI認識ビューワー — ✅ 完了（`experiments/exp_003_gui_recognition_viewer/`）
- **B-001**: FPS過大表示バグ — ✅ 修正完了（`_new_frame_available` フラグ導入）

## 技術スタック

- Python 3.11+ / uv
- PySide6 (GUI + 音声再生)
- OpenCV + ONNX Runtime (認識)
- pHash (数字・固定テキスト認識) — 汎用OCR不使用
- QSoundEffect (音声)
- ライセンス: MIT

## 環境セットアップ

```bash
uv sync
```

## 実行

```bash
# メインアプリ
uv run python -m salmon_buddy

# ミニアプリ（例: exp_001 静止画テスト）
uv run python experiments/exp_001_baito_text_recognition/main.py \
    --image data/test_fixtures/text/positive/baito_001.png --debug

# ミニアプリ（例: exp_002 動画テスト）
uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 --debug
```

## テスト

```bash
uv run pytest tests/unit/         # ユニットテスト
uv run pytest tests/integration/  # 統合テスト
uv run pytest                     # 全部
```

## サブエージェント活用ルール

### テスト・品質チェックはサブエージェントに委譲する
- コード変更後の `pytest`, `ruff check`, `mypy --strict` → test-runner サブエージェントに委譲
- ミニアプリの精度検証バッチ → accuracy-checker サブエージェントに委譲

### サブエージェントに委譲しない作業
- 要求仕様書・機能設計書の作成（人間の承認が必要）
- アーキテクチャの設計判断
- experiments/ への新ファイル作成（メインエージェントが担当）

### 明示的な委譲指示の例
実装完了後は以下のように指示する:
- 「test-runner サブエージェントでテストを実行して」
- 「accuracy-checker で F-004 の精度検証をして」
```

---

## 使い方（Claude Codeでの操作）

### 自動委譲
`description` に "proactively" と書いてあるので、コード変更後にClaude Codeが自動的にtest-runnerを呼ぶことがあります。ただしClaude はサブエージェントを控えめに使う傾向があるので、どのステップをサブエージェントに委譲するか明示的に指示すると最良の結果が得られます。 

### 明示的な呼び出し（推奨）
```
> WaveNumberPlugin の実装が終わったので、test-runner サブエージェントで全テストを実行して
```
```
> accuracy-checker サブエージェントで exp_004 の test_fixtures/wave/ に対する精度検証をして
```

### 並列実行
複数のチェックを同時に走らせることもできます:
```
> test-runner でユニットテストを実行して、同時に ruff と mypy のチェックもして

## コーディング規約

```bash
uv run ruff format src/ shared/ experiments/ tests/
uv run ruff check src/ shared/ experiments/ tests/
uv run mypy --strict src/ shared/
```

## リポジトリ構成

```
salmon-buddy/
├── CLAUDE.md          # ← このファイル
├── DEVELOPMENT.md     # 開発方法論（ミニアプリ駆動開発のルール）
├── DESIGN.md          # アーキテクチャ設計
├── src/salmon_buddy/  # メインアプリ
├── shared/salmon_types/ # 共通の型定義（Protocol, dataclass, Enum）
├── shared/recognition/  # 共通の認識ユーティリティ（pHash等）
├── experiments/       # ミニアプリ群（exp_NNN_description/）
├── docs/issues/       # 管理番号別ドキュメント（要求仕様書・機能設計書）
├── models/            # ONNXモデル（gitignore）
├── assets/            # 音声・アイコン・テンプレート画像
├── data/              # 学習データ・テストfixture
├── tests/             # unit/ と integration/
└── tools/             # ユーティリティスクリプト
```

## 開発の進め方（必ずこの順序で）

```
1. 管理番号を発行      → docs/issues/index.md に追記
2. 要求仕様書を書く    → docs/issues/F-NNN_xxx/requirements.md
3. 機能設計書を書く    → docs/issues/F-NNN_xxx/design.md
4. ミニアプリで実装    → experiments/exp_NNN_xxx/
5. テスト・検証        → ミニアプリREADMEに結果記録
6. メインアプリに統合  → src/salmon_buddy/
7. ステータス更新      → docs/issues/index.md を「✅ 完了」に
```

管理番号: `F-NNN`（機能）/ `B-NNN`（バグ）

## ミニアプリの作り方

1. `experiments/exp_NNN_description/` を作成
2. `main.py`（--camera / --image / --debug 対応）と `README.md` を置く
3. `shared/salmon_types/` の Protocol と dataclass を import して使う
4. 結果を README.md に記録する
5. テンプレートは DEVELOPMENT.md のセクション3を参照

## 統合の手順

1. ミニアプリの README に結果記録済みを確認
2. Protocol 準拠を mypy で確認
3. ハードコード → config引数、print → logging、cv2.imshow → 削除
4. tests/unit/ にテスト追加
5. docs/integration_log.md に記録
6. 詳細チェックリストは DEVELOPMENT.md のセクション5を参照

## コミットメッセージ

```
[F-NNN] 機能の仕様・設計・実装
[B-NNN] バグ修正
[exp:NNN] ミニアプリ関連
[integrate] 統合作業
[app] メインアプリ（軽微な変更）
[shared] 共有コード
[test] テスト
[docs] ドキュメント
```

## 重要な設計判断

- 認識パイプラインは5段構成: Scene分類 → Object検出 → Digit認識(pHash) → Text識別(pHash) → Feature Match
- 数字・固定テキスト認識にはpHash（パーセプチュアルハッシュ）を使用
  - 汎用OCRはゲーム画面（独自フォント・動く背景・透過・斜め配置）に不向きと判明済み
- ゲーム状態はFSMで管理、Qt Signal/Slot でGUI/音声に伝播
- すべてのROI座標はFHD (1920x1080) 基準
- ONNXモデルはCPU推論前提（GPU不要）
- 詳細は DESIGN.md を参照

## 用語集

このプロジェクト固有の用語。会話やコード中で使われる。

| 用語 | 説明 |
|------|------|
| サーモンランNW | Nintendo SwitchのゲームソフトであるSplatoon 3のゲームモードの一つ「サーモンラン NEXT WAVE」の略称 |
| バイト | アルバイトの略称で、このリポジトリ内ではSalmonrun Next Waveのゲームをプレイしている事象を指す |
| バイトの時間です | サーモンランNWのマッチング完了後、ゲーム開始前に画面に表示される固定テキストの一つ |
| Wave | バイト1回のなかにWave（ウェーブ）と呼ばれる区切りが存在する。ひとつのWAVEは10カウントの「準備時間」と100カウントの「本番」、WAVEクリア時の短い「休憩時間」で構成されている。通常のバイトだとWaveは1~3と"Extra Wave"の4種類がある。不定期に開催される"バイトチームコンテスト"ではWave1~5の5種類がある |
