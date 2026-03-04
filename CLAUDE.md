# CLAUDE.md — shakeop

## プロジェクト概要

Splatoon 3「サーモンラン NEXT WAVE」のリアルタイム解析オペレーターアプリ。
ゲーム画面を解析して Wave情報・ノルマ・オオモノ出現・潮位等を音声＋GUIで通知する。

## 最重要ドキュメント

**作業を始める前に必ず読むこと:**

1. **DEVELOPMENT.md** — 開発方法論（ミニアプリ駆動開発のルール）
2. **DESIGN.md** — アーキテクチャ設計、認識パイプライン詳細
3. **docs/game_rules.md** — サーモンランNWのゲームルール詳細（ドメイン知識）
4. **docs/salmon_run_state_transitions.md** — ゲーム内状態遷移まとめ（FSM設計用）

**仕様書の作成・レビュー時に従うこと:**

5. **docs/REQUIREMENTS_STANDARD.md** — 要求仕様書の記述基準
6. **docs/DESIGN_STANDARD.md** — 機能設計書の記述基準
7. **docs/REVIEW_CRITERIA.md** — 要求仕様書・機能設計書のレビュー基準

## 開発の鉄則

```
★ 実装の前に必ず管理番号を発行し、要求仕様書と機能設計書を書く。
★ コード実装前に要求仕様書と機能設計書を必ず完成させる。これは厳守事項である。
★ 新機能は必ず experiments/ にミニアプリとして作る。いきなり src/ に書かない。
★ ミニアプリは shared/salmon_types/ の共通型（Protocol/dataclass）を使う。
★ 精度・速度が基準を満たしたら統合チェックリストに沿って src/ に移植する。
★ ミニアプリは統合後も削除しない（回帰テスト・比較実験用に残す）。
★ ライブラリの追加・変更・削除を行った場合は docs/TECH_STACK.md も更新すること。
★ 新規ライブラリ導入時は用途・選定理由・バージョンを TECH_STACK.md に追記すること。
```

### 開発方針

- **シンプルな機能を一つずつ作り、積み重ねて目的を達成する**
- 大きな機能を一度に作らない。小さく作って動作確認し、次の機能へ進む

### ドキュメント作成ルール

- **実装前に必ず「要求仕様書」と「機能設計書」を作成し、ファイルに保存すること**
- ドキュメントが保存されていない場合は、**実装を中止**する
- 要求仕様書：何を達成すべきか（入出力、制約、品質基準）。作成時は `docs/REQUIREMENTS_STANDARD.md` の基準に従うこと
- 機能設計書：どう実現するか（モジュール構成、アルゴリズム、データ構造）。作成時は `docs/DESIGN_STANDARD.md` の基準に従うこと
- ドキュメントは `docs/issues/{管理番号}_{名称}/` ディレクトリに置く
- **/clear 後でも実装がスムーズにできるよう、必要な情報を全て記述する**
- 暗黙知に頼らず、**自己完結したドキュメント**にする（前の会話コンテキストがなくても実装できること）

## 機能ごとの開発フロー（厳守）

各機能について、以下のフローを**厳守**する。**planモードは使わない**（通常モードで調査・計画を行う）。
管理番号: `F-NNN`（機能）/ `B-NNN`（バグ）/ `R-NNN`（リファクタリング）

### ステップ1: 調査・計画

通常モードで既存コードを調査し、要求仕様書と機能設計書を設計する。

- 関連コード・既存ミニアプリ・DESIGN.md等を調査
- 管理番号を決定（F-NNN / B-NNN / R-NNN）
- 要求仕様書（requirements.md）の内容を設計
- 機能設計書（design.md）の内容を設計

### ステップ2: ドキュメント保存

要求仕様書と機能設計書を `docs/issues/` にファイル保存する。**保存が完了するまで実装に進んではならない**。

1. `docs/issues/index.md` に管理番号を追記
2. `docs/issues/{管理番号}_{名称}/requirements.md` を作成 — **`docs/REQUIREMENTS_STANDARD.md` に従うこと**
3. `docs/issues/{管理番号}_{名称}/design.md` を作成 — **`docs/DESIGN_STANDARD.md` に従うこと**

### ステップ3: レビュー（Subagent + 人）

`docs/issues/` に保存されたドキュメントをSubagent（Agentツール）でレビューする。ユーザーも同時にレビューする。レビュー実行時は **`docs/REVIEW_CRITERIA.md`** の基準に従うこと。

### ステップ4: 修正（必要な場合）

レビューで問題があれば、再調査してドキュメントを更新する。**ステップ1〜3を問題がなくなるまで繰り返す**。

### ステップ5: 引き継ぎ・/clear

`CLAUDE.md` の「現在の作業状況」セクションを更新し、実装セッションに必要な情報を整える。その後ユーザーが `/clear` を実行する。

保存する仕様書は以下の要件を満たすこと:

```
★ 要求仕様書と機能設計書だけで実装に必要な情報がすべて揃っていること。
★ 会話の文脈に依存する情報を仕様書に残さない（/clear後は会話履歴が消える）。
★ 具体的なROI座標、ファイルパス、閾値、アルゴリズム手順等を明記すること。
★ CLAUDE.mdの「現在の作業状況」に、実装すべきタスクと参照先の仕様書パスを記載すること。
★ /clear後のClaude CodeがCLAUDE.md→仕様書の順に読めば実装を開始できる状態にすること。
```

### ステップ6: 実装（/clear後）

ドキュメント（要求仕様書・機能設計書・CLAUDE.md）を読んで実装する。

## 現在の作業状況

### 作業中

（なし）

<!-- 作業中タスクのテンプレート:
- **F-NNN**: タスク名 — 🔬 実装中
  - 要求仕様書: `docs/issues/F-NNN_xxx/requirements.md`
  - 機能設計書: `docs/issues/F-NNN_xxx/design.md`
  - 実装先: `experiments/exp_NNN_xxx/`
  - 次のアクション: （具体的に何をすべきか）
-->

### 完了済み

- **F-007**: "Clear!!" リザルト画面（クリア版）認識 — ✅ 完了（`experiments/exp_007_clear_result_recognition/`）
  - ROI全体の1段pHash判定。精度100% (155/155), 0.17ms, 閾値50
  - GUI統合: `exp_003_gui_recognition_viewer/plugins/clear_result.py`
- **F-006**: "Work's Over!!" テキスト認識 — ✅ 完了（`experiments/exp_006_works_over_recognition/`）
  - ROI全体の1段pHash判定。精度100% (149/149), 0.20ms, 閾値50
  - GUI統合: `exp_003_gui_recognition_viewer/plugins/works_over.py`
- **R-001**: pHash共通化リファクタリング — ✅ 完了
  - exp_001/exp_002/exp_003の`cv2.img_hash.PHash`(8x8) → `shared/recognition`(16x16) に統一
  - 精度100% (4/4), 0.22ms, 閾値62

- **F-005**: Extra Wave判定 — ✅ 完了（`experiments/exp_005_extra_wave_recognition/`）
  - ROI全体の1段pHash判定。精度100% (150/150), 0.19ms, 閾値110
  - GUI統合: `exp_003_gui_recognition_viewer/plugins/extra_wave.py`
- **F-004**: Wave数判定 — ✅ 完了（`experiments/exp_004_wave_number_recognition/`）
  - ステップ1: CLI版ミニアプリ、ステップ2: GUI統合版、ステップ3: shared/recognition/ 共通化
- **F-003**: GUI認識ビューワー — ✅ 完了（`experiments/exp_003_gui_recognition_viewer/`）
- **B-001**: FPS過大表示バグ — ✅ 修正完了（`_new_frame_available` フラグ導入）

## 技術スタック

- Python 3.12+ / uv
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

- テストは `tests/` ディレクトリに置く
- **テスト実行はSubagent（Agentツール）を使う**
- **テスト結果は `tests/results/` にファイル保存する**
  - ファイル名：`{管理番号}_test_result.txt`（例：`F-007_test_result.txt`）
  - 内容：pytestの `-v` 出力をそのまま保存する

## テスト実行ルール（厳守）

```
★ テスト・品質チェックは必ずサブエージェント（test-runner）を使って実行すること。
★ メインエージェントが直接 pytest, ruff, mypy を実行してはならない。
★ コードを変更したら、必ずサブエージェントでテストを実行すること。
```

### test-runner サブエージェントに委譲する作業
- `pytest`（ユニットテスト・統合テスト）
- `ruff check`（リンター）
- `ruff format --check`（フォーマットチェック）
- `mypy --strict`（型チェック）

### accuracy-checker サブエージェントに委譲する作業
- ミニアプリの精度検証バッチ（test_fixturesに対するバッチテスト）

### サブエージェントに委譲しない作業
- 要求仕様書・機能設計書の作成（人間の承認が必要）
- アーキテクチャの設計判断
- experiments/ への新ファイル作成（メインエージェントが担当）

### 実装中のテスト実行タイミング
- コード変更のたびにtest-runnerを呼ぶ（修正→テスト→修正→テストのサイクル）
- テストが失敗したらメインエージェントが修正し、再度test-runnerで確認する
- すべてのテストがパスするまでこのサイクルを繰り返す

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
--- ステップ1: 調査・計画（通常モード。planモードは使わない） ---
1. 関連コード・既存設計を調査
2. 管理番号を決定、要求仕様書・機能設計書の内容を設計

--- ステップ2: ドキュメント保存（保存完了まで実装に進まない） ---
3. 管理番号を発行      → docs/issues/index.md に追記
4. 要求仕様書を書く    → docs/issues/F-NNN_xxx/requirements.md（docs/REQUIREMENTS_STANDARD.md に従う）
5. 機能設計書を書く    → docs/issues/F-NNN_xxx/design.md（docs/DESIGN_STANDARD.md に従う）

--- ステップ3: レビュー（Subagent + 人） ---
6. Subagent（Agentツール）でドキュメントレビュー（docs/REVIEW_CRITERIA.md の基準に従う）
7. ユーザーも同時にレビュー

--- ステップ4: 修正（必要な場合） ---
※ レビューで問題があればステップ1〜3を繰り返す

--- ステップ5: 引き継ぎ・/clear ---
8. CLAUDE.md「現在の作業状況 > 作業中」に追記
9. ユーザーが /clear を実行

--- ステップ6: 実装（/clear後） ---
10. CLAUDE.md → 仕様書を読んで実装内容を把握
11. ミニアプリで実装    → experiments/exp_NNN_xxx/
12. テスト実行          → test-runner サブエージェントで pytest/ruff/mypy を実行（厳守）
13. 精度検証            → accuracy-checker サブエージェントで検証（必要な場合）
14. テスト・検証結果    → ミニアプリREADMEに結果記録
15. メインアプリに統合  → src/salmon_buddy/
16. ステータス更新      → docs/issues/index.md を「✅ 完了」に
```

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
[R-NNN] リファクタリング
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
