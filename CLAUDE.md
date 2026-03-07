# CLAUDE.md — shakeop

## プロジェクト概要

Splatoon 3「サーモンラン NEXT WAVE」のリアルタイム解析オペレーターアプリ。
ゲーム画面を解析して Wave情報・ノルマ・オオモノ出現・潮位等を音声＋GUIで通知する。

## 参照ドキュメント

| ドキュメント | 用途 |
|---|---|
| **DEVELOPMENT.md** | 開発方法論・フロー・テスト戦略・Git運用（最上位ルール文書） |
| **docs/game_rules.md** | サーモンランNWのゲームルール（ドメイン知識） |
| **docs/salmon_run_state_transitions.md** | ゲーム内状態遷移（FSM設計用） |
| **docs/REQUIREMENTS_STANDARD.md** | 要求仕様書の記述基準 |
| **docs/DESIGN_STANDARD.md** | 機能設計書の記述基準 |
| **docs/REVIEW_CRITERIA.md** | 仕様書のレビュー基準 |
| **docs/BUGFIX_STANDARD.md** | 不具合修正の記述基準（手動テスト差し戻し時） |
| **docs/issues/index.md** | 全管理番号の一覧と完了状況 |

## 開発の鉄則

1. **実装前にドキュメントを完成させる** — 管理番号を発行し、要求仕様書と機能設計書を `docs/issues/` に保存してからコードを書く
2. **新機能は experiments/ にミニアプリとして作る** — いきなり src/ に書かない
3. **ミニアプリは shared/ の共通型・ユーティリティを使う**
4. **テスト・lint・型チェックは test-runner サブエージェントで実行する** — メインエージェントが直接実行しない
5. **ライブラリを追加・変更・削除したら `docs/TECH_STACK.md` を更新する**
6. **開発フロー・テスト戦略・Git運用の詳細は DEVELOPMENT.md に従う**

## 現在の作業状況

### 作業中

（なし）

<!-- 作業中タスクのテンプレート:
ステータス凡例: 📋 未着手 / 📝 仕様作成中 / 🔬 実装中 / 🧪 テスト中
- **F-NNN**: タスク名 — {ステータス}
  - 要求仕様書: `docs/issues/F-NNN_xxx/requirements.md`
  - 機能設計書: `docs/issues/F-NNN_xxx/design.md`
  - 実装先: `experiments/exp_NNN_xxx/`
  - 次のアクション: （具体的に何をすべきか）
-->

### 完了済み

`docs/issues/index.md` を参照。

## 技術スタック

- Python 3.12+ / uv
- PySide6 (GUI + 音声再生)
- OpenCV + ONNX Runtime (認識)
- pHash (数字・固定テキスト認識) — 汎用OCR不使用
- QSoundEffect (音声)
- ライセンス: MIT

## リポジトリ構成

```
shakeop/
├── CLAUDE.md            # ← このファイル
├── DEVELOPMENT.md       # 開発方法論（最上位ルール文書）
├── src/salmon_buddy/    # メインアプリ
├── shared/salmon_types/ # 共通の型定義（Protocol, dataclass, Enum）
├── shared/recognition/  # 共通の認識ユーティリティ（pHash等）
├── experiments/         # ミニアプリ群（exp_NNN_description/）
├── docs/issues/         # 管理番号別ドキュメント（要求仕様書・機能設計書）
├── models/              # ONNXモデル（gitignore）
├── assets/              # 音声・アイコン・テンプレート画像
├── data/                # 学習データ・テストfixture
├── tests/               # unit/ と integration/
└── tools/               # ユーティリティスクリプト
```

## 重要な設計判断

- 認識パイプラインは5段構成: Scene分類 → Object検出 → Digit認識(pHash) → Text識別(pHash) → Feature Match
- 数字・固定テキスト認識にはpHash（パーセプチュアルハッシュ）を使用
  - 汎用OCRはゲーム画面（独自フォント・動く背景・透過・斜め配置）に不向きと判明済み
- ゲーム状態はFSMで管理、Qt Signal/Slot でGUI/音声に伝播
- すべてのROI座標はFHD (1920x1080) 基準
- ONNXモデルはCPU推論前提（GPU不要）


## 用語集

| 用語 | 説明 |
|---|---|
| サーモンランNW | Splatoon 3のゲームモード「サーモンラン NEXT WAVE」の略称 |
| バイト | サーモンランNWのゲームをプレイすること |
| バイトの時間です | マッチング完了後、ゲーム開始前に表示される固定テキスト |
| Wave | バイト内の区切り。準備時間(10カウント)＋本番(100カウント)＋休憩で構成。通常Wave1〜3＋Extra Wave。バイトチームコンテストではWave1〜5 |
