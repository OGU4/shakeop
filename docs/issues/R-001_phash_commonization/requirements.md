# [R-001] pHash共通化リファクタリング — 要求仕様書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | R-001 |
| 作成日 | 2026-02-23 |
| ステータス | ✅ 仕様確定 |
| 対象実験 | exp_001, exp_002, exp_003 |
| 前提 | F-004 ステップ3（shared/recognition/ 共通化済み） |

## 背景・目的

F-004のステップ3で`shared/recognition/`にpHash共通関数（`compute_phash`, `hamming_distance`）が切り出された。
F-004, F-005はこの共通関数を使用しているが、先行して実装されたF-001/F-002/F-003は
OpenCV組込みの`cv2.img_hash.PHash`（8x8固定, 64bit）を独自に使用している。

本リファクタリングでは、F-001/F-002/F-003のpHash実装を`shared/recognition/`に統一し、
プロジェクト全体で同一のpHash基盤を使用する状態にする。

### 現状の不一致

| 実験 | pHash実装 | ハッシュサイズ | テンプレート |
|---|---|---|---|
| exp_001 (F-001) | `cv2.img_hash.PHash` (OpenCV組込み) | 8x8 = 64bit | `exp_001/template_hash8.npy` |
| exp_002 (F-002) | exp_001をimport | 同上 | 同上 |
| exp_003 (F-003) BaitoTextPlugin | exp_001をimport | 同上 | 同上 |
| exp_004 (F-004) | `shared/recognition` | 16x16 = 256bit | `assets/templates/wave/*.npy` |
| exp_005 (F-005) | `shared/recognition` | 16x16 = 256bit | `assets/templates/wave/*.npy` |

### リファクタリング後の目標状態

| 実験 | pHash実装 | ハッシュサイズ | テンプレート |
|---|---|---|---|
| 全実験共通 | `shared/recognition` | 16x16 = 256bit | `assets/templates/` 配下 |

## スコープ

**スコープ内:**

1. exp_001 `BaitoTextRecognizer` のpHash実装を `shared/recognition` に置換
2. テンプレートハッシュの新規生成（256bit）と `assets/templates/text/` への配置
3. 閾値の再チューニング（64bit範囲→256bit範囲）
4. exp_001 `create_template()` 関数の `shared/recognition` 対応
5. exp_002 のテンプレートパス更新
6. exp_003 `BaitoTextPlugin` のテンプレートパス更新
7. 既存テストフィクスチャ（4枚）での精度検証

**スコープ外:**

- 前処理パイプラインの変更（グレースケール変換のまま。HSVフィルタへの変更は行わない）
- テストフィクスチャの追加収集
- ユニットテストの追加・変更
- exp_001 README.md の構成変更（結果セクションの更新は行う）
- src/ への統合作業

## 要求事項

### 必須要求 (MUST)

#### pHash実装の統一

- exp_001 `BaitoTextRecognizer` が `shared.recognition.compute_phash` と `shared.recognition.hamming_distance` を使用すること
- `cv2.img_hash.PHash` への依存を完全に除去すること
- ハッシュサイズは 16x16（256bit）とすること（`shared/recognition` のデフォルト動作）

#### テンプレート

- 新しい256bitテンプレートを `assets/templates/text/baito.npy` に配置すること
- テンプレートは既存の正例画像（`data/test_fixtures/text/positive/baito_001.png`）から生成すること
- exp_001 の `--create-template` モードも `shared/recognition/compute_phash` を使用するように更新すること

#### テンプレートパスの更新

- exp_002 `DEFAULT_TEMPLATE_PATH` を新しいテンプレートパスに更新すること
- exp_003 `BaitoTextPlugin` の `DEFAULT_TEMPLATE_PATH` を新しいテンプレートパスに更新すること

#### 閾値

- 256bitハッシュに対応した閾値を設定すること
- 閾値はテストフィクスチャ4枚（正例2・負例2）の全通過を条件として決定すること
- 正例と負例の間に十分なマージンを確保すること

#### 精度

- 既存テストフィクスチャ全4枚（正例2枚・負例2枚）で100%正解率を維持すること
  - 正例（baito_001.png, baito_002.png）→ テキストあり判定
  - 負例（lobby_001.png, wave1_active_001.png）→ テキストなし判定

#### 前処理

- 前処理パイプラインは変更しないこと（ROI切り出し→グレースケール変換→pHash）

### 推奨要求 (SHOULD)

- 処理速度がリファクタリング前と同等水準であること（参考: F-001は平均0.09ms）
  - `shared/recognition/compute_phash` は 32x32リサイズ+DCT を行うため、多少の増加は許容する
- exp_001 README.md の結果セクションを更新し、リファクタリング後の精度・閾値・速度を記録すること

## 変更対象ファイル

| ファイル | 変更内容 |
|---|---|
| `experiments/exp_001_baito_text_recognition/main.py` | `BaitoTextRecognizer`: pHash実装をshared/recognitionに置換。`create_template()`: 同様に置換。デフォルトテンプレートパス変更。`--hash-size`引数の除去（16x16固定）。 |
| `experiments/exp_002_video_baito_text_recognition/main.py` | `DEFAULT_TEMPLATE_PATH` を新テンプレートパスに変更 |
| `experiments/exp_003_gui_recognition_viewer/plugins/baito_text.py` | `DEFAULT_TEMPLATE_PATH` を新テンプレートパスに変更 |
| `assets/templates/text/baito.npy` | 新規作成（256bitテンプレートハッシュ） |

## 受け入れ基準

- [ ] `BaitoTextRecognizer` が `shared.recognition.compute_phash` と `shared.recognition.hamming_distance` を使用していること
- [ ] `cv2.img_hash` への依存がexp_001から除去されていること
- [ ] テンプレートが `assets/templates/text/baito.npy` に配置されていること
- [ ] exp_002 が新テンプレートパスを参照していること
- [ ] exp_003 `BaitoTextPlugin` が新テンプレートパスを参照していること
- [ ] テストフィクスチャ全4枚で100%正解率であること（`--test-all data/test_fixtures/text/`）
- [ ] 閾値の根拠（各画像のハミング距離）が記録されていること

## 既存テストフィクスチャ

| ファイル | 種別 | 期待判定 | F-001(8x8)でのハミング距離 |
|---|---|---|---|
| `data/test_fixtures/text/positive/baito_001.png` | 正例 | テキストあり | 0 |
| `data/test_fixtures/text/positive/baito_002.png` | 正例 | テキストあり | 1 |
| `data/test_fixtures/text/negative/lobby_001.png` | 負例 | テキストなし | 31 |
| `data/test_fixtures/text/negative/wave1_active_001.png` | 負例 | テキストなし | 26 |

> 16x16 (256bit) 切り替え後のハミング距離は実装時に再計測し、閾値設定の根拠とする。

## 備考

- 本リファクタリングは認識ロジックの変更を伴わない（pHashアルゴリズムの差し替えのみ）
- ただし、OpenCV PHash (8x8, DCTサイズ不明) と shared/recognition (16x16, 32x32 DCT) はアルゴリズム詳細が異なるため、ハミング距離の分布が変わる。閾値の再チューニングが必須
- 管理番号プレフィックス「R」はリファクタリング用として本件で新設する
