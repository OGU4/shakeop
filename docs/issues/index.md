# 管理番号一覧

## 機能 (Feature)

| 番号 | 名称 | 状態 | 実験 | 統合先 | 備考 |
|---|---|---|---|---|---|
| F-001 | 「バイトの時間です」テキスト認識 | ✅ 完了 | exp_001 | recognition/text_identifier.py | ROI + pHash方式。精度100%, 0.09ms |
| F-002 | 「バイトの時間です」動画テキスト認識 | ✅ 完了 | exp_002 | - | F-001の動画入力拡張。カメラ(/dev/video10)からリアルタイム認識 |
| F-003 | GUI認識ビューワー | ✅ 完了 | exp_003 | - | F-002のGUI化。PySide6 GUIミニアプリ。B-001修正済み |
| F-004 | Wave数判定 | ✅ 完了 | exp_004 | - | ステップ1,2,3完了。shared/recognition/ にpHash共通関数を切り出し済み |
| F-005 | Extra Wave判定 | ✅ 完了 | exp_005 | - | ROI (38,35,238,80) 200x45px, pHash 1段判定。精度100%, 0.19ms, 閾値110 |
| F-006 | "Work's Over!!" テキスト認識 | ✅ 完了 | exp_006 | - | ゲームオーバー時の固定テキスト認識。ROI (728,872,1048,976) 320x104px, pHash 1段判定。精度100% (149/149), 0.20ms, 閾値50。GUI統合済み(WorksOverPlugin)。[要求仕様書](F-006_works_over_recognition/requirements.md) / [機能設計書](F-006_works_over_recognition/design.md) |

## リファクタリング (Refactoring)

| 番号 | 名称 | 状態 | 対象 | 備考 |
|---|---|---|---|---|
| R-001 | pHash共通化リファクタリング | ✅ 完了 | exp_001, exp_002, exp_003 | cv2.img_hash.PHash(8x8)→shared/recognition(16x16)に統一。精度100%, 0.22ms, 閾値62。[要求仕様書](R-001_phash_commonization/requirements.md) / [機能設計書](R-001_phash_commonization/design.md) |

## バグ修正 (Bug)

| 番号 | 名称 | 状態 | 原因 | 修正先 | 備考 |
|---|---|---|---|---|---|
| B-001 | FPS過大表示（同一フレーム再処理） | ✅ 完了 | RecognitionWorkerが新フレーム到着を検知せず同一フレームを繰り返し処理 | exp_003 recognition_worker.py | `_new_frame_available` フラグ導入で修正 |

### ステータス凡例
- 📋 未着手
- 📝 仕様作成中
- 🔬 実装中
- 🧪 テスト中
- ✅ 完了
- ❌ 取り下げ（理由をrequirements.mdに記載）
