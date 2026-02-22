# [B-001] FPS過大表示（同一フレーム再処理） — 要求仕様書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | B-001 |
| 作成日 | 2026-02-22 |
| ステータス | ✅ 修正完了 |
| 関連機能 | F-003（GUI認識ビューワー） |
| 発生箇所 | `experiments/exp_003_gui_recognition_viewer/recognition_worker.py` |

## 現象

F-003 GUI認識ビューワーにて、カメラ入力が60fps固定であるにもかかわらず、
ステータスバーのFPS表示が300を超える値を示す。

## 原因分析

`RecognitionWorker.run()` のメインループに、新しいフレームが到着したかどうかを判別する仕組みがない。

1. `CaptureWorker` が60fpsでフレームを送出し、`RecognitionWorker._latest_frame` を上書きする
2. `RecognitionWorker.run()` のループは `_get_latest_frame()` で最新フレームを取得するが、
   **一度フレームが届くと `_latest_frame` は二度と `None` に戻らない**
3. pHash認識はサブミリ秒で完了するため、ループは待機なしで全力回転し、
   **同じフレームに対して毎秒300回以上の認識処理を実行する**
4. `FPSCounter.tick()` はループ1回ごとに呼ばれるため、300+fps と表示される

### 問題箇所（コード）

```python
# recognition_worker.py — 現在の実装
def _get_latest_frame(self) -> np.ndarray | None:
    """最新フレームを取得する。"""
    locker = QMutexLocker(self._frame_lock)
    return self._latest_frame  # ← 常に同じフレームを返し続ける
```

### 影響

- **FPS表示が不正確**: 実際のユニークフレーム処理FPSではなく、ループ回転数を表示している
- **CPU無駄遣い**: 同一フレームの再処理でCPUリソースを浪費する
- **ログ重複**: ログ有効時、同一フレームの認識結果が大量に重複出力される

## 修正要求

### 必須要求 (MUST)

- RecognitionWorker は **新しいフレームが到着したときだけ** 認識処理を行うこと
- 新フレームが未到着の場合は短時間待機し、ビジーウェイトを回避すること
- FPS表示は **ユニークフレーム（新着フレーム）の認識完了FPS** を反映すること
- ログ出力は **ユニークフレームごとに1行** であること（同一フレームの重複出力を排除）

### 修正方針

`_new_frame_available` ブール型フラグを導入する。

- `on_frame_captured()`: フレーム更新時にフラグを `True` にセット
- `_get_latest_frame()`: フラグが `True` の場合のみフレームを返し、フラグを `False` にリセット。フラグが `False` なら `None` を返す

この方式を採用する理由:
- `_latest_frame` を `None` にリセットする方式だと、取得と認識の間にフレームが失われるリスクがある
- フラグ方式なら「データ」と「新着通知」が分離され、レースコンディションに強い

### 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `recognition_worker.py` | `_new_frame_available` フラグ追加、`on_frame_captured()` と `_get_latest_frame()` の修正 |

### 修正不要のファイル

| ファイル | 理由 |
|---|---|
| `fps_counter.py` | FPSCounter自体のロジックは正常。呼び出しタイミングが修正されれば正しく動作する |
| `capture_worker.py` | フレーム取得側に問題なし |
| `main_window.py` | 表示側に問題なし |

## 受け入れ基準

- [x] 60fps入力に対してFPS表示が60以下（認識処理の実時間に応じた値）になること
- [x] 同一フレームが繰り返し認識処理されないこと
- [x] 新フレーム未到着時にCPUがビジーウェイトしないこと
- [x] ログ出力が新着フレームごとに1行であること（60fps入力なら毎秒最大60行）
