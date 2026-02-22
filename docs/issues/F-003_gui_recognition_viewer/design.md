# [F-003] GUI認識ビューワー — 機能設計書

## 基本情報

| 項目 | 内容 |
|---|---|
| 管理番号 | F-003 |
| 要求仕様書 | [requirements.md](./requirements.md) |
| 作成日 | 2026-02-22 |
| ステータス | ✅ 完了 |

## 設計方針

PySide6でGUIミニアプリを構築する。
スレッドモデルは **カメラ取得スレッド** と **認識処理スレッド** を分離し、GUIスレッドをブロックしない設計とする。
認識処理は「プラグイン」として Protocol ベースで差し替え可能にし、手動リスト登録方式で管理する。

## 採用手法

| 候補 | 採否 | 理由 |
|---|---|---|
| PySide6 + QThread (取得・認識分離) | ✅ 採用 | プロジェクト標準のGUIフレームワーク。取得と認識を分離することでカメラFPSと認識FPSを独立させられる |
| PySide6 + QThread (単一ワーカー) | ❌ 不採用 | 認識が重くなった場合にフレーム取得が詰まる |
| OpenCVウィンドウ (cv2.imshow) | ❌ 不採用 | GUIの操作性・拡張性が不足。F-002のdebugモードで検証済み |
| 手動プラグインリスト登録 | ✅ 採用 | シンプルで明確。現時点でプラグイン数が少ないため十分 |
| plugins/ 自動検出 | ❌ 不採用 | 時期尚早。プラグイン数が増えてから検討 |

## モジュール構成

```
experiments/exp_003_gui_recognition_viewer/
├── main.py                 # エントリーポイント（QApplication起動 + プラグイン登録）
├── main_window.py          # メインウィンドウ（UI構成 + Signal/Slot接続）
├── capture_worker.py       # カメラ取得ワーカー（QThread）
├── recognition_worker.py   # 認識処理ワーカー（QThread）
├── recognition_plugin.py   # 認識プラグインのProtocol定義
├── fps_counter.py          # FPS計算ユーティリティ
├── log_writer.py           # テキストログ出力
├── plugins/                # 認識プラグイン実装
│   ├── __init__.py
│   └── baito_text.py       # 「バイトの時間です」認識プラグイン
├── README.md               # 実験結果記録
└── notes/                  # スクリーンショット等
```

## スレッドモデル

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CaptureWorker (QThread)         RecognitionWorker (QThread)│
│  ┌──────────────────────┐       ┌──────────────────────┐   │
│  │ loop:                │       │ on new_frame:        │   │
│  │   frame = cap.read() │──Signal──▶ latest_frame 更新  │   │
│  │   emit frame_captured│       │                      │   │
│  │     (frame)          │       │ loop:                │   │
│  └──────────────────────┘       │   frame = latest取得  │   │
│                                 │   result = plugin     │   │
│                                 │     .process(frame)   │   │
│                                 │   overlay = plugin    │   │
│                                 │     .draw_overlay(    │   │
│                                 │       frame, result)  │   │
│                                 │   fps = counter.tick()│   │
│                                 │   emit frame_ready    │   │
│                                 │     (overlay, result, │   │
│                                 │      fps)       ──Signal──┐│
│                                 └──────────────────────┘  ││
│                                                            ││
│  MainWindow (GUI Thread)                                   ││
│  ┌──────────────────────────────────────────────────────┐  ││
│  │ on frame_ready(overlay, result, fps):         ◀──────┘│
│  │   display overlay on QLabel                           │
│  │   update FPS label                                    │
│  │   if log_enabled: log_writer.write(result)            │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### フレーム同期方式

認識スレッドは常に **最新の1フレームのみ** を処理し、**新しいフレームが到着したときだけ** 認識を実行する。

- CaptureWorker がフレームを取得するたびに、RecognitionWorker 内の `_latest_frame` を上書きし、`_new_frame_available` フラグを `True` にセットする
- RecognitionWorker は `_new_frame_available` が `True` の場合のみフレームを消費して認識を開始する（フラグを `False` にリセット）
- `_new_frame_available` が `False` の場合は短時間 sleep してリトライする（ビジーウェイト回避）
- カメラが30fpsで認識が10fpsの場合、間のフレームは破棄される（リアルタイム性を優先）
- GUI映像表示は **認識完了ごと** に更新する（オーバーレイ付きフレームのみ表示）

> **B-001修正**: 初期設計では `_new_frame_available` フラグがなく、`_get_latest_frame()` が常に同じフレームを返し続けていた。
> これにより認識ループが全力回転し、FPSが300+と過大表示される不具合が発生した。
> フラグ方式を導入することで、ユニークフレームのみ認識処理する設計に修正。

```python
# RecognitionWorker 内のフレーム保持（スレッドセーフ）
class RecognitionWorker(QThread):
    def __init__(self, ...):
        self._latest_frame: np.ndarray | None = None
        self._new_frame_available: bool = False
        self._frame_lock = QMutex()

    @Slot(np.ndarray)
    def on_frame_captured(self, frame: np.ndarray) -> None:
        """CaptureWorkerからのフレームを受け取り、最新フレームを更新する。"""
        with QMutexLocker(self._frame_lock):
            self._latest_frame = frame
            self._new_frame_available = True

    def _get_latest_frame(self) -> np.ndarray | None:
        """新着フレームがあれば取得する。なければNoneを返す。

        フラグをFalseにリセットすることで、同一フレームの再処理を防ぐ。
        """
        with QMutexLocker(self._frame_lock):
            if not self._new_frame_available:
                return None
            self._new_frame_available = False
            return self._latest_frame
```

## 認識プラグインProtocol

```python
# recognition_plugin.py
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class RecognitionPlugin(Protocol):
    """認識プラグインのインターフェース。

    各プラグインはこのProtocolを満たすクラスとして実装する。
    """

    @property
    def name(self) -> str:
        """GUI表示用のプラグイン名。

        Returns:
            プラグイン名（例: "バイトの時間です"）
        """
        ...

    def process(self, frame: np.ndarray) -> dict:
        """認識処理を実行する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            認識結果の辞書。共通キー:
                "detected" (bool): 検出されたか
                "confidence" (float): 信頼度 0.0-1.0
            プラグイン固有のキーを追加してもよい。
        """
        ...

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """認識結果をフレーム上にオーバーレイ描画する。

        Args:
            frame: BGR画像（元フレーム。書き換えず、コピーを返すこと）
            result: process() の戻り値
        Returns:
            描画済みフレーム（元フレームのコピー）
        """
        ...

    def format_log(self, result: dict) -> str:
        """認識結果をログ出力用文字列にフォーマットする。

        Args:
            result: process() の戻り値
        Returns:
            ログ1行分の文字列（タイムスタンプは呼び出し側で付与）
        """
        ...
```

## クラス設計

### main.py

```python
"""exp_003: GUI認識ビューワー

エントリーポイント。QApplicationの起動とプラグインの登録を行う。
"""
import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from plugins.baito_text import BaitoTextPlugin

AVAILABLE_PLUGINS = [
    BaitoTextPlugin(),
    # 将来追加:
    # WaveTextPlugin(),
    # QuotaDigitPlugin(),
]

def main():
    app = QApplication(sys.argv)
    window = MainWindow(plugins=AVAILABLE_PLUGINS)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

### MainWindow

```python
class MainWindow(QMainWindow):
    """メインウィンドウ。

    UI構成:
        - 上部: コントロールパネル
            - カメラ選択 (QComboBox + 更新QPushButton)
            - 認識機能選択 (QComboBox)
            - 開始/停止トグル (QPushButton)
            - ログ出力設定 (QCheckBox + QLineEdit + 参照QPushButton)
        - 中央: 映像表示 (QLabel, スケーリング対応)
        - 下部: ステータスバー (FPS表示)
    """

    def __init__(self, plugins: list[RecognitionPlugin]):
        ...

    # --- カメラデバイス列挙 ---
    def _enumerate_cameras(self) -> list[str]:
        """'/dev/video*' をglob して返す。"""
        ...

    def _on_refresh_cameras(self) -> None:
        """更新ボタン押下時。カメラリストを再取得してドロップダウンを更新する。"""
        ...

    # --- 開始/停止 ---
    def _on_toggle_start_stop(self) -> None:
        """トグルボタン押下時。停止中なら開始、動作中なら停止する。"""
        ...

    def _start_capture(self) -> None:
        """CaptureWorker と RecognitionWorker を起動する。"""
        ...

    def _stop_capture(self) -> None:
        """CaptureWorker と RecognitionWorker を停止する。"""
        ...

    # --- 動作中のカメラ切替 ---
    def _on_camera_changed(self, index: int) -> None:
        """カメラドロップダウン変更時。動作中なら停止→デバイス変更→再開。"""
        ...

    # --- 動作中の認識機能切替 ---
    def _on_plugin_changed(self, index: int) -> None:
        """認識機能ドロップダウン変更時。RecognitionWorkerのプラグインを差し替える。"""
        ...

    # --- フレーム受信 ---
    @Slot(np.ndarray, dict, float)
    def _on_frame_ready(self, overlay_frame: np.ndarray, result: dict, fps: float) -> None:
        """RecognitionWorkerから認識済みフレームを受信。

        1. overlay_frame を QPixmap に変換して QLabel に表示
        2. FPS ラベルを更新
        3. ログ有効時に LogWriter.write() を呼び出す
        """
        ...

    # --- フレーム変換 ---
    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """BGR numpy配列 → QPixmap に変換。QLabel のサイズに合わせてスケーリング。"""
        ...

    # --- ログ設定 ---
    def _on_log_toggled(self, enabled: bool) -> None:
        """ログ出力チェックボックス変更時。LogWriterを開く/閉じる。"""
        ...

    def _on_browse_log_path(self) -> None:
        """参照ボタン押下時。QFileDialogで保存先を選択する。"""
        ...

    # --- エラー表示 ---
    def _show_error(self, message: str) -> None:
        """GUI上にエラーメッセージを表示する（QMessageBox）。"""
        ...
```

### CaptureWorker

```python
class CaptureWorker(QThread):
    """カメラ取得ワーカースレッド。

    指定されたデバイスから cv2.VideoCapture でフレームを連続取得し、
    frame_captured Signal で送出する。
    """

    frame_captured = Signal(np.ndarray)  # BGR画像
    error_occurred = Signal(str)         # エラーメッセージ

    def __init__(self, device_path: str):
        """
        Args:
            device_path: カメラデバイスパス（例: "/dev/video10"）
        """
        ...

    def run(self) -> None:
        """メインループ: フレーム取得 → Signal送出。

        cap.read() が False を返したらループ終了。
        カメラが開けない場合は error_occurred を送出して終了。
        """
        ...

    def stop(self) -> None:
        """外部からループ停止を要求する。"""
        self._running = False
```

### RecognitionWorker

```python
class RecognitionWorker(QThread):
    """認識処理ワーカースレッド。

    CaptureWorker から受け取った最新フレームに対して認識処理を行い、
    オーバーレイ描画済みフレームと認識結果を Signal で送出する。

    新しいフレームが到着したときだけ認識を実行する（B-001修正）。
    """

    frame_ready = Signal(np.ndarray, dict, float)  # (描画済みフレーム, 認識結果, fps)

    def __init__(self, plugin: RecognitionPlugin):
        """
        Args:
            plugin: 使用する認識プラグイン
        """
        ...
        self._latest_frame: np.ndarray | None = None
        self._new_frame_available: bool = False  # B-001修正: 新フレーム到着フラグ
        self._frame_lock = QMutex()
        self._fps_counter = FPSCounter()

    @Slot(np.ndarray)
    def on_frame_captured(self, frame: np.ndarray) -> None:
        """CaptureWorkerからのフレーム受信。最新フレームを上書き保持する。

        QMutex でスレッドセーフに最新フレームを更新し、
        _new_frame_available フラグを True にセットする。
        """
        ...

    def run(self) -> None:
        """メインループ:

        1. 新着フレームを取得（なければ短時間待機してリトライ）
        2. plugin.process(frame) で認識
        3. plugin.draw_overlay(frame, result) でオーバーレイ描画
        4. FPS計算
        5. frame_ready Signal 送出

        B-001修正: _new_frame_available が False の場合は msleep して
        ビジーウェイトを回避する。同一フレームの再処理は行わない。
        """
        ...

    def _get_latest_frame(self) -> np.ndarray | None:
        """新着フレームがあれば取得する。なければNoneを返す。

        B-001修正: _new_frame_available フラグを確認し、
        True の場合のみフレームを返してフラグを False にリセットする。
        """
        ...

    def set_plugin(self, plugin: RecognitionPlugin) -> None:
        """認識プラグインを動作中に差し替える。

        QMutex でスレッドセーフにプラグインを入れ替える。
        """
        ...

    def stop(self) -> None:
        """外部からループ停止を要求する。"""
        self._running = False
```

### BaitoTextPlugin

```python
# plugins/baito_text.py
class BaitoTextPlugin:
    """「バイトの時間です」テキスト認識プラグイン。

    exp_001 の BaitoTextRecognizer をラップして RecognitionPlugin Protocol を満たす。
    """

    def __init__(self):
        # exp_001 のテンプレートハッシュを読み込み
        # BaitoTextRecognizer を初期化
        ...

    @property
    def name(self) -> str:
        return "バイトの時間です"

    def process(self, frame: np.ndarray) -> dict:
        """BaitoTextRecognizer.recognize() を呼び出し、結果をdictで返す。

        Returns:
            {"detected": bool, "confidence": float}
        """
        detected, confidence = self._recognizer.recognize(frame)
        return {"detected": detected, "confidence": confidence}

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """F-002と同等のオーバーレイ描画。

        - ROI矩形（検出時: 緑、未検出時: 赤）
        - 判定テキスト + 信頼度
        """
        overlay = frame.copy()
        x1, y1, x2, y2 = self.ROI
        detected = result["detected"]
        confidence = result["confidence"]

        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{'DETECTED' if detected else 'NOT DETECTED'} conf={confidence:.4f}"
        cv2.putText(overlay, label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return overlay

    def format_log(self, result: dict) -> str:
        """ログ1行分をフォーマットする。

        Returns:
            "detected=True   confidence=0.9844" のような文字列
        """
        return f"detected={str(result['detected']):<5}  confidence={result['confidence']:.4f}"
```

### FPSCounter

```python
# fps_counter.py
import time


class FPSCounter:
    """スライディングウィンドウ方式のFPS計算。"""

    def __init__(self, window_size: int = 30):
        self._timestamps: list[float] = []
        self._window_size = window_size

    def tick(self) -> float:
        """1フレーム処理完了時に呼び出し、現在のFPSを返す。

        直近 window_size フレームの平均FPSを算出する。

        Returns:
            現在のFPS値。フレーム数が不足する場合は 0.0
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        while len(self._timestamps) > self._window_size:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        """カウンタをリセットする。"""
        self._timestamps.clear()
```

### LogWriter

```python
# log_writer.py
import datetime
from pathlib import Path
from recognition_plugin import RecognitionPlugin


class LogWriter:
    """認識結果をテキストファイルに出力する。"""

    def __init__(self, file_path: str | Path):
        """
        Args:
            file_path: ログファイルパス
        """
        self._file_path = Path(file_path)
        self._file = None

    def open(self) -> None:
        """ログファイルを追記モードで開く。"""
        self._file = open(self._file_path, "a", encoding="utf-8")

    def write(self, plugin: RecognitionPlugin, result: dict) -> None:
        """タイムスタンプ付きで認識結果を1行出力する。

        形式: "2026-02-22 15:30:01.123  detected=True   confidence=0.9844"
        """
        if self._file is None:
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = plugin.format_log(result)
        self._file.write(f"{timestamp}  {log_line}\n")
        self._file.flush()

    def close(self) -> None:
        """ログファイルを閉じる。"""
        if self._file is not None:
            self._file.close()
            self._file = None
```

## データフロー（詳細）

```
[ユーザー操作]
  │
  ├── カメラ選択 → MainWindow._on_camera_changed()
  │     → 動作中: stop_capture() → デバイス変更 → start_capture()
  │     → 停止中: デバイス変更のみ
  │
  ├── 認識機能選択 → MainWindow._on_plugin_changed()
  │     → RecognitionWorker.set_plugin(new_plugin)  ※動作中でも即切替
  │
  ├── トグルボタン → MainWindow._on_toggle_start_stop()
  │     → 停止中: start_capture()  ボタン表示を「■ 停止」に
  │     → 動作中: stop_capture()   ボタン表示を「▶ 開始」に
  │
  ├── ログ有効/無効 → MainWindow._on_log_toggled()
  │     → 有効: LogWriter.open()
  │     → 無効: LogWriter.close()
  │
  └── ログ保存先参照 → MainWindow._on_browse_log_path()
        → QFileDialog → パス更新

[データフロー（動作中）]

  CaptureWorker                 RecognitionWorker               MainWindow
  ─────────────                 ─────────────────               ──────────
  cap.read()
    │
    ▼
  frame_captured ──Signal──▶ on_frame_captured()
                             latest_frame = frame
                               │
                               ▼
                             _get_latest_frame()
                               │
                               ▼
                             plugin.process(frame)
                               │
                               ▼
                             plugin.draw_overlay(frame, result)
                               │
                               ▼
                             fps_counter.tick()
                               │
                               ▼
                             frame_ready ──Signal──▶ _on_frame_ready()
                                                       │
                                                       ├── QLabel に表示
                                                       ├── FPS ラベル更新
                                                       └── (ログ有効時)
                                                           log_writer.write()
```

## カメラデバイス列挙

```python
import glob

def enumerate_cameras() -> list[str]:
    """'/dev/video*' をglob して、パス名のソート済みリストを返す。

    Returns:
        ["/dev/video0", "/dev/video10", "/dev/video11", ...] 等
    """
    devices = sorted(glob.glob("/dev/video*"))
    return devices
```

## フレーム変換（BGR → QPixmap）

```python
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

def frame_to_pixmap(frame: np.ndarray, target_size: tuple[int, int]) -> QPixmap:
    """OpenCV BGR画像 → QPixmap に変換し、target_sizeにスケーリングする。

    Args:
        frame: BGR画像 (numpy配列)
        target_size: (width, height) 表示先のサイズ
    Returns:
        スケーリング済みの QPixmap
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(
        target_size[0], target_size[1],
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
```

## 動作中のカメラ切替シーケンス

```
1. ユーザーがカメラドロップダウンを変更
2. MainWindow._on_camera_changed() が呼ばれる
3. 動作中の場合:
   a. CaptureWorker.stop() → wait()
   b. RecognitionWorker.stop() → wait()
   c. 新しいデバイスパスで CaptureWorker を再生成
   d. RecognitionWorker を再生成（プラグインは現在のものを引き継ぐ）
   e. Signal/Slot を再接続
   f. 両ワーカーを start()
4. 停止中の場合:
   a. 選択デバイスを記憶するだけ（次回開始時に使用）
```

## 動作中の認識プラグイン切替

```
1. ユーザーが認識機能ドロップダウンを変更
2. MainWindow._on_plugin_changed() が呼ばれる
3. RecognitionWorker.set_plugin(new_plugin) を呼び出す
   → QMutex でスレッドセーフにプラグインを入れ替え
4. 次の認識ループから新プラグインが使用される
   ※ CaptureWorker は影響を受けない（停止・再起動不要）
```

## エラー処理

| 状況 | 処理 |
|---|---|
| カメラが開けない | CaptureWorker が error_occurred Signal を送出 → MainWindow が QMessageBox で表示 |
| カメラ切断（cap.read() 失敗） | CaptureWorker がループ終了 → error_occurred Signal → MainWindow がメッセージ表示 + 停止状態に戻す |
| テンプレートファイル未発見 | BaitoTextPlugin 初期化時に例外 → main.py でキャッチしてエラーメッセージ表示 |

## GUIレイアウト（詳細）

```
┌──────────────────────────────────────────────────────────┐
│ GUI認識ビューワー                                  [_][□][×] │
├──────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────┐ │
│ │ カメラ:  [/dev/video10          ▼] [更新]            │ │
│ │ 認識機能: [バイトの時間です       ▼]                   │ │
│ │ [▶ 開始]                                            │ │
│ │ ☐ ログ出力  [experiments/.../output.log    ] [参照..] │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────┐ │
│ │                                                      │ │
│ │                                                      │ │
│ │              映像表示エリア (QLabel)                   │ │
│ │        ウィンドウサイズに合わせてスケーリング            │ │
│ │           アスペクト比 16:9 維持                       │ │
│ │                                                      │ │
│ │                                                      │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ FPS: 30.0                                                │
└──────────────────────────────────────────────────────────┘
```

### ウィジェット一覧

| ウィジェット | 型 | 用途 |
|---|---|---|
| camera_combo | QComboBox | カメラデバイス選択 |
| refresh_btn | QPushButton | カメラリスト更新 |
| plugin_combo | QComboBox | 認識機能選択 |
| toggle_btn | QPushButton | 開始/停止トグル |
| log_checkbox | QCheckBox | ログ出力有効/無効 |
| log_path_edit | QLineEdit | ログファイルパス表示・編集 |
| browse_btn | QPushButton | ログ保存先選択 |
| video_label | QLabel | 映像表示 |
| fps_label | QLabel | FPS表示（ステータスバー内） |

### トグルボタンの状態遷移

| 状態 | ボタンテキスト | 押下時の動作 |
|---|---|---|
| 停止中 | ▶ 開始 | CaptureWorker + RecognitionWorker を起動 |
| 動作中 | ■ 停止 | 両ワーカーを停止 |

## 依存関係

- 前提: F-001（BaitoTextRecognizer + テンプレートハッシュ）、F-002（動画認識の検証結果）
- 使用ライブラリ:
  - PySide6 — GUI構築、QThread、Signal/Slot、QMutex
  - OpenCV — カメラ入力 (`cv2.VideoCapture`)、画像変換 (`cv2.cvtColor`)、描画 (`cv2.rectangle`, `cv2.putText`)
  - NumPy — フレームデータ操作
- exp_001 からの import:
  - `BaitoTextRecognizer` クラス
  - `template_hash8.npy` テンプレートハッシュファイル

## テスト計画

- 手動テスト: 仮想カメラに動画を流してGUI上で動作確認
  - カメラ選択・切替
  - 認識プラグイン切替
  - ログ出力の有効/無効・保存先変更
  - FPS表示
  - エラー時のメッセージ表示
- 認識ロジック自体のテストは F-001 のテストでカバー済み

## 備考

- ミニアプリのためOpenCVの HighGUI (`cv2.imshow`) は使わず、全てPySide6で表示する
- `RecognitionPlugin` Protocol はミニアプリ内で定義する（`shared/` には移動しない）
- 統合時にプラグインプロトコルを `shared/salmon_types/protocols.py` に移動する可能性あり
- カメラデバイスの列挙は `/dev/video*` の glob で行う（Linux前提）
- `RecognitionWorker` の認識ループは新着フレームがない場合（`_new_frame_available == False`）は短時間 sleep してリトライする（ビジーウェイト回避。B-001修正）
- `LogWriter.write()` は毎行 `flush()` する（クラッシュ時のデータロス防止）
