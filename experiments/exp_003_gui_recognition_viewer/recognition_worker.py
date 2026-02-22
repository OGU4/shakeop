"""認識処理ワーカースレッド。

CaptureWorker から受け取った最新フレームに対して認識処理を行い、
オーバーレイ描画済みフレームと認識結果を Signal で送出する。
"""

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal, Slot

from fps_counter import FPSCounter
from recognition_plugin import RecognitionPlugin


class RecognitionWorker(QThread):
    """認識処理ワーカースレッド。"""

    frame_ready = Signal(np.ndarray, dict, float)  # (描画済みフレーム, 認識結果, fps)

    def __init__(self, plugin: RecognitionPlugin) -> None:
        super().__init__()
        self._plugin = plugin
        self._plugin_lock = QMutex()
        self._latest_frame: np.ndarray | None = None
        self._new_frame_available: bool = False
        self._frame_lock = QMutex()
        self._fps_counter = FPSCounter()
        self._running = False

    @Slot(np.ndarray)
    def on_frame_captured(self, frame: np.ndarray) -> None:
        """CaptureWorkerからのフレーム受信。最新フレームを上書き保持する。"""
        locker = QMutexLocker(self._frame_lock)  # noqa: F841
        self._latest_frame = frame
        self._new_frame_available = True

    def run(self) -> None:
        """メインループ: 最新フレーム取得 → 認識 → オーバーレイ描画 → Signal送出。"""
        self._running = True
        self._fps_counter.reset()
        while self._running:
            # 最新フレームを取得
            frame = self._get_latest_frame()
            if frame is None:
                self.msleep(10)
                continue

            # プラグインをスレッドセーフに取得
            locker = QMutexLocker(self._plugin_lock)  # noqa: F841
            plugin = self._plugin
            del locker

            # 認識処理
            result = plugin.process(frame)
            overlay = plugin.draw_overlay(frame, result)
            fps = self._fps_counter.tick()
            self.frame_ready.emit(overlay, result, fps)

    def set_plugin(self, plugin: RecognitionPlugin) -> None:
        """認識プラグインを動作中に差し替える。"""
        locker = QMutexLocker(self._plugin_lock)  # noqa: F841
        self._plugin = plugin
        self._fps_counter.reset()

    def stop(self) -> None:
        """外部からループ停止を要求する。"""
        self._running = False

    def _get_latest_frame(self) -> np.ndarray | None:
        """新着フレームがあれば取得する。なければNoneを返す。"""
        locker = QMutexLocker(self._frame_lock)  # noqa: F841
        if not self._new_frame_available:
            return None
        self._new_frame_available = False
        return self._latest_frame
