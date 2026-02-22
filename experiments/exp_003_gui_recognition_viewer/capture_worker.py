"""カメラ取得ワーカースレッド。

指定されたデバイスから cv2.VideoCapture でフレームを連続取得し、
frame_captured Signal で送出する。
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CaptureWorker(QThread):
    """カメラ取得ワーカースレッド。"""

    frame_captured = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, device_path: str) -> None:
        super().__init__()
        self._device_path = device_path
        self._running = False

    def run(self) -> None:
        """メインループ: フレーム取得 → Signal送出。"""
        cap = cv2.VideoCapture(self._device_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"カメラを開けません: {self._device_path}")
            return

        self._running = True
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self.error_occurred.emit(
                        f"カメラからの読み取りに失敗しました: {self._device_path}"
                    )
                    break
                self.frame_captured.emit(frame)
        finally:
            cap.release()

    def stop(self) -> None:
        """外部からループ停止を要求する。"""
        self._running = False
