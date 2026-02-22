"""FPS計算ユーティリティ（スライディングウィンドウ方式）。"""

import time
from collections import deque


class FPSCounter:
    """スライディングウィンドウ方式のFPS計算。"""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """1フレーム処理完了時に呼び出し、現在のFPSを返す。

        直近 window_size フレームの平均FPSを算出する。

        Returns:
            現在のFPS値。フレーム数が不足する場合は 0.0
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        """カウンタをリセットする。"""
        self._timestamps.clear()
