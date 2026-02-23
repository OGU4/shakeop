"""
ExtraWaveRecognizer: Extra WaveのpHash認識器（ROI全体1段判定）

設計書: docs/issues/F-005_extra_wave_recognition/design.md

認識フロー:
1. ROI切り出し (38, 35, 238, 80) → 200x45 px
2. HSV白色テキスト抽出 → 二値画像
3. pHashでテンプレートとのハミング距離を算出 → "EXTRA" or "NONE"
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.recognition import compute_phash, hamming_distance  # noqa: E402


class ExtraWaveRecognizer:
    """Extra WaveのpHash認識器（ROI全体1段判定）"""

    # ROI座標 (FHD 1920x1080 基準)
    EXTRA_WAVE_ROI = (38, 35, 238, 80)  # (x1, y1, x2, y2) 200x45 px

    # HSVフィルタ（白色テキスト抽出） — F-004と同一パラメータ
    # V≥210: 霧イベントの背景ノイズ(V≈200-210)を除外しつつテキストを残す
    HSV_LOWER = np.array([0, 0, 210])
    HSV_UPPER = np.array([180, 80, 255])

    # モルフォロジー演算カーネル（ノイズ除去用）
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        extra_wave_hash: np.ndarray,
        threshold: int = 110,
    ):
        """
        Args:
            extra_wave_hash: 「EXTRA WAVE」テンプレートのpHashハッシュ
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        self.extra_wave_hash = extra_wave_hash
        self.threshold = threshold

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームからExtra Waveの有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "EXTRA", "NONE"
        """
        roi = self._extract_roi(frame)
        binary = self._preprocess(roi)
        phash = compute_phash(binary)
        distance = hamming_distance(phash, self.extra_wave_hash)
        confidence = 1.0 - (distance / self.MAX_DISTANCE)

        if distance <= self.threshold:
            return "EXTRA", confidence
        else:
            return "NONE", confidence

    def recognize_debug(self, frame: np.ndarray) -> dict:
        """
        デバッグ用の詳細情報付き認識。

        Returns:
            {
                "result": str,
                "confidence": float,
                "distance": int,
                "roi": np.ndarray,
                "binary": np.ndarray,
            }
        """
        roi = self._extract_roi(frame)
        binary = self._preprocess(roi)
        phash = compute_phash(binary)
        distance = hamming_distance(phash, self.extra_wave_hash)
        confidence = 1.0 - (distance / self.MAX_DISTANCE)

        if distance <= self.threshold:
            result = "EXTRA"
        else:
            result = "NONE"

        return {
            "result": result,
            "confidence": confidence,
            "distance": distance,
            "roi": roi,
            "binary": binary,
        }

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出す (200x45 BGR)"""
        x1, y1, x2, y2 = self.EXTRA_WAVE_ROI
        return frame[y1:y2, x1:x2]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """HSVフィルタで白色テキストを抽出し、モルフォロジー演算でノイズ除去した二値画像を返す"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.HSV_LOWER, self.HSV_UPPER)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.MORPH_KERNEL)
        return cleaned
