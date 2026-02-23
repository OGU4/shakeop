"""
WorksOverRecognizer: "Work's Over!!" のpHash認識器（ROI全体1段判定）

設計書: docs/issues/F-006_works_over_recognition/design.md

認識フロー:
1. ROI切り出し (728, 872, 1048, 976) → 320x104 px
2. グレースケール変換
3. pHashでテンプレートとのハミング距離を算出 → "WORKS_OVER" or "NONE"
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.recognition import compute_phash, hamming_distance  # noqa: E402


class WorksOverRecognizer:
    """'Work's Over!!' のpHash認識器（ROI全体1段判定）"""

    # ROI座標 (FHD 1920x1080 基準)
    WORKS_OVER_ROI = (728, 872, 1048, 976)  # (x1, y1, x2, y2) 320x104 px

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        works_over_hash: np.ndarray,
        threshold: int = 50,
    ):
        """
        Args:
            works_over_hash: "Work's Over!!" テンプレートのpHashハッシュ
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        self.works_over_hash = works_over_hash
        self.threshold = threshold

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームから "Work's Over!!" の有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "WORKS_OVER", "NONE"
        """
        roi = self._extract_roi(frame)
        gray = self._preprocess(roi)
        phash = compute_phash(gray)
        distance = hamming_distance(phash, self.works_over_hash)
        confidence = 1.0 - (distance / self.MAX_DISTANCE)

        if distance <= self.threshold:
            return "WORKS_OVER", confidence
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
                "gray": np.ndarray,
            }
        """
        roi = self._extract_roi(frame)
        gray = self._preprocess(roi)
        phash = compute_phash(gray)
        distance = hamming_distance(phash, self.works_over_hash)
        confidence = 1.0 - (distance / self.MAX_DISTANCE)

        if distance <= self.threshold:
            result = "WORKS_OVER"
        else:
            result = "NONE"

        return {
            "result": result,
            "confidence": confidence,
            "distance": distance,
            "roi": roi,
            "gray": gray,
        }

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出す (320x104 BGR)"""
        x1, y1, x2, y2 = self.WORKS_OVER_ROI
        return frame[y1:y2, x1:x2]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """グレースケール変換"""
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
