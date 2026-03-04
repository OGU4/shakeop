"""
ClearResultRecognizer: "Clear!!" のpHash認識器（ROI全体1段判定）

設計書: docs/issues/F-007_clear_result_recognition/design.md

認識フロー:
1. ROI切り出し (18, 12, 370, 115) → 352x103 px
2. グレースケール変換
3. pHashでテンプレートとのハミング距離を算出 → "CLEAR" or "NONE"
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.recognition import compute_phash, hamming_distance  # noqa: E402


class ClearResultRecognizer:
    """'Clear!!' のpHash認識器（ROI全体1段判定）"""

    # ROI座標 (FHD 1920x1080 基準)
    CLEAR_RESULT_ROI = (18, 12, 370, 115)  # (x1, y1, x2, y2) 352x103 px

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        clear_hash: np.ndarray,
        threshold: int = 50,
    ):
        """
        Args:
            clear_hash: "Clear!!" テンプレートのpHashハッシュ
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        self.clear_hash = clear_hash
        self.threshold = threshold

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームから "Clear!!" の有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "CLEAR", "NONE"
        """
        roi = self._extract_roi(frame)
        gray = self._preprocess(roi)
        phash = compute_phash(gray)
        distance = hamming_distance(phash, self.clear_hash)
        confidence = 1.0 - (distance / self.MAX_DISTANCE)

        if distance <= self.threshold:
            return "CLEAR", confidence
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
        distance = hamming_distance(phash, self.clear_hash)
        confidence = 1.0 - (distance / self.MAX_DISTANCE)

        if distance <= self.threshold:
            result = "CLEAR"
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
        """ROI領域を切り出す (352x103 BGR)"""
        x1, y1, x2, y2 = self.CLEAR_RESULT_ROI
        return frame[y1:y2, x1:x2]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """グレースケール変換"""
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
