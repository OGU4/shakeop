"""「バイトの時間です」テキスト認識プラグイン。

exp_001 の BaitoTextRecognizer をラップして RecognitionPlugin Protocol を満たす。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# exp_001 のディレクトリをパスに追加
_EXP_001_DIR = (
    Path(__file__).resolve().parent.parent.parent / "exp_001_baito_text_recognition"
)
sys.path.insert(0, str(_EXP_001_DIR))

from main import BaitoTextRecognizer  # noqa: E402

# FHD基準のROI座標 (x1, y1, x2, y2)
DEFAULT_ROI = (750, 545, 1170, 600)

# テンプレートハッシュのデフォルトパス
DEFAULT_TEMPLATE_PATH = _EXP_001_DIR / "template_hash8.npy"


class BaitoTextPlugin:
    """「バイトの時間です」テキスト認識プラグイン。"""

    ROI = DEFAULT_ROI

    def __init__(self, template_path: Path | None = None) -> None:
        if template_path is None:
            template_path = DEFAULT_TEMPLATE_PATH
        if not template_path.exists():
            raise FileNotFoundError(f"テンプレートが見つかりません: {template_path}")
        template_hash = np.load(str(template_path))
        self._recognizer = BaitoTextRecognizer(
            template_hash=template_hash,
            roi=self.ROI,
        )

    @property
    def name(self) -> str:
        return "バイトの時間です"

    def process(self, frame: np.ndarray) -> dict:
        """BaitoTextRecognizer.recognize() を呼び出し、結果をdictで返す。"""
        detected, confidence = self._recognizer.recognize(frame)
        return {"detected": detected, "confidence": confidence}

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """ROI矩形と判定テキストをオーバーレイ描画する。"""
        overlay = frame.copy()
        x1, y1, x2, y2 = self.ROI
        detected = result["detected"]
        confidence = result["confidence"]

        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{'DETECTED' if detected else 'NOT DETECTED'} conf={confidence:.4f}"
        cv2.putText(
            overlay,
            label,
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return overlay

    def format_log(self, result: dict) -> str:
        """ログ1行分をフォーマットする。"""
        return f"detected={str(result['detected']):<5}  confidence={result['confidence']:.4f}"
