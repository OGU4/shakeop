"""Clear!!判定の認識プラグイン。

exp_007 の ClearResultRecognizer をラップして RecognitionPlugin Protocol を満たす。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# exp_007 のディレクトリをパスに追加
_EXP_007_DIR = (
    Path(__file__).resolve().parent.parent.parent / "exp_007_clear_result_recognition"
)
sys.path.insert(0, str(_EXP_007_DIR))

from clear_result_recognizer import ClearResultRecognizer  # noqa: E402

# プロジェクトルート: plugins/ → exp_003_gui_recognition_viewer/ → experiments/ → shakeop/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "assets" / "templates" / "clear_result"

# 必須テンプレートファイル
_REQUIRED_TEMPLATE = "clear_result.npy"


class ClearResultPlugin:
    """'Clear!!' 判定の認識プラグイン (RecognitionPlugin Protocol準拠)"""

    # ROI座標（ClearResultRecognizer.CLEAR_RESULT_ROI と同値。描画用に保持）
    ROI = (18, 12, 370, 115)

    # オーバーレイ描画の色
    COLOR_DETECTED = (0, 255, 0)  # 緑: Clear!!検出時
    COLOR_NOT_DETECTED = (0, 0, 255)  # 赤: 未検出時

    # テキスト描画位置: ROI矩形の右横
    TEXT_OFFSET_X = 10  # ROI右端からの水平マージン (px)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2

    def __init__(
        self,
        template_dir: Path | None = None,
        threshold: int = 50,
    ) -> None:
        """
        Args:
            template_dir: テンプレートディレクトリ。Noneなら DEFAULT_TEMPLATE_DIR を使用。
            threshold: pHashハミング距離の閾値。
        """
        if template_dir is None:
            template_dir = DEFAULT_TEMPLATE_DIR

        # テンプレートの存在確認
        template_path = template_dir / _REQUIRED_TEMPLATE
        if not template_path.exists():
            raise FileNotFoundError(f"テンプレートが見つかりません: {template_path}")

        # テンプレート読み込み & 認識器を初期化
        clear_hash = np.load(str(template_path))
        self._recognizer = ClearResultRecognizer(
            clear_hash=clear_hash,
            threshold=threshold,
        )

    @property
    def name(self) -> str:
        return "Clear!!判定"

    def process(self, frame: np.ndarray) -> dict:
        """ClearResultRecognizer.recognize_debug() を呼び出し、結果をdictで返す。"""
        debug = self._recognizer.recognize_debug(frame)
        threshold = self._recognizer.threshold

        return {
            "detected": debug["result"] == "CLEAR",
            "confidence": debug["confidence"],
            "clear_result": debug["result"],
            "distance": debug["distance"],
            "threshold": threshold,
        }

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """ROI矩形と判定結果テキストをオーバーレイ描画する。"""
        overlay = frame.copy()
        x1, y1, x2, y2 = self.ROI
        detected = result["detected"]

        color = self.COLOR_DETECTED if detected else self.COLOR_NOT_DETECTED

        # ROI矩形
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 判定結果テキスト
        text_x = x2 + self.TEXT_OFFSET_X
        line_height = 22

        status = "OK" if detected else "NG"
        label = f"CLEAR: {status} dist={result['distance']}/{result['threshold']}"
        cv2.putText(
            overlay,
            label,
            (text_x, y1 + line_height),
            self.TEXT_FONT,
            self.TEXT_SCALE,
            color,
            self.TEXT_THICKNESS,
        )

        return overlay

    def format_log(self, result: dict) -> str:
        """ログ1行分をフォーマットする。"""
        status = "OK" if result["detected"] else "NG"
        return (
            f"result={result['clear_result']:<12}  "
            f"CLEAR:{status}(d={result['distance']})"
        )
