"""Wave数判定の認識プラグイン。

exp_004 の WaveNumberRecognizer をラップして RecognitionPlugin Protocol を満たす。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# exp_004 のディレクトリをパスに追加
_EXP_004_DIR = (
    Path(__file__).resolve().parent.parent.parent / "exp_004_wave_number_recognition"
)
sys.path.insert(0, str(_EXP_004_DIR))

from wave_recognizer import WaveNumberRecognizer  # noqa: E402

# プロジェクトルート: plugins/ → exp_003_gui_recognition_viewer/ → experiments/ → shakeop/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "assets" / "templates" / "wave"

# 必須テンプレートファイル一覧
_REQUIRED_TEMPLATES = [
    "wave_text.npy",
    "digit_1.npy",
    "digit_2.npy",
    "digit_3.npy",
    "digit_4.npy",
    "digit_5.npy",
]


class WaveNumberPlugin:
    """Wave数判定の認識プラグイン (RecognitionPlugin Protocol準拠)"""

    # ROI座標（WaveNumberRecognizer.WAVE_ROI と同値。描画用に保持）
    ROI = (76, 35, 199, 80)

    # オーバーレイ描画の色
    COLOR_DETECTED = (0, 255, 0)  # 緑: Wave検出時
    COLOR_NOT_DETECTED = (0, 0, 255)  # 赤: 未検出時

    # テキスト描画位置: ROI矩形の右横
    TEXT_OFFSET_X = 10  # ROI右端からの水平マージン (px)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2

    def __init__(
        self,
        template_dir: Path | None = None,
        threshold: int = 116,
    ) -> None:
        """
        Args:
            template_dir: テンプレートディレクトリ。Noneなら DEFAULT_TEMPLATE_DIR を使用。
            threshold: pHashハミング距離の閾値。ステップ1で最適化された値。
        """
        if template_dir is None:
            template_dir = DEFAULT_TEMPLATE_DIR

        # 全テンプレートの存在確認
        for filename in _REQUIRED_TEMPLATES:
            path = template_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"テンプレートが見つかりません: {path}")

        # テンプレート読み込み
        wave_text_hash = np.load(str(template_dir / "wave_text.npy"))
        digit_hashes = {
            i: np.load(str(template_dir / f"digit_{i}.npy")) for i in range(1, 6)
        }

        # 認識器を初期化
        self._recognizer = WaveNumberRecognizer(
            wave_text_hash=wave_text_hash,
            digit_hashes=digit_hashes,
            threshold=threshold,
        )

    @property
    def name(self) -> str:
        return "Wave数判定"

    def process(self, frame: np.ndarray) -> dict:
        """WaveNumberRecognizer.recognize_debug() を呼び出し、結果をdictで返す。

        WAVEテキスト判定と数字判定の内訳を個別に返す。
        """
        debug = self._recognizer.recognize_debug(frame)
        threshold = self._recognizer.threshold
        wave_text_dist = debug["wave_text_distance"]
        wave_text_ok = wave_text_dist <= threshold

        # 数字判定の内訳
        digit_distances = debug["digit_distances"]
        best_digit = min(digit_distances, key=digit_distances.get)
        best_digit_dist = digit_distances[best_digit]
        digit_ok = wave_text_ok and best_digit_dist <= threshold

        return {
            "detected": debug["result"] != "NONE",
            "confidence": debug["confidence"],
            "wave_result": debug["result"],
            # WAVEテキスト判定の内訳
            "wave_text_ok": wave_text_ok,
            "wave_text_dist": wave_text_dist,
            # 数字判定の内訳
            "digit_ok": digit_ok,
            "best_digit": best_digit,
            "best_digit_dist": best_digit_dist,
            "digit_distances": digit_distances,
            # 閾値
            "threshold": threshold,
        }

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """ROI矩形とWAVEテキスト/数字の判定結果を個別にオーバーレイ描画する。

        1行目: WAVEテキスト判定 (OK/NG + ハミング距離)
        2行目: 数字判定 (OK/NG + 数字 + ハミング距離)
        """
        overlay = frame.copy()
        x1, y1, x2, y2 = self.ROI
        detected = result["detected"]

        color = self.COLOR_DETECTED if detected else self.COLOR_NOT_DETECTED

        # ROI矩形
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        text_x = x2 + self.TEXT_OFFSET_X
        line_height = 22
        threshold = result["threshold"]

        # 1行目: WAVEテキスト判定
        wave_status = "OK" if result["wave_text_ok"] else "NG"
        wave_color = self.COLOR_DETECTED if result["wave_text_ok"] else self.COLOR_NOT_DETECTED
        label_wave = f"WAVE: {wave_status} dist={result['wave_text_dist']}/{threshold}"
        cv2.putText(
            overlay, label_wave, (text_x, y1 + line_height),
            self.TEXT_FONT, self.TEXT_SCALE, wave_color, self.TEXT_THICKNESS,
        )

        # 2行目: 数字判定
        digit_status = "OK" if result["digit_ok"] else "NG"
        digit_color = self.COLOR_DETECTED if result["digit_ok"] else self.COLOR_NOT_DETECTED
        label_digit = (
            f"Digit: {digit_status} "
            f"best={result['best_digit']}(dist={result['best_digit_dist']}/{threshold})"
        )
        cv2.putText(
            overlay, label_digit, (text_x, y1 + line_height * 2),
            self.TEXT_FONT, self.TEXT_SCALE, digit_color, self.TEXT_THICKNESS,
        )

        return overlay

    def format_log(self, result: dict) -> str:
        """ログ1行分をフォーマットする。"""
        wave_ok = "OK" if result["wave_text_ok"] else "NG"
        digit_ok = "OK" if result["digit_ok"] else "NG"
        return (
            f"result={result['wave_result']:<7}  "
            f"WAVE:{wave_ok}(d={result['wave_text_dist']})  "
            f"Digit:{digit_ok} best={result['best_digit']}(d={result['best_digit_dist']})"
        )
