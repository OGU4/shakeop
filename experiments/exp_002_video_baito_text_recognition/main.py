#!/usr/bin/env python3
"""
exp_002: 「バイトの時間です」動画テキスト認識

目的: F-001のpHash認識が動画フレームでも安定動作するか検証 + リアルタイム認識パイプラインの雛形
手法: exp_001の BaitoTextRecognizer を import し、カメラ入力に適用
実行: uv run python experiments/exp_002_video_baito_text_recognition/main.py --camera 10 --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# exp_001 のディレクトリをパスに追加
EXP_001_DIR = Path(__file__).resolve().parent.parent / "exp_001_baito_text_recognition"
sys.path.insert(0, str(EXP_001_DIR))

from main import BaitoTextRecognizer  # noqa: E402

# デフォルトパス
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEMPLATE_PATH = _PROJECT_ROOT / "assets" / "templates" / "text" / "baito.npy"
DEFAULT_ROI = (750, 545, 1170, 600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="exp_002: 「バイトの時間です」動画テキスト認識"
    )
    parser.add_argument(
        "--camera", type=int, default=10, help="カメラデバイス番号 (default: 10)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=str(DEFAULT_TEMPLATE_PATH),
        help="テンプレートハッシュパス (.npy)",
    )
    parser.add_argument(
        "--threshold", type=int, default=62, help="ハミング距離の閾値 (default: 62)"
    )
    parser.add_argument("--debug", action="store_true", help="OpenCVデバッグ表示ON")
    return parser.parse_args()


def draw_debug(frame: np.ndarray, detected: bool, confidence: float) -> np.ndarray:
    """デバッグ用の可視化フレームを生成する。"""
    debug_frame = frame.copy()
    x1, y1, x2, y2 = DEFAULT_ROI
    color = (0, 255, 0) if detected else (0, 0, 255)

    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

    label = f"{'DETECTED' if detected else 'NOT DETECTED'} conf={confidence:.4f}"
    cv2.putText(
        debug_frame,
        label,
        (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return debug_frame


def run_camera(args: argparse.Namespace) -> None:
    """カメラ入力でリアルタイム認識を行うメインループ。"""
    # テンプレート読み込み
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"Error: テンプレートが見つかりません: {template_path}")
        print("先に exp_001 で --create-template を実行してください")
        sys.exit(1)

    template_hash = np.load(str(template_path))
    recognizer = BaitoTextRecognizer(
        template_hash=template_hash,
        threshold=args.threshold,
    )

    # カメラオープン
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: カメラ {args.camera} を開けません")
        sys.exit(1)

    print(f"Camera {args.camera} opened. Press Ctrl+C to quit.")
    if args.debug:
        print("Debug mode ON. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected, confidence = recognizer.recognize(frame)
            print(f"detected={str(detected):<5}  confidence={confidence:.4f}")

            if args.debug:
                debug_frame = draw_debug(frame, detected, confidence)
                cv2.imshow("exp_002: video baito text", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_camera(args)
