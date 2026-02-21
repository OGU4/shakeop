#!/usr/bin/env python3
"""
exp_001: 「バイトの時間です」テキスト認識 (pHash)

目的: ROI切り出し + pHash で固定テキストの有無を判定できるか検証
手法: OpenCV cv2.img_hash.PHash
実行: uv run python experiments/exp_001_baito_text_recognition/main.py --image <path> --debug
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# FHD基準のROI座標 (x1, y1, x2, y2)
DEFAULT_ROI = (750, 545, 1170, 600)


class BaitoTextRecognizer:
    """「バイトの時間です」テキストのpHash認識器"""

    def __init__(
        self,
        template_hash: np.ndarray,
        roi: tuple[int, int, int, int] = DEFAULT_ROI,
        threshold: int = 10,
        hash_size: int = 8,
    ):
        self.template_hash = template_hash
        self.roi = roi
        self.threshold = threshold
        self.hash_size = hash_size
        self._hasher = cv2.img_hash.PHash.create()

    def recognize(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        フレームから「バイトの時間です」テキストの有無を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (テキストの有無, 信頼度 0.0-1.0)
        """
        roi_gray = self._extract_roi(frame)
        input_hash = self._compute_phash(roi_gray)
        distance = self._hamming_distance(input_hash, self.template_hash)
        max_distance = self.hash_size * self.hash_size  # 8x8=64, 16x16=256
        confidence = 1.0 - (distance / max_distance)
        detected = distance <= self.threshold
        return detected, confidence

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出してグレースケール変換する"""
        x1, y1, x2, y2 = self.roi
        roi = frame[y1:y2, x1:x2]
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def _compute_phash(self, image: np.ndarray) -> np.ndarray:
        """pHashを計算する"""
        return self._hasher.compute(image)

    def _hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> int:
        """2つのハッシュ間のハミング距離を算出する"""
        return self._hasher.compare(hash1, hash2)


def create_template(image_path: str, roi: tuple[int, int, int, int], hash_size: int) -> np.ndarray:
    """テスト画像からテンプレートハッシュを作成する"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    x1, y1, x2, y2 = roi
    roi_img = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    hasher = cv2.img_hash.PHash.create()
    template_hash = hasher.compute(gray)
    return template_hash


def run_single_image(args, template_hash: np.ndarray):
    """単一画像でテスト"""
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: 画像を読み込めません: {args.image}")
        return

    recognizer = BaitoTextRecognizer(
        template_hash=template_hash,
        threshold=args.threshold,
        hash_size=args.hash_size,
    )

    start = time.perf_counter()
    detected, confidence = recognizer.recognize(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"画像: {args.image}")
    print(f"結果: {'テキストあり' if detected else 'テキストなし'}")
    print(f"信頼度: {confidence:.4f}")
    print(f"処理時間: {elapsed_ms:.2f}ms")

    if args.debug:
        x1, y1, x2, y2 = DEFAULT_ROI
        debug_frame = frame.copy()
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{'DETECTED' if detected else 'NOT DETECTED'} conf={confidence:.3f} {elapsed_ms:.1f}ms"
        cv2.putText(debug_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("exp_001: baito text recognition", debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_test_all(args, template_hash: np.ndarray):
    """テストディレクトリ内の全画像をテスト"""
    test_dir = Path(args.test_all)
    positive_dir = test_dir / "positive"
    negative_dir = test_dir / "negative"

    recognizer = BaitoTextRecognizer(
        template_hash=template_hash,
        threshold=args.threshold,
        hash_size=args.hash_size,
    )

    results = []
    total_time = 0.0

    # 正例テスト
    if positive_dir.exists():
        for img_path in sorted(positive_dir.glob("*.png")):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            start = time.perf_counter()
            detected, confidence = recognizer.recognize(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_time += elapsed_ms
            correct = detected is True
            results.append(("positive", img_path.name, detected, confidence, elapsed_ms, correct))

    # 負例テスト
    if negative_dir.exists():
        for img_path in sorted(negative_dir.glob("*.png")):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            start = time.perf_counter()
            detected, confidence = recognizer.recognize(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_time += elapsed_ms
            correct = detected is False
            results.append(("negative", img_path.name, detected, confidence, elapsed_ms, correct))

    # 結果表示
    print(f"\n=== テスト結果 (hash_size={args.hash_size}, threshold={args.threshold}) ===\n")
    print(f"{'種別':<10} {'ファイル':<30} {'判定':<12} {'信頼度':<10} {'時間(ms)':<10} {'正否'}")
    print("-" * 85)

    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    for category, name, detected, confidence, elapsed_ms, correct in results:
        status = "OK" if correct else "NG"
        det_str = "テキストあり" if detected else "テキストなし"
        print(f"{category:<10} {name:<30} {det_str:<12} {confidence:<10.4f} {elapsed_ms:<10.2f} {status}")
        if category == "positive":
            positive_total += 1
            if correct:
                positive_correct += 1
        else:
            negative_total += 1
            if correct:
                negative_correct += 1

    total = len(results)
    total_correct = positive_correct + negative_correct
    print("-" * 85)
    if positive_total > 0:
        print(f"正例正解率: {positive_correct}/{positive_total} ({positive_correct / positive_total:.0%})")
    if negative_total > 0:
        print(f"負例正解率: {negative_correct}/{negative_total} ({negative_correct / negative_total:.0%})")
    if total > 0:
        print(f"総合正解率: {total_correct}/{total} ({total_correct / total:.0%})")
        print(f"平均処理時間: {total_time / total:.2f}ms")


def parse_args():
    parser = argparse.ArgumentParser(description="exp_001: 「バイトの時間です」テキスト認識 (pHash)")
    parser.add_argument("--image", type=str, default=None, help="テスト画像パス")
    parser.add_argument("--create-template", action="store_true", help="テンプレートハッシュを作成して保存")
    parser.add_argument("--test-all", type=str, default=None, help="テストディレクトリ（positive/negative配下を全テスト）")
    parser.add_argument("--template", type=str, default=None, help="テンプレートファイルパス (.npy)")
    parser.add_argument("--hash-size", type=int, default=8, choices=[8, 16], help="pHashのハッシュサイズ (default: 8)")
    parser.add_argument("--threshold", type=int, default=10, help="ハミング距離の閾値 (default: 10)")
    parser.add_argument("--debug", action="store_true", help="デバッグ表示ON")
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = Path(__file__).parent
    default_template_path = exp_dir / f"template_hash{args.hash_size}.npy"
    template_path = Path(args.template) if args.template else default_template_path

    # テンプレート作成モード
    if args.create_template:
        if not args.image:
            print("Error: --create-template には --image が必要です")
            return
        template_hash = create_template(args.image, DEFAULT_ROI, args.hash_size)
        np.save(str(template_path), template_hash)
        print(f"テンプレート保存: {template_path}")
        print(f"ハッシュサイズ: {args.hash_size}x{args.hash_size}")
        print(f"ハッシュ値: {template_hash}")

        if args.debug:
            frame = cv2.imread(args.image)
            x1, y1, x2, y2 = DEFAULT_ROI
            roi = frame[y1:y2, x1:x2]
            cv2.imshow("ROI", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # テンプレート読み込み
    if not template_path.exists():
        print(f"Error: テンプレートが見つかりません: {template_path}")
        print("先に --create-template で作成してください")
        return
    template_hash = np.load(str(template_path))

    # テスト実行
    if args.test_all:
        run_test_all(args, template_hash)
    elif args.image:
        run_single_image(args, template_hash)
    else:
        print("Error: --image または --test-all を指定してください")


if __name__ == "__main__":
    main()
