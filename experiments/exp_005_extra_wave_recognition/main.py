#!/usr/bin/env python3
"""
exp_005: Extra Wave判定 (pHash + HSV白色抽出 + ROI全体1段判定)

目的: ゲーム画面の「EXTRA WAVE」表示をpHashで判定できるか検証
手法: ROI切り出し → HSV白色テキスト抽出 → pHash (1段判定)
実行:
  # テンプレート作成
  uv run python experiments/exp_005_extra_wave_recognition/main.py \
      --create-template --image <EXTRA WAVE表示の画像>

  # ディレクトリ一括判定
  uv run python experiments/exp_005_extra_wave_recognition/main.py \
      --image-dir data/test_fixtures/wave/ --debug
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# プロジェクトルートをパスに追加
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.recognition import compute_phash  # noqa: E402
from extra_wave_recognizer import ExtraWaveRecognizer  # noqa: E402

# テンプレート配置先
DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "assets" / "templates" / "wave"

# テンプレートファイル名
TEMPLATE_FILE = "extra_wave.npy"

# 画像ファイル拡張子
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def create_recognizer(template_dir: Path, threshold: int) -> ExtraWaveRecognizer:
    """テンプレートを読み込んでRecognizerを生成する"""
    template_path = template_dir / TEMPLATE_FILE
    if not template_path.exists():
        raise FileNotFoundError(
            f"テンプレートが見つかりません: {template_path}\n"
            f"先に --create-template で作成してください"
        )

    extra_wave_hash = np.load(str(template_path))
    return ExtraWaveRecognizer(
        extra_wave_hash=extra_wave_hash,
        threshold=threshold,
    )


def run_create_template(args):
    """テンプレート作成モード"""
    template_dir = Path(args.template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: 画像を読み込めません: {args.image}")
        return

    # ROI切り出し & HSV前処理 + モルフォロジーノイズ除去（認識時と同一処理）
    x1, y1, x2, y2 = ExtraWaveRecognizer.EXTRA_WAVE_ROI
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, ExtraWaveRecognizer.HSV_LOWER, ExtraWaveRecognizer.HSV_UPPER
    )
    binary = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ExtraWaveRecognizer.MORPH_KERNEL)

    # pHash計算 & 保存
    extra_wave_hash = compute_phash(binary)
    save_path = template_dir / TEMPLATE_FILE
    np.save(str(save_path), extra_wave_hash)
    print(f"テンプレート保存: {save_path}")
    print(f"  EXTRA WAVEハッシュ: {extra_wave_hash.flatten()}")

    if args.debug:
        print("\n--- デバッグ情報 ---")
        print(f"画像: {args.image}")
        print(f"ROI座標: {ExtraWaveRecognizer.EXTRA_WAVE_ROI}")
        print(f"ROIサイズ: {roi.shape[1]}x{roi.shape[0]}")
        print(f"二値画像の白ピクセル割合: {np.count_nonzero(binary) / binary.size:.1%}")

        # デバッグ画像をファイルに保存
        debug_dir = Path(__file__).parent / "debug_output"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "roi_bgr.png"), roi)
        cv2.imwrite(str(debug_dir / "roi_binary.png"), binary)

        # ROI位置を元画像に描画して保存
        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "frame_with_roi.png"), frame_with_roi)

        print(f"\nデバッグ画像を保存しました: {debug_dir}/")
        print("  frame_with_roi.png  — ROI矩形(緑)を元画像に描画")
        print("  roi_bgr.png         — ROI切り出し (カラー)")
        print("  roi_binary.png      — ROI二値化")


def get_expected_result(filepath: Path) -> str | None:
    """ファイルパスから正解ラベルを推定する（ディレクトリ名ベース）"""
    parent = filepath.parent.name
    mapping = {
        "extra": "EXTRA",
        "wave1": "NONE",
        "wave2": "NONE",
        "wave3": "NONE",
        "wave4": "NONE",
        "wave5": "NONE",
        "negative": "NONE",
    }
    return mapping.get(parent)


def run_image_dir(args):
    """ディレクトリ一括判定モード"""
    template_dir = Path(args.template_dir)
    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        print(f"Error: ディレクトリが見つかりません: {image_dir}")
        return

    recognizer = create_recognizer(template_dir, args.threshold)

    # 全画像ファイルを収集（サブディレクトリも含む）
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(image_dir.rglob(f"*{ext}"))
    image_files.sort()

    if not image_files:
        print(f"Error: 画像ファイルが見つかりません: {image_dir}")
        return

    print(f"画像ディレクトリ: {image_dir}")
    print(f"画像ファイル数: {len(image_files)}")
    print(f"テンプレート: {template_dir}")
    print(f"閾値: {args.threshold}")
    print()

    # 判定実行
    results = []
    total_time = 0.0

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: 読み込めません: {img_path}")
            continue

        start = time.perf_counter()
        if args.debug:
            debug_info = recognizer.recognize_debug(frame)
            result = debug_info["result"]
            confidence = debug_info["confidence"]
        else:
            result, confidence = recognizer.recognize(frame)
            debug_info = None
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_time += elapsed_ms

        expected = get_expected_result(img_path)
        correct = (expected == result) if expected is not None else None

        results.append(
            {
                "path": img_path,
                "filename": img_path.relative_to(image_dir),
                "result": result,
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
                "expected": expected,
                "correct": correct,
                "debug_info": debug_info,
            }
        )

    # コンソール出力
    print(
        f"{'ファイル':<50} {'判定':<10} {'期待':<10} {'信頼度':<10} {'時間(ms)':<10} {'正否'}"
    )
    print("-" * 105)

    for r in results:
        status = ""
        if r["correct"] is not None:
            status = "OK" if r["correct"] else "NG"
        expected_str = r["expected"] or "-"
        print(
            f"{str(r['filename']):<50} "
            f"{r['result']:<10} "
            f"{expected_str:<10} "
            f"{r['confidence']:<10.4f} "
            f"{r['elapsed_ms']:<10.2f} "
            f"{status}"
        )

    # デバッグ: NG画像の詳細情報
    if args.debug:
        ng_results = [r for r in results if r["correct"] is False]
        if ng_results:
            print(f"\n=== NG画像の詳細 ({len(ng_results)}枚) ===\n")
            for r in ng_results:
                info = r["debug_info"]
                print(f"  {r['filename']}")
                print(f"    判定: {r['result']} (期待: {r['expected']})")
                print(f"    距離: {info['distance']}")
                print()

    # 精度サマリー
    print()
    print("=" * 60)
    print("精度サマリー")
    print("=" * 60)

    categories = {}
    for r in results:
        cat = r["path"].parent.name
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["correct"] is True:
            categories[cat]["correct"] += 1

    total_correct = 0
    total_count = 0
    for cat in sorted(categories.keys()):
        c = categories[cat]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        total_correct += c["correct"]
        total_count += c["total"]
        print(f"  {cat:<12}: {c['correct']}/{c['total']} ({acc:.0%})")

    if total_count > 0:
        overall_acc = total_correct / total_count
        avg_time = total_time / len(results)
        print(f"  {'総合':<12}: {total_correct}/{total_count} ({overall_acc:.0%})")
        print(f"  平均処理時間: {avg_time:.2f}ms")

    # 結果ファイル出力
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    result_file = results_dir / f"{timestamp}_extra_wave_result.txt"

    with open(result_file, "w") as f:
        f.write("# F-005 Extra Wave Recognition Result\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Image dir: {image_dir.resolve()}\n")
        f.write(f"# Threshold: {args.threshold}\n")
        f.write("filename\tresult\tconfidence\n")
        for r in results:
            f.write(f"{r['filename']}\t{r['result']}\t{r['confidence']:.4f}\n")

    print(f"\n結果ファイル: {result_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="exp_005: Extra Wave判定 (pHash + HSV白色抽出 + ROI全体1段判定)"
    )

    # テンプレート作成モード
    parser.add_argument(
        "--create-template", action="store_true", help="テンプレート作成モード"
    )
    parser.add_argument("--image", type=str, help="テンプレート作成時のソース画像パス")

    # 判定モード
    parser.add_argument("--image-dir", type=str, help="判定対象の画像ディレクトリパス")

    # 共通オプション
    parser.add_argument(
        "--template-dir",
        type=str,
        default=str(DEFAULT_TEMPLATE_DIR),
        help=f"テンプレートの配置ディレクトリ (default: {DEFAULT_TEMPLATE_DIR})",
    )
    parser.add_argument(
        "--threshold", type=int, default=110, help="ハミング距離の閾値 (default: 110)"
    )
    parser.add_argument("--debug", action="store_true", help="ROI可視化・詳細情報表示")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.create_template:
        if not args.image:
            print("Error: --create-template には --image が必要です")
            return
        run_create_template(args)

    elif args.image_dir:
        run_image_dir(args)

    else:
        print("Error: --create-template または --image-dir を指定してください")
        print()
        print("テンプレート作成:")
        print("  uv run python experiments/exp_005_extra_wave_recognition/main.py \\")
        print("      --create-template --image <EXTRA WAVE表示の画像>")
        print()
        print("ディレクトリ一括判定:")
        print("  uv run python experiments/exp_005_extra_wave_recognition/main.py \\")
        print("      --image-dir data/test_fixtures/wave/")


if __name__ == "__main__":
    main()
