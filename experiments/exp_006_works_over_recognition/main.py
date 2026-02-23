#!/usr/bin/env python3
"""
exp_006: "Work's Over!!" テキスト認識 (pHash 1段判定)

目的: ゲームオーバー時の "Work's Over!!" テキストをpHashで判定できるか検証
手法: ROI切り出し → グレースケール変換 → pHash (1段判定)
実行:
  # テンプレート作成
  uv run python experiments/exp_006_works_over_recognition/main.py \
      --create-template --image data/test_fixtures/works_over/positive/<画像>

  # ディレクトリ一括判定
  uv run python experiments/exp_006_works_over_recognition/main.py \
      --image-dir data/test_fixtures/works_over/ --debug
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
from works_over_recognizer import WorksOverRecognizer  # noqa: E402

# テンプレート配置先
DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "assets" / "templates" / "works_over"

# テンプレートファイル名
TEMPLATE_FILE = "works_over.npy"

# 画像ファイル拡張子
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def create_recognizer(template_dir: Path, threshold: int) -> WorksOverRecognizer:
    """テンプレートを読み込んでRecognizerを生成する"""
    template_path = template_dir / TEMPLATE_FILE
    if not template_path.exists():
        raise FileNotFoundError(
            f"テンプレートが見つかりません: {template_path}\n"
            f"先に --create-template で作成してください"
        )

    works_over_hash = np.load(str(template_path))
    return WorksOverRecognizer(
        works_over_hash=works_over_hash,
        threshold=threshold,
    )


def run_create_template(args: argparse.Namespace) -> None:
    """テンプレート作成モード"""
    template_dir = Path(args.template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: 画像を読み込めません: {args.image}")
        return

    # ROI切り出し → グレースケール変換
    x1, y1, x2, y2 = WorksOverRecognizer.WORKS_OVER_ROI
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # pHash計算 & 保存
    works_over_hash = compute_phash(gray)
    save_path = template_dir / TEMPLATE_FILE
    np.save(str(save_path), works_over_hash)
    print(f"テンプレート保存: {save_path}")
    print(f"  Work's Over!! ハッシュ: {works_over_hash.flatten()}")

    if args.debug:
        print("\n--- デバッグ情報 ---")
        print(f"画像: {args.image}")
        print(f"ROI座標: {WorksOverRecognizer.WORKS_OVER_ROI}")
        print(f"ROIサイズ: {roi.shape[1]}x{roi.shape[0]}")

        # デバッグ画像をファイルに保存
        debug_dir = Path(__file__).parent / "debug_output"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "roi_bgr.png"), roi)
        cv2.imwrite(str(debug_dir / "roi_gray.png"), gray)

        # ROI位置を元画像に描画して保存
        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "frame_with_roi.png"), frame_with_roi)

        print(f"\nデバッグ画像を保存しました: {debug_dir}/")
        print("  frame_with_roi.png  — ROI矩形(緑)を元画像に描画")
        print("  roi_bgr.png         — ROI切り出し (カラー)")
        print("  roi_gray.png        — ROIグレースケール")


def get_expected_result(filepath: Path) -> str | None:
    """ファイルパスから正解ラベルを推定する（ディレクトリ名ベース）"""
    parent = filepath.parent.name
    mapping = {
        "positive": "WORKS_OVER",
        "negative": "NONE",
    }
    return mapping.get(parent)


def run_image_dir(args: argparse.Namespace) -> None:
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
        f"{'ファイル':<50} {'判定':<12} {'期待':<12} {'信頼度':<10} {'時間(ms)':<10} {'正否'}"
    )
    print("-" * 110)

    for r in results:
        status = ""
        if r["correct"] is not None:
            status = "OK" if r["correct"] else "NG"
        expected_str = r["expected"] or "-"
        print(
            f"{str(r['filename']):<50} "
            f"{r['result']:<12} "
            f"{expected_str:<12} "
            f"{r['confidence']:<10.4f} "
            f"{r['elapsed_ms']:<10.2f} "
            f"{status}"
        )

    # デバッグ: ハミング距離の分布を表示
    if args.debug:
        print("\n=== ハミング距離分布 ===\n")
        for category in ["positive", "negative"]:
            cat_results = [
                r for r in results if r["path"].parent.name == category and r["debug_info"]
            ]
            if cat_results:
                distances = [r["debug_info"]["distance"] for r in cat_results]
                print(f"  {category}:")
                print(f"    件数: {len(distances)}")
                print(f"    最小: {min(distances)}")
                print(f"    最大: {max(distances)}")
                print(f"    平均: {sum(distances) / len(distances):.1f}")
                print()

        # NG画像の詳細
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
    result_file = results_dir / f"{timestamp}_works_over_result.txt"

    with open(result_file, "w") as f:
        f.write("# F-006 Work's Over!! Recognition Result\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Image dir: {image_dir.resolve()}\n")
        f.write(f"# Threshold: {args.threshold}\n")
        f.write("filename\tresult\tconfidence\n")
        for r in results:
            f.write(f"{r['filename']}\t{r['result']}\t{r['confidence']:.4f}\n")

    print(f"\n結果ファイル: {result_file}")


def run_single_image(args: argparse.Namespace) -> None:
    """単一画像判定モード"""
    template_dir = Path(args.template_dir)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: 画像を読み込めません: {args.image}")
        return

    recognizer = create_recognizer(template_dir, args.threshold)

    start = time.perf_counter()
    if args.debug:
        debug_info = recognizer.recognize_debug(frame)
        result = debug_info["result"]
        confidence = debug_info["confidence"]
    else:
        result, confidence = recognizer.recognize(frame)
        debug_info = None
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"画像: {args.image}")
    print(f"判定: {result}")
    print(f"信頼度: {confidence:.4f}")
    print(f"処理時間: {elapsed_ms:.2f}ms")

    if args.debug and debug_info:
        print(f"\n--- デバッグ情報 ---")
        print(f"ハミング距離: {debug_info['distance']}")
        print(f"閾値: {args.threshold}")

        debug_dir = Path(__file__).parent / "debug_output"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "roi_bgr.png"), debug_info["roi"])
        cv2.imwrite(str(debug_dir / "roi_gray.png"), debug_info["gray"])
        print(f"\nデバッグ画像を保存しました: {debug_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='exp_006: "Work\'s Over!!" テキスト認識 (pHash 1段判定)'
    )

    # テンプレート作成モード
    parser.add_argument(
        "--create-template", action="store_true", help="テンプレート作成モード"
    )
    parser.add_argument("--image", type=str, help="単一画像パス（テンプレート作成時または判定時）")

    # 判定モード
    parser.add_argument("--image-dir", type=str, help="判定対象の画像ディレクトリパス")

    # カメラモード
    parser.add_argument("--camera", type=int, help="カメラデバイス番号")

    # 共通オプション
    parser.add_argument(
        "--template-dir",
        type=str,
        default=str(DEFAULT_TEMPLATE_DIR),
        help=f"テンプレートの配置ディレクトリ (default: {DEFAULT_TEMPLATE_DIR})",
    )
    parser.add_argument(
        "--threshold", type=int, default=50, help="ハミング距離の閾値 (default: 50)"
    )
    parser.add_argument("--debug", action="store_true", help="ROI可視化・詳細情報表示")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.create_template:
        if not args.image:
            print("Error: --create-template には --image が必要です")
            return
        run_create_template(args)

    elif args.image_dir:
        run_image_dir(args)

    elif args.image:
        run_single_image(args)

    else:
        print('Error: --create-template, --image-dir, または --image を指定してください')
        print()
        print("テンプレート作成:")
        print("  uv run python experiments/exp_006_works_over_recognition/main.py \\")
        print("      --create-template --image <Work's Over!!表示の画像> --debug")
        print()
        print("単一画像判定:")
        print("  uv run python experiments/exp_006_works_over_recognition/main.py \\")
        print("      --image <判定対象画像>")
        print()
        print("ディレクトリ一括判定:")
        print("  uv run python experiments/exp_006_works_over_recognition/main.py \\")
        print("      --image-dir data/test_fixtures/works_over/ --debug")


if __name__ == "__main__":
    main()
