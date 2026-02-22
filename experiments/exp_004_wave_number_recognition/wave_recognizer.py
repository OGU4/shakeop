"""
WaveNumberRecognizer: Wave数のpHash認識器（ROI分割方式）

設計書: docs/issues/F-004_wave_number_recognition/design.md

認識フロー:
1. ROI切り出し (38, 35, 199, 80) → 161x45 px
2. HSV白色テキスト抽出 → 二値画像
3. 左128pxで「WAVE」テキスト判定 (pHash 16x16)
4. 一致 → 右33pxで数字(1~5)判定 (pHash 16x16)

※ EXTRA WAVE の認識は F-005 に分離
"""

import cv2
import numpy as np


class WaveNumberRecognizer:
    """Wave数のpHash認識器（ROI分割方式）"""

    # ROI座標 (FHD 1920x1080 基準)
    WAVE_ROI = (76, 35, 199, 80)  # (x1, y1, x2, y2) 全体: 123x45 px
    WAVE_TEXT_WIDTH = 90  # 左側「WAVE」テキスト部の幅
    WAVE_DIGIT_OFFSET = 90  # 右側 数字部の開始位置

    # HSVフィルタ（白色テキスト抽出）
    # V≥210: 霧イベントの背景ノイズ(V≈200-210)を除外しつつテキストを残す
    HSV_LOWER = np.array([0, 0, 210])
    HSV_UPPER = np.array([180, 80, 255])

    # モルフォロジー演算カーネル（ノイズ除去用）
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # pHashの最大ハミング距離 (16x16 = 256bit)
    MAX_DISTANCE = 256

    def __init__(
        self,
        wave_text_hash: np.ndarray,
        digit_hashes: dict[int, np.ndarray],
        threshold: int = 116,
    ):
        """
        Args:
            wave_text_hash: 「WAVE」テキスト部のテンプレートハッシュ
            digit_hashes: 数字1~5のテンプレートハッシュ辞書 {1: hash, ..., 5: hash}
            threshold: ハミング距離の閾値（これ以下なら一致と判定）
        """
        self.wave_text_hash = wave_text_hash
        self.digit_hashes = digit_hashes
        self.threshold = threshold

    def recognize(self, frame: np.ndarray) -> tuple[str, float]:
        """
        フレームからWave数を判定する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            (判定結果, 信頼度 0.0-1.0)
            判定結果: "WAVE_1" ~ "WAVE_5", "NONE"
        """
        roi = self._extract_roi(frame)
        binary = self._preprocess(roi)

        # Stage 1: 「WAVE」テキスト判定
        wave_text_region = self._split_wave_text(binary)
        wave_text_phash = self.compute_phash(wave_text_region)
        wave_text_dist = self.hamming_distance(wave_text_phash, self.wave_text_hash)

        if wave_text_dist <= self.threshold:
            # Stage 2: 数字判定 (1~5)
            digit_region = self._split_digit(binary)
            digit_phash = self.compute_phash(digit_region)

            best_digit = -1
            best_dist = self.MAX_DISTANCE + 1

            for digit, digit_hash in self.digit_hashes.items():
                dist = self.hamming_distance(digit_phash, digit_hash)
                if dist < best_dist:
                    best_dist = dist
                    best_digit = digit

            if best_dist <= self.threshold:
                confidence = 1.0 - (best_dist / self.MAX_DISTANCE)
                return f"WAVE_{best_digit}", confidence
            else:
                # WAVEテキストは検出されたが数字が判定不能
                confidence = 1.0 - (best_dist / self.MAX_DISTANCE)
                return "NONE", confidence
        else:
            # WAVEテキスト非検出 → Wave表示なし
            confidence = 1.0 - (wave_text_dist / self.MAX_DISTANCE)
            return "NONE", confidence

    def recognize_debug(self, frame: np.ndarray) -> dict:
        """
        デバッグ用の詳細情報付き認識。

        Returns:
            {
                "result": str,
                "confidence": float,
                "wave_text_distance": int,
                "digit_distances": dict[int, int],
                "roi": np.ndarray,
                "binary": np.ndarray,
                "wave_text_region": np.ndarray,
                "digit_region": np.ndarray,
            }
        """
        roi = self._extract_roi(frame)
        binary = self._preprocess(roi)

        wave_text_region = self._split_wave_text(binary)
        wave_text_phash = self.compute_phash(wave_text_region)
        wave_text_dist = self.hamming_distance(wave_text_phash, self.wave_text_hash)

        digit_region = self._split_digit(binary)
        digit_phash = self.compute_phash(digit_region)
        digit_distances = {}
        for digit, digit_hash in self.digit_hashes.items():
            digit_distances[digit] = self.hamming_distance(digit_phash, digit_hash)

        result, confidence = self.recognize(frame)

        return {
            "result": result,
            "confidence": confidence,
            "wave_text_distance": wave_text_dist,
            "digit_distances": digit_distances,
            "roi": roi,
            "binary": binary,
            "wave_text_region": wave_text_region,
            "digit_region": digit_region,
        }

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI領域を切り出す (161x45 BGR)"""
        x1, y1, x2, y2 = self.WAVE_ROI
        return frame[y1:y2, x1:x2]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """HSVフィルタで白色テキストを抽出し、モルフォロジー演算でノイズ除去した二値画像を返す"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.HSV_LOWER, self.HSV_UPPER)
        # モルフォロジーオープニング: 小さなノイズ点を除去（背景の映り込み対策）
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.MORPH_KERNEL)
        return cleaned

    def _split_wave_text(self, binary: np.ndarray) -> np.ndarray:
        """二値画像から左128pxの「WAVE」テキスト部を取得"""
        return binary[:, : self.WAVE_TEXT_WIDTH]

    def _split_digit(self, binary: np.ndarray) -> np.ndarray:
        """二値画像から右33pxの数字部を取得"""
        return binary[:, self.WAVE_DIGIT_OFFSET :]

    @staticmethod
    def compute_phash(image: np.ndarray) -> np.ndarray:
        """16x16 pHashを計算する (256bit)

        32x32にリサイズ → DCT → 低周波16x16係数 → 平均値で二値化 → 256bitハッシュ
        """
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        float_img = np.float32(resized)
        dct = cv2.dct(float_img)
        dct_low = dct[:16, :16]
        dct_flat = dct_low.flatten()
        # DC成分(index 0)を除いた平均値で閾値を決定（OpenCV PHashと同じ手法）
        mean_val = np.mean(dct_flat[1:])
        hash_bits = (dct_flat > mean_val).astype(np.uint8)
        return np.packbits(hash_bits)  # 256bit = 32bytes

    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        """2つのハッシュ間のハミング距離を算出する"""
        return int(np.unpackbits(np.bitwise_xor(hash1, hash2)).sum())
