"""pHash (パーセプチュアルハッシュ) ユーティリティ。

DCTベースの16x16 pHash計算とハミング距離算出を提供する。
WaveNumberRecognizer (F-004) から切り出した共通関数。
"""

import cv2
import numpy as np


def compute_phash(image: np.ndarray) -> np.ndarray:
    """16x16 pHashを計算する (256bit, DCTベース)

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


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """2つのハッシュ間のハミング距離を算出する"""
    return int(np.unpackbits(np.bitwise_xor(hash1, hash2)).sum())
