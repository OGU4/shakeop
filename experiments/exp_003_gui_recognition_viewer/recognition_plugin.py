"""認識プラグインのProtocol定義。"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RecognitionPlugin(Protocol):
    """認識プラグインのインターフェース。

    各プラグインはこのProtocolを満たすクラスとして実装する。
    """

    @property
    def name(self) -> str:
        """GUI表示用のプラグイン名。"""
        ...

    def process(self, frame: np.ndarray) -> dict:
        """認識処理を実行する。

        Args:
            frame: BGR画像 (1920x1080)
        Returns:
            認識結果の辞書。共通キー:
                "detected" (bool): 検出されたか
                "confidence" (float): 信頼度 0.0-1.0
        """
        ...

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """認識結果をフレーム上にオーバーレイ描画する。

        Args:
            frame: BGR画像（元フレーム。書き換えず、コピーを返すこと）
            result: process() の戻り値
        Returns:
            描画済みフレーム（元フレームのコピー）
        """
        ...

    def format_log(self, result: dict) -> str:
        """認識結果をログ出力用文字列にフォーマットする。

        Args:
            result: process() の戻り値
        Returns:
            ログ1行分の文字列（タイムスタンプは呼び出し側で付与）
        """
        ...
