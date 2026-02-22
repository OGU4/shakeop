"""認識結果をテキストファイルに出力する。"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recognition_plugin import RecognitionPlugin


class LogWriter:
    """認識結果をテキストファイルに出力する。"""

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._file: object = None

    def open(self) -> None:
        """ログファイルを追記モードで開く。"""
        self._file = open(self._file_path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, plugin: RecognitionPlugin, result: dict) -> None:
        """タイムスタンプ付きで認識結果を1行出力する。"""
        if self._file is None:
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = plugin.format_log(result)
        self._file.write(f"{timestamp}  {log_line}\n")  # type: ignore[union-attr]
        self._file.flush()  # type: ignore[union-attr]

    def close(self) -> None:
        """ログファイルを閉じる。"""
        if self._file is not None:
            self._file.close()  # type: ignore[union-attr]
            self._file = None
