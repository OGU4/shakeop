#!/usr/bin/env python3
"""exp_003: GUI認識ビューワー

エントリーポイント。QApplicationの起動とプラグインの登録を行う。

実行: uv run python experiments/exp_003_gui_recognition_viewer/main.py
"""

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from main_window import MainWindow


def _load_plugins() -> list:
    """利用可能なプラグインをロードする。"""
    plugins = []
    try:
        from plugins.baito_text import BaitoTextPlugin

        plugins.append(BaitoTextPlugin())
    except Exception as e:
        print(f"Warning: BaitoTextPlugin の読み込みに失敗: {e}")
    return plugins


def main() -> None:
    app = QApplication(sys.argv)

    plugins = _load_plugins()
    if not plugins:
        QMessageBox.critical(
            None,
            "エラー",
            "認識プラグインが1つも読み込めませんでした。\n"
            "テンプレートファイルの存在を確認してください。",
        )
        sys.exit(1)

    window = MainWindow(plugins=plugins)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
