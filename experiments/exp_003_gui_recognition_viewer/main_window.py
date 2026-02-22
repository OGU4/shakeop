"""メインウィンドウ。

UI構成:
    - 上部: コントロールパネル（カメラ選択、認識機能選択、トグルボタン、ログ設定）
    - 中央: 映像表示（QLabel、スケーリング対応）
    - 下部: ステータスバー（FPS表示）
"""

import glob
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from capture_worker import CaptureWorker
from log_writer import LogWriter
from recognition_plugin import RecognitionPlugin
from recognition_worker import RecognitionWorker

# デフォルトログ出力先
_DEFAULT_LOG_PATH = Path(__file__).parent / "output.log"


class MainWindow(QMainWindow):
    """メインウィンドウ。"""

    def __init__(self, plugins: list[RecognitionPlugin]) -> None:
        super().__init__()
        self._plugins = plugins
        self._capture_worker: CaptureWorker | None = None
        self._recognition_worker: RecognitionWorker | None = None
        self._log_writer = LogWriter(_DEFAULT_LOG_PATH)
        self._is_running = False

        self.setWindowTitle("GUI認識ビューワー")
        self.resize(960, 640)

        self._setup_ui()
        self._connect_signals()

    # --- UI構築 ---

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # コントロールパネル
        main_layout.addLayout(self._create_control_panel())

        # 映像表示
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(320, 180)
        self._video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self._video_label, stretch=1)

        # ステータスバー
        self._fps_label = QLabel("FPS: --")
        self.statusBar().addPermanentWidget(self._fps_label)

    def _create_control_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        # カメラ選択行
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("カメラ:"))
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(200)
        camera_row.addWidget(self._camera_combo)
        self._refresh_btn = QPushButton("更新")
        camera_row.addWidget(self._refresh_btn)
        camera_row.addStretch()
        layout.addLayout(camera_row)

        # 認識機能選択行
        plugin_row = QHBoxLayout()
        plugin_row.addWidget(QLabel("認識機能:"))
        self._plugin_combo = QComboBox()
        self._plugin_combo.setMinimumWidth(200)
        for plugin in self._plugins:
            self._plugin_combo.addItem(plugin.name)
        plugin_row.addWidget(self._plugin_combo)
        plugin_row.addStretch()
        layout.addLayout(plugin_row)

        # トグルボタン行
        toggle_row = QHBoxLayout()
        self._toggle_btn = QPushButton("▶ 開始")
        self._toggle_btn.setFixedWidth(100)
        toggle_row.addWidget(self._toggle_btn)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # ログ設定行
        log_row = QHBoxLayout()
        self._log_checkbox = QCheckBox("ログ出力")
        log_row.addWidget(self._log_checkbox)
        self._log_path_edit = QLineEdit(str(_DEFAULT_LOG_PATH))
        log_row.addWidget(self._log_path_edit)
        self._browse_btn = QPushButton("参照...")
        log_row.addWidget(self._browse_btn)
        layout.addLayout(log_row)

        # カメラリスト初期化
        self._refresh_cameras()

        return layout

    def _connect_signals(self) -> None:
        self._refresh_btn.clicked.connect(self._on_refresh_cameras)
        self._toggle_btn.clicked.connect(self._on_toggle_start_stop)
        self._camera_combo.activated.connect(self._on_camera_changed)
        self._plugin_combo.activated.connect(self._on_plugin_changed)
        self._log_checkbox.toggled.connect(self._on_log_toggled)
        self._browse_btn.clicked.connect(self._on_browse_log_path)

    # --- カメラデバイス列挙 ---

    def _enumerate_cameras(self) -> list[str]:
        return sorted(glob.glob("/dev/video*"))

    def _refresh_cameras(self) -> None:
        self._camera_combo.clear()
        devices = self._enumerate_cameras()
        for device in devices:
            self._camera_combo.addItem(device)

    def _on_refresh_cameras(self) -> None:
        self._refresh_cameras()

    # --- 開始/停止 ---

    def _on_toggle_start_stop(self) -> None:
        if self._is_running:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self) -> None:
        device = self._camera_combo.currentText()
        if not device:
            self._show_error("カメラデバイスが選択されていません。")
            return

        if not self._plugins:
            self._show_error("認識プラグインがありません。")
            return

        plugin_index = self._plugin_combo.currentIndex()
        plugin = self._plugins[plugin_index]

        # ワーカー生成
        self._capture_worker = CaptureWorker(device)
        self._recognition_worker = RecognitionWorker(plugin)

        # Signal/Slot接続
        self._capture_worker.frame_captured.connect(
            self._recognition_worker.on_frame_captured
        )
        self._capture_worker.error_occurred.connect(self._on_capture_error)
        self._recognition_worker.frame_ready.connect(self._on_frame_ready)

        # 起動
        self._recognition_worker.start()
        self._capture_worker.start()

        self._is_running = True
        self._toggle_btn.setText("■ 停止")
        self._camera_combo.setEnabled(False)

    def _stop_capture(self) -> None:
        if self._capture_worker is not None:
            self._capture_worker.stop()
            self._capture_worker.wait()
            self._capture_worker = None

        if self._recognition_worker is not None:
            self._recognition_worker.stop()
            self._recognition_worker.wait()
            self._recognition_worker = None

        self._is_running = False
        self._toggle_btn.setText("▶ 開始")
        self._camera_combo.setEnabled(True)
        self._fps_label.setText("FPS: --")

    # --- 動作中のカメラ切替 ---

    def _on_camera_changed(self, _index: int) -> None:
        if self._is_running:
            self._stop_capture()
            self._start_capture()

    # --- 動作中の認識機能切替 ---

    def _on_plugin_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._plugins):
            return
        new_plugin = self._plugins[index]
        if self._recognition_worker is not None:
            self._recognition_worker.set_plugin(new_plugin)

    # --- フレーム受信 ---

    @Slot(np.ndarray, dict, float)
    def _on_frame_ready(
        self, overlay_frame: np.ndarray, result: dict, fps: float
    ) -> None:
        pixmap = self._frame_to_pixmap(overlay_frame)
        self._video_label.setPixmap(pixmap)
        self._fps_label.setText(f"FPS: {fps:.1f}")

        # ログ出力
        if self._log_checkbox.isChecked():
            plugin_index = self._plugin_combo.currentIndex()
            if 0 <= plugin_index < len(self._plugins):
                self._log_writer.write(self._plugins[plugin_index], result)

    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """BGR numpy配列 → QPixmap に変換。QLabel のサイズに合わせてスケーリング。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label_size = self._video_label.size()
        return pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    # --- ログ設定 ---

    def _on_log_toggled(self, enabled: bool) -> None:
        if enabled:
            log_path = self._log_path_edit.text()
            self._log_writer = LogWriter(log_path)
            self._log_writer.open()
        else:
            self._log_writer.close()

    def _on_browse_log_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "ログ保存先を選択",
            self._log_path_edit.text(),
            "Log files (*.log);;All files (*)",
        )
        if path:
            self._log_path_edit.setText(path)
            # ログ有効中ならファイルを切り替え
            if self._log_checkbox.isChecked():
                self._log_writer.close()
                self._log_writer = LogWriter(path)
                self._log_writer.open()

    # --- エラー表示 ---

    @Slot(str)
    def _on_capture_error(self, message: str) -> None:
        self._stop_capture()
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "エラー", message)

    # --- クリーンアップ ---

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_capture()
        self._log_writer.close()
        super().closeEvent(event)
