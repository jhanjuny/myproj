"""
Processing dialog: shows progress while AI generates the note.
Streams text into a preview pane in real time.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTextEdit, QListWidget, QListWidgetItem,
)
from PyQt5.QtGui import QColor


class ProcessingDialog(QDialog):
    """Modal dialog shown during note generation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 노트 생성 중...")
        self.setModal(True)
        self.setMinimumSize(700, 500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Status label
        self._status = QLabel("준비 중...")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet("font-size: 14px; color: #7aa2f7;")
        layout.addWidget(self._status)

        # Progress bar (indeterminate)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        layout.addWidget(self._progress)

        # Error list
        self._error_list = QListWidget()
        self._error_list.setMaximumHeight(80)
        self._error_list.setVisible(False)
        layout.addWidget(self._error_list)

        # Live preview
        preview_label = QLabel("노트 미리보기 (실시간 스트리밍):")
        layout.addWidget(preview_label)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet("""
            QTextEdit {
                background: #16161e;
                color: #c0caf5;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self._preview)

        # Close button (hidden until done)
        self._close_btn = QPushButton("닫기")
        self._close_btn.setVisible(False)
        self._close_btn.clicked.connect(self.accept)
        layout.addWidget(self._close_btn)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_status(self, text: str):
        self._status.setText(text)

    def append_chunk(self, chunk: str):
        self._preview.moveCursor(self._preview.textCursor().End)
        self._preview.insertPlainText(chunk)
        self._preview.moveCursor(self._preview.textCursor().End)

    def add_file_error(self, filename: str, error: str):
        if not self._error_list.isVisible():
            self._error_list.setVisible(True)
        item = QListWidgetItem(f"⚠️  {filename}: {error.splitlines()[0]}")
        item.setToolTip(error)
        item.setForeground(QColor("#f7768e"))
        self._error_list.addItem(item)

    def mark_done(self):
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._status.setText("✅ 노트 생성 완료!")
        self._status.setStyleSheet("font-size: 14px; color: #9ece6a;")
        self._close_btn.setVisible(True)

    def mark_error(self, message: str):
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status.setText(f"❌ 오류 발생")
        self._status.setStyleSheet("font-size: 14px; color: #f7768e;")
        self._preview.append(f"\n\n[오류]\n{message}")
        self._close_btn.setVisible(True)
