"""
NoteEditor: side panel for editing the raw markdown of a note.
Shows a plain-text editor that syncs to the NoteViewer on save.
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QPlainTextEdit, QLabel,
)


class NoteEditorPanel(QWidget):
    """
    Markdown editor panel.
    save_requested is emitted with the new markdown text.
    discard_requested hides the panel.
    """

    save_requested = pyqtSignal(str)
    discard_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("✏️  노트 편집")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        discard_btn = QPushButton("취소")
        discard_btn.setObjectName("secondary")
        discard_btn.clicked.connect(self.discard_requested.emit)
        header.addWidget(discard_btn)

        save_btn = QPushButton("저장")
        save_btn.clicked.connect(self._on_save)
        header.addWidget(save_btn)

        layout.addLayout(header)

        hint = QLabel("LaTeX: 인라인 $...$ · 블록 $$...$$")
        hint.setObjectName("subtitle")
        layout.addWidget(hint)

        self._editor = QPlainTextEdit()
        self._editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._editor.setPlaceholderText("마크다운으로 노트를 편집하세요...")
        self._editor.setStyleSheet("""
            QPlainTextEdit {
                font-family: 'Consolas', 'D2Coding', monospace;
                font-size: 13px;
                line-height: 1.6;
            }
        """)
        layout.addWidget(self._editor)

    def set_markdown(self, markdown: str):
        self._editor.setPlainText(markdown)
        cursor = self._editor.textCursor()
        cursor.movePosition(cursor.Start)
        self._editor.setTextCursor(cursor)

    def get_markdown(self) -> str:
        return self._editor.toPlainText()

    def _on_save(self):
        self.save_requested.emit(self.get_markdown())
