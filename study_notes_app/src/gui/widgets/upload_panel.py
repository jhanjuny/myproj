"""
File upload panel with drag-and-drop support.
Shows queued files, subject/chapter selection, and the Generate button.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QFrame, QComboBox, QFileDialog, QSizePolicy,
)

from src.config import SUPPORTED_EXTENSIONS
from src.storage import database as db


class UploadPanel(QWidget):
    """
    Collects file paths + destination (subject/chapter),
    then emits generate_requested when the user clicks Generate.
    """

    generate_requested = pyqtSignal(list, object, object)
    # args: (file_paths: list[Path], subject_id: int|None, chapter_id: int|None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files: list[Path] = []
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Title
        title = QLabel("📂 파일 업로드")
        title.setObjectName("title")
        layout.addWidget(title)

        # Drop zone
        self._drop_zone = _DropZone()
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self._drop_zone)

        # Browse button
        browse_btn = QPushButton("파일 선택...")
        browse_btn.setObjectName("secondary")
        browse_btn.clicked.connect(self._browse_files)
        layout.addWidget(browse_btn)

        # File list
        list_label = QLabel("추가된 파일:")
        layout.addWidget(list_label)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(140)
        layout.addWidget(self._file_list)

        # Clear list button
        clear_btn = QPushButton("목록 지우기")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self._clear_files)
        layout.addWidget(clear_btn)

        # Destination subject
        layout.addWidget(QLabel("저장 위치:"))

        subj_row = QHBoxLayout()
        subj_row.addWidget(QLabel("과목:"))
        self._subject_combo = QComboBox()
        self._subject_combo.setEditable(True)
        self._subject_combo.setPlaceholderText("자동 분류 (AI 추론)")
        self._subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        subj_row.addWidget(self._subject_combo)
        layout.addLayout(subj_row)

        chap_row = QHBoxLayout()
        chap_row.addWidget(QLabel("단원:"))
        self._chapter_combo = QComboBox()
        self._chapter_combo.setEditable(True)
        self._chapter_combo.setPlaceholderText("자동 분류 (AI 추론)")
        chap_row.addWidget(self._chapter_combo)
        layout.addLayout(chap_row)

        self._refresh_combos()

        layout.addStretch()

        # Generate button
        self._gen_btn = QPushButton("🤖  AI 노트 생성")
        self._gen_btn.setFixedHeight(44)
        self._gen_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._gen_btn)

        hint = QLabel("지원 형식: PDF · DOCX · PPTX · MP4 · 이미지(손필기)")
        hint.setObjectName("subtitle")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # ── Combos ────────────────────────────────────────────────────────────────

    def _refresh_combos(self):
        self._subject_combo.clear()
        self._subject_combo.addItem("(AI 자동 분류)", None)
        for s in db.get_subjects():
            self._subject_combo.addItem(s["name"], s["id"])

        self._chapter_combo.clear()
        self._chapter_combo.addItem("(AI 자동 분류)", None)

    def _on_subject_changed(self, idx):
        subject_id = self._subject_combo.itemData(idx)
        self._chapter_combo.clear()
        self._chapter_combo.addItem("(AI 자동 분류)", None)
        if subject_id is not None:
            for c in db.get_chapters(subject_id):
                self._chapter_combo.addItem(c["title"], c["id"])

    def refresh_subjects(self):
        self._refresh_combos()

    # ── File handling ─────────────────────────────────────────────────────────

    def _on_files_dropped(self, paths: list[Path]):
        self._add_files(paths)

    def _browse_files(self):
        ext_filter = "강의 자료 (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "파일 선택", "", f"{ext_filter};;모든 파일 (*)"
        )
        self._add_files([Path(p) for p in paths])

    def _add_files(self, paths: list[Path]):
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                item = QListWidgetItem(f"  {_file_icon(p)}  {p.name}")
                item.setToolTip(str(p))
                self._file_list.addItem(item)

    def _clear_files(self):
        self._files.clear()
        self._file_list.clear()

    # ── Generate ──────────────────────────────────────────────────────────────

    def _on_generate(self):
        if not self._files:
            return

        subject_id = self._subject_combo.currentData()
        chapter_id = self._chapter_combo.currentData()

        # If user typed a new subject name without selecting from list
        if subject_id is None and self._subject_combo.currentText() not in (
            "", "(AI 자동 분류)"
        ):
            name = self._subject_combo.currentText().strip()
            if name:
                subject_id = db.create_subject(name)
                chapter_id = None

        self.generate_requested.emit(list(self._files), subject_id, chapter_id)


def _file_icon(path: Path) -> str:
    ext = path.suffix.lower()
    icons = {".pdf": "📕", ".docx": "📘", ".doc": "📘",
             ".pptx": "📙", ".ppt": "📙",
             ".mp4": "🎬", ".mov": "🎬", ".avi": "🎬", ".mkv": "🎬",
             ".png": "🖼️", ".jpg": "🖼️", ".jpeg": "🖼️"}
    return icons.get(ext, "📄")


# ── Drop zone widget ──────────────────────────────────────────────────────────

class _DropZone(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)
        self.setFixedHeight(100)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        lbl = QLabel("여기에 파일을 드래그하세요")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #7aa2f7; font-size: 14px;")
        layout.addWidget(lbl)
        sublbl = QLabel("PDF · DOCX · PPTX · MP4 · 이미지")
        sublbl.setAlignment(Qt.AlignCenter)
        sublbl.setObjectName("subtitle")
        layout.addWidget(sublbl)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls()
                 if Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS]
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()
